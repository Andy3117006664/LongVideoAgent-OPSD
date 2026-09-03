#!/usr/bin/env python3
"""
Convert LongTVQA / LongTVQA+ question files into VERL-compatible GRPO parquet.

This converter is "parquet-only":
  - train.parquet
  - val.parquet

Key behavior:
1) initial clip is loaded from grounding cache JSON produced by
   src/dataset/build_grounding_cache.py, aligned by question text.
2) subtitle assets are loaded from one subtitles directory:
   - LongTVQA_plus_subtitle_clip_level.json
   - LongTVQA_plus_subtitle_episode_level.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset


PROMPT_TEMPLATE = """You must follow these rules in every turn:
Reasoning First: Start by conducting your reasoning inside <reasoning>...</reasoning>. This is where you analyze the current information and decide your next step.
Choose One Action: After reasoning, you must choose exactly one of the following three actions:

Action A: If the current information is insufficient or you are somewhat uncertain, and you need to search for visual information on the current located <clipX>, then search for visual information. If your reasoning indicates that you lack necessary visual knowledge, you can call a visual engine. To do this, use the following format: <search>query</search>

Action B: If the current information is insufficient or you are somewhat uncertain, and you cannot obtain the final answer from the previous location and its possible visual information, then you need to call the grounding agent again for relocation, and output in the <request_grounding> format.

Action C: Provide the Final Answer
If your reasoning indicates that you have enough information to answer, provide the final answer inside <answer>...</answer>.
The answer must be concise and direct, in the format <answer>ax</answer>, where 'x' is the index of the selected option (e.g., <answer>a1</answer> for option a1).

question: {question}
a0: {a0}
a1: {a1}
a2: {a2}
a3: {a3}
a4: {a4}
<information>subtitles: {initial_clip_subtitle}</information>
"""


def _load_json_obj(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Questions file must be a JSON list: {path}")
            return [x for x in data if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass

    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    if not rows:
        raise ValueError(f"Questions file is empty or invalid JSON/JSONL: {path}")
    return rows


def _normalize_question_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _stable_sample_uid(item: Dict[str, Any], fallback_index: int) -> str:
    """Return a split-stable identifier without using answer/clip leakage.

    LongTVQA(+)'s source rows normally contain ``qid``.  A deterministic hash
    fallback keeps custom JSON/JSONL inputs usable while avoiding Python's
    process-randomized ``hash()``.
    """

    qid = item.get("qid")
    if qid is not None and str(qid).strip():
        return f"qid:{str(qid).strip()}"
    raw = "\x1f".join(
        [
            str(item.get("video_id") or item.get("vid_name") or item.get("occur_clip") or ""),
            str(item.get("question") or item.get("q") or ""),
            str(fallback_index),
        ]
    )
    return "row:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _parse_answer_to_idx(answer_value: Any) -> Optional[int]:
    if isinstance(answer_value, int):
        return answer_value if 0 <= answer_value <= 4 else None
    if isinstance(answer_value, str):
        s = answer_value.strip().lower()
        if s.startswith("a") and s[1:].isdigit():
            idx = int(s[1:])
            return idx if 0 <= idx <= 4 else None
        if s.isdigit():
            idx = int(s)
            return idx if 0 <= idx <= 4 else None
    return None


def _shuffle_options(
    answer: str,
    wrong_1: str,
    wrong_2: str,
    wrong_3: str,
    rng: random.Random,
    wrong_4: Optional[str] = None,
) -> Tuple[Dict[str, str], str]:
    """Shuffle options while retaining the native five-choice schema.

    The legacy LongVideoAgent JSON format has one answer and three explicit
    distractors, so ``wrong_4`` is absent and we keep the historical four
    options.  Native TVQA/LongTVQA rows have one answer plus *four*
    distractors; callers pass the fourth distractor explicitly and all five
    options are shuffled together.  In particular, never write an empty
    ``a4`` for a native five-choice row.
    """
    opts = [answer, wrong_1, wrong_2, wrong_3]
    if wrong_4 is not None:
        opts.append(wrong_4)
    indices = list(range(len(opts)))
    rng.shuffle(indices)
    choices: Dict[str, str] = {}
    gt_key: Optional[str] = None
    for i, idx in enumerate(indices):
        choices[str(i)] = opts[idx]
        if idx == 0:
            gt_key = f"a{i}"
    if len(opts) == 5:
        # Keep the keys contiguous so downstream prompts/reward code see the
        # canonical TVQA a0..a4 contract.
        assert len(choices) == 5
    else:
        # Preserve the legacy four-option contract for custom inputs that only
        # provide wrong_1..wrong_3; this branch is not used by LongTVQA.
        choices["4"] = ""
    return choices, (gt_key or "a0")


def _select_split_rows(
    rows: List[Dict[str, Any]],
    train_per_creator: int,
    val_per_creator: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    n_total = len(rows)
    if n_total == 0:
        return [], []

    target_total = train_per_creator + val_per_creator
    if n_total >= target_total:
        val_rows = rows[:val_per_creator]
        train_rows = rows[val_per_creator : val_per_creator + train_per_creator]
        return train_rows, val_rows

    if n_total == 1:
        return rows, []

    n_val = max(1, n_total // 5)
    n_val = min(n_val, n_total - 1)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    return train_rows, val_rows


def _load_grounding_cache(path: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Returns:
      idx_to_clip: str(idx) -> clip
      q_to_clip: normalized question -> clip
      qid_to_clip: source qid -> clip
      uid_to_clip: source sample UID -> clip
    """
    raw = _load_json_obj(path)
    idx_to_clip: Dict[str, str] = {}
    q_to_clip: Dict[str, str] = {}
    qid_to_clip: Dict[str, str] = {}
    uid_to_clip: Dict[str, str] = {}

    def _consume_entry(k: str, v: Dict[str, Any]) -> None:
        if not isinstance(v, dict):
            return
        clip = str(v.get("clip") or "").strip()
        if not clip:
            return
        status = str(v.get("status") or "").strip().lower()
        if status and status not in {"ok", "empty_clip"}:
            return
        idx_to_clip[str(k)] = clip
        if v.get("qid") is not None:
            qid_to_clip.setdefault(str(v.get("qid")), clip)
        if v.get("sample_uid"):
            uid_to_clip.setdefault(str(v.get("sample_uid")), clip)
        q = _normalize_question_text(str(v.get("question") or ""))
        if q and q not in q_to_clip:
            q_to_clip[q] = clip

    if isinstance(raw, dict):
        # Accept both the legacy flat ``{index: payload}`` format and the
        # versioned local cache with explicit UID/qid/question namespaces.
        namespaces = (("by_uid", "uid"), ("by_qid", "qid"), ("by_question", "question"))
        consumed_namespace = False
        for namespace, kind in namespaces:
            values = raw.get(namespace)
            if not isinstance(values, dict):
                continue
            consumed_namespace = True
            for k, v in values.items():
                if isinstance(v, dict):
                    v = dict(v)
                    if kind == "uid":
                        v.setdefault("sample_uid", str(k))
                    elif kind == "qid":
                        v.setdefault("qid", str(k))
                    elif kind == "question":
                        v.setdefault("question", str(k))
                _consume_entry(str(k), v)
        if not consumed_namespace:
            for k, v in raw.items():
                _consume_entry(str(k), v)
    elif isinstance(raw, list):
        for i, v in enumerate(raw):
            _consume_entry(str(i), v if isinstance(v, dict) else {})

    return idx_to_clip, q_to_clip, qid_to_clip, uid_to_clip


def _resolve_episode_sub_block(
    clip_id: str,
    episode_subtitles: Dict[str, str],
    item: Dict[str, Any],
) -> str:
    # Try explicit episode fields first.
    episode_name = str(item.get("episode_name") or item.get("show_name") or "").strip()
    candidates: List[str] = []
    if episode_name:
        candidates.append(episode_name)

    # Common candidate forms.
    if "_seg" in clip_id:
        candidates.append(clip_id.split("_seg", 1)[0])
    candidates.append(clip_id[:6])

    # Deduplicate while keeping order.
    dedup_candidates = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            dedup_candidates.append(c)
            seen.add(c)

    for c in dedup_candidates:
        if c in episode_subtitles:
            return str(episode_subtitles[c])

    # Fallback: first key that is prefix of clip_id.
    for k, v in episode_subtitles.items():
        if clip_id.startswith(str(k)):
            return str(v)

    return ""


def _resolve_initial_clip(
    item: Dict[str, Any],
    q_to_clip: Dict[str, str],
    qid_to_clip: Optional[Dict[str, str]] = None,
    uid_to_clip: Optional[Dict[str, str]] = None,
) -> str:
    qid_to_clip = qid_to_clip or {}
    uid_to_clip = uid_to_clip or {}
    sample_uid = item.get("sample_uid") or item.get("uid")
    if sample_uid is not None:
        clip = uid_to_clip.get(str(sample_uid), "").strip()
        if clip:
            return clip
    qid = item.get("qid")
    if qid is not None:
        clip = qid_to_clip.get(str(qid), "").strip()
        if clip:
            return clip
    # Primary: by normalized question text from grounding cache.
    q = _normalize_question_text(str(item.get("question") or item.get("q") or ""))
    if q:
        clip = q_to_clip.get(q, "").strip()
        if clip:
            return clip

    return ""


def _build_episode_to_clips(clip_subtitles: Dict[str, Any]) -> Dict[str, List[str]]:
    """Group clip ids by episode id for random fallback sampling."""
    episode_to_clips: Dict[str, List[str]] = defaultdict(list)
    for clip_id in clip_subtitles.keys():
        clip = str(clip_id)
        if not clip:
            continue
        if "_seg" in clip:
            episode = clip.split("_seg", 1)[0]
        else:
            episode = clip[:6]
        if episode:
            episode_to_clips[episode].append(clip)
    return episode_to_clips


def _sample_random_initial_clip(
    item: Dict[str, Any],
    episode_to_clips: Dict[str, List[str]],
    all_clip_ids: List[str],
    rng: random.Random,
) -> str:
    """Fallback when cache misses: sample a clip from same episode if possible."""
    episode_name = str(item.get("episode_name") or item.get("show_name") or "").strip()
    vid_name = str(item.get("vid_name") or item.get("occur_clip") or "").strip()

    candidates: List[str] = []
    if episode_name:
        candidates.append(episode_name)
    if "_seg" in vid_name:
        candidates.append(vid_name.split("_seg", 1)[0])
    if vid_name:
        candidates.append(vid_name[:6])

    seen = set()
    dedup_candidates: List[str] = []
    for c in candidates:
        if c and c not in seen:
            dedup_candidates.append(c)
            seen.add(c)

    for episode in dedup_candidates:
        clips = episode_to_clips.get(episode) or []
        if clips:
            return rng.choice(clips)

    if all_clip_ids:
        return rng.choice(all_clip_ids)
    return ""


def _extract_qa_fields(
    item: Dict[str, Any], rng: random.Random
) -> Optional[Tuple[str, str, str, str, str, str]]:
    question = str(item.get("question") or item.get("q") or "").strip()
    if not question:
        return None

    # LongVideoAgent-style fields: answer + three distractors.
    if any(k in item for k in ["wrong_1", "wrong_2", "wrong_3"]):
        answer_text = str(item.get("answer") or "")
        wrong_1_text = str(item.get("wrong_1") or "")
        wrong_2_text = str(item.get("wrong_2") or "")
        wrong_3_text = str(item.get("wrong_3") or "")
        if not answer_text:
            return None
        # The sixth slot is deliberately empty: this legacy format only has
        # three explicit distractors (four choices total).
        return question, answer_text, wrong_1_text, wrong_2_text, wrong_3_text, ""

    # Native TVQA-style fields: a0..a4 + answer_idx/answer.
    options = [str(item.get(f"a{i}") or "") for i in range(5)]
    answer_idx = item.get("answer_idx")
    if answer_idx is None:
        answer_idx = _parse_answer_to_idx(item.get("answer"))
    else:
        answer_idx = _parse_answer_to_idx(answer_idx)
    if answer_idx is None:
        return None

    answer_text = options[answer_idx]
    if not answer_text:
        return None

    wrong_candidates = [options[i] for i in range(5) if i != answer_idx and options[i]]
    # LongTVQA/TVQA is a five-choice benchmark.  Silently sampling only three
    # distractors changes the task and used to leave ``a4`` blank in the
    # generated parquet.  Require all four non-answer options instead.
    if len(wrong_candidates) < 4:
        return None
    # Preserve every native distractor.  We still randomize the final option
    # order in ``_shuffle_options`` so the answer position is not fixed.
    return question, answer_text, wrong_candidates[0], wrong_candidates[1], wrong_candidates[2], wrong_candidates[3]


def convert(
    questions_path: str,
    output_dir: str,
    grounding_cache_json: Optional[str],
    subtitles_dir: str,
    train_per_creator: int = 16,
    val_per_creator: int = 4,
    seed: int = 42,
    skip_video_ids: Optional[List[str]] = None,
    subset_size: int = 0,
    all_train: bool = False,
    require_grounding_cache: bool = False,
    require_five_choices: bool = False,
) -> None:
    rng = random.Random(seed)

    clip_subtitles_json = os.path.join(subtitles_dir, "LongTVQA_plus_subtitle_clip_level.json")
    episode_subtitles_json = os.path.join(subtitles_dir, "LongTVQA_plus_subtitle_episode_level.json")
    clip_subtitles = _load_json_obj(clip_subtitles_json)
    episode_subtitles = _load_json_obj(episode_subtitles_json)
    if not isinstance(clip_subtitles, dict) or not isinstance(episode_subtitles, dict):
        raise ValueError("subtitle files must both be JSON objects")
    episode_to_clips = _build_episode_to_clips(clip_subtitles)
    all_clip_ids = list(clip_subtitles.keys())

    q_to_clip: Dict[str, str] = {}
    qid_to_clip: Dict[str, str] = {}
    uid_to_clip: Dict[str, str] = {}
    if grounding_cache_json:
        try:
            _, q_to_clip, qid_to_clip, uid_to_clip = _load_grounding_cache(grounding_cache_json)
            if not q_to_clip:
                print(
                    "[warn] grounding cache provided but no usable question->clip mapping found; "
                    "will fallback to random initial clip when cache misses."
                )
        except FileNotFoundError:
            print(
                f"[warn] grounding cache file not found: {grounding_cache_json}; "
                "will fallback to random initial clip."
            )
        except Exception as e:
            print(
                f"[warn] failed to load grounding cache ({grounding_cache_json}): {e}; "
                "will fallback to random initial clip."
            )
    else:
        print("[info] no grounding cache provided; using random initial clip fallback.")

    raw_items = _load_json_or_jsonl(questions_path)
    all_items: List[Tuple[str, str, Dict[str, Any]]] = []
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        vid_name = str(it.get("vid_name") or it.get("occur_clip") or "").strip()
        episode_name = str(it.get("episode_name") or it.get("show_name") or "").strip()
        creator = episode_name or (vid_name.split("_seg", 1)[0] if "_seg" in vid_name else vid_name[:6]) or "default"
        all_items.append(("tvqa", creator, it))

    if not all_items:
        raise RuntimeError("No usable items found from questions file")

    # Optional global random subset.
    if subset_size > 0:
        total_before_subset = len(all_items)
        k = min(subset_size, len(all_items))
        all_items = rng.sample(all_items, k)
        print(f"[subset] random subset enabled: {k}/{total_before_subset} selected")

    by_creator: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for platform, creator, item in all_items:
        by_creator[f"{platform}::{creator}"].append((platform, item))

    skip_set = frozenset(skip_video_ids or [])
    os.makedirs(output_dir, exist_ok=True)

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    skipped_invalid = 0
    skipped_video_ids = 0
    skipped_no_cache_clip = 0
    fallback_random_initial_clip = 0
    skipped_no_clip_sub = 0
    skipped_no_episode_sub = 0
    native_five_choice_rows = 0

    source_file = os.path.basename(questions_path)
    for creator_key in sorted(by_creator.keys()):
        creator_rows: List[Dict[str, Any]] = []
        for local_index, (_, item) in enumerate(by_creator[creator_key]):
            video_id = str(item.get("video_id") or item.get("vid_name") or item.get("occur_clip") or "").strip()
            if video_id in skip_set:
                skipped_video_ids += 1
                continue

            qa_fields = _extract_qa_fields(item=item, rng=rng)
            if qa_fields is None:
                skipped_invalid += 1
                continue
            (
                question,
                answer_text,
                wrong_1_text,
                wrong_2_text,
                wrong_3_text,
                wrong_4_text,
            ) = qa_fields
            if require_five_choices and not wrong_4_text:
                raise RuntimeError(
                    "--require-five-choice received a legacy four-choice row; "
                    f"qid={item.get('qid')!r}"
                )

            initial_clip_id = _resolve_initial_clip(
                item=item, q_to_clip=q_to_clip, qid_to_clip=qid_to_clip, uid_to_clip=uid_to_clip
            )
            initial_clip_from_cache = bool(initial_clip_id)
            if not initial_clip_id:
                if require_grounding_cache:
                    skipped_no_cache_clip += 1
                    continue
                initial_clip_id = _sample_random_initial_clip(
                    item=item,
                    episode_to_clips=episode_to_clips,
                    all_clip_ids=all_clip_ids,
                    rng=rng,
                )
                if initial_clip_id:
                    fallback_random_initial_clip += 1
                else:
                    skipped_no_cache_clip += 1
                    continue

            initial_clip_subtitle = str(clip_subtitles.get(initial_clip_id) or "").strip()
            if not initial_clip_subtitle:
                skipped_no_clip_sub += 1
                continue

            episode_sub_block = _resolve_episode_sub_block(
                clip_id=initial_clip_id,
                episode_subtitles=episode_subtitles,
                item=item,
            )
            if not episode_sub_block:
                skipped_no_episode_sub += 1
                continue

            choices, ground_truth = _shuffle_options(
                answer=answer_text,
                wrong_1=wrong_1_text,
                wrong_2=wrong_2_text,
                wrong_3=wrong_3_text,
                rng=rng,
                wrong_4=wrong_4_text if wrong_4_text else None,
            )
            # Native TVQA/LongTVQA rows must retain all five non-empty
            # choices.  Keep a counter in the conversion log so a malformed
            # or accidentally legacy-formatted pilot cannot pass unnoticed.
            if wrong_4_text:
                if len(choices) != 5 or any(not str(choices.get(str(i), "")).strip() for i in range(5)):
                    raise RuntimeError(f"native five-choice row lost an option (qid={item.get('qid')!r})")
                native_five_choice_rows += 1
            prompt_content = PROMPT_TEMPLATE.format(
                question=question,
                a0=choices.get("0", ""),
                a1=choices.get("1", ""),
                a2=choices.get("2", ""),
                a3=choices.get("3", ""),
                a4=choices.get("4", ""),
                initial_clip_subtitle=initial_clip_subtitle,
            )

            creator_rows.append(
                {
                    "prompt": [{"role": "user", "content": prompt_content}],
                    "agent_name": "longvideoagent_multiturn",
                    "data_source": "tvqa_plus_vision",
                    "reward_model": {"ground_truth": ground_truth, "one_turn": False},
                    "extra_info": {
                        "vid_name": initial_clip_id,
                        "video_id": video_id,
                        "sample_uid": f"{source_file}:{_stable_sample_uid(item, local_index)}",
                        "qid": item.get("qid"),
                        "timestamp": item.get("ts"),
                        "source_occur_clip": item.get("occur_clip"),
                        "source_episode_name": item.get("episode_name") or item.get("show_name"),
                        "initial_clip_from_cache": initial_clip_from_cache,
                        "original_question": question,
                        "choices": choices,
                        "episode_sub_block": episode_sub_block,
                    },
                    # Keep stable identifiers at the row top level as well as in
                    # extra_info: async rollout workers may drop nested fields.
                    "sample_uid": f"{source_file}:{_stable_sample_uid(item, local_index)}",
                    "qid": item.get("qid"),
                }
            )

        rng.shuffle(creator_rows)
        if all_train:
            c_train, c_val = creator_rows, []
        else:
            c_train, c_val = _select_split_rows(creator_rows, train_per_creator, val_per_creator)
        train_rows.extend(c_train)
        val_rows.extend(c_val)

    if not train_rows:
        raise RuntimeError("Generated 0 training rows. Please check cache/subtitles alignment.")

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    train_ds = Dataset.from_list(train_rows)
    if val_rows:
        val_ds = Dataset.from_list(val_rows)
    else:
        # Keep val.parquet schema aligned with train when val is intentionally empty.
        val_ds = train_ds.select([])
    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print(f"Done: {output_dir}")
    print(f"  train: {train_path} ({len(train_rows)} rows)")
    print(f"  val:   {val_path} ({len(val_rows)} rows)")
    print(f"  skipped_invalid: {skipped_invalid}")
    print(f"  skipped_video_ids: {skipped_video_ids}")
    print(f"  skipped_no_cache_clip: {skipped_no_cache_clip}")
    print(f"  fallback_random_initial_clip: {fallback_random_initial_clip}")
    print(f"  skipped_no_clip_subtitle: {skipped_no_clip_sub}")
    print(f"  skipped_no_episode_subtitle: {skipped_no_episode_sub}")
    print(f"  native_five_choice_rows: {native_five_choice_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LongTVQA/LongTVQA+ JSON/JSONL -> GRPO parquet converter")
    parser.add_argument("--questions-path", type=str, required=True, help="Path to LongTVQA or LongTVQA+ questions file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output dir for train.parquet and val.parquet")
    parser.add_argument(
        "--grounding-cache-json",
        type=str,
        default=None,
        help=(
            "Optional cache JSON from build_grounding_cache.py. "
            "If omitted/unavailable/empty, initial clip falls back to random sampling."
        ),
    )
    parser.add_argument(
        "--subtitles-dir",
        type=str,
        required=True,
        help=(
            "Directory containing both subtitle files: "
            "LongTVQA_plus_subtitle_clip_level.json and "
            "LongTVQA_plus_subtitle_episode_level.json"
        ),
    )
    parser.add_argument("--train-per-creator", type=int, default=16)
    parser.add_argument("--val-per-creator", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-videos", type=str, nargs="*", default=None, help="Optional video_id list to skip")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=0,
        help="If >0, randomly sample N items globally before split; else use full dataset",
    )
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="Put all valid rows into train set (no per-creator cap, val will be empty)",
    )
    parser.add_argument(
        "--require-grounding-cache",
        action="store_true",
        help="Skip rows missing an offline grounding-cache entry; never use random/GT clip fallback.",
    )
    parser.add_argument(
        "--require-five-choice",
        action="store_true",
        help="Fail if any input row uses the legacy four-choice format; native LongTVQA rows must keep a0..a4.",
    )
    args = parser.parse_args()

    convert(
        questions_path=args.questions_path,
        output_dir=args.output_dir,
        grounding_cache_json=args.grounding_cache_json,
        subtitles_dir=args.subtitles_dir,
        train_per_creator=args.train_per_creator,
        val_per_creator=args.val_per_creator,
        seed=args.seed,
        skip_video_ids=args.skip_videos,
        subset_size=args.subset_size,
        all_train=args.all_train,
        require_grounding_cache=args.require_grounding_cache,
        require_five_choices=args.require_five_choice,
    )


if __name__ == "__main__":
    main()
