"""Read one answer-free fixed300 identity contract before starting a run."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class IdentityMapping:
    path: Path
    sha256: str
    rows: tuple[dict, ...]

    def metadata(self) -> dict[str, str]:
        return {"identity_mapping": str(self.path), "identity_mapping_sha256": self.sha256}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in identity mapping")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant in identity mapping")


def resolve_identity_mapping(path: Path | None = None, *, repo: Path | None = None) -> IdentityMapping:
    """Resolve and validate the exact bytes used for both guards and provenance.

    The default remains the archived public fixed300 mapping. A replacement is
    an equally strict 300-row, answer-free contract; it never relaxes overlap
    checks or imports model/scoring code. Consumers reuse these loaded rows
    rather than rereading a possibly changed mapping for the recorded hash.
    """
    if path is None:
        repo = repo or Path(__file__).resolve().parents[2]
        path = repo / "baselines/videoseal_6000d/selection/identity_mapping.json"
    path = Path(path).expanduser().resolve(strict=True)
    content = path.read_bytes()
    rows = json.loads(content.decode("utf-8"), object_pairs_hook=_object,
                      parse_constant=_reject_constant)
    if not isinstance(rows, list) or len(rows) != 300:
        raise ValueError("Identity mapping must contain exactly 300 rows")
    required = {"ordinal", "qid", "video_id", "question_sha256"}
    allowed = required | {"sample_uid", "original_video_id"}
    ordinals, qids, aliases, video_aliases = set(), set(), {}, {}
    for row in rows:
        if not isinstance(row, dict) or required - set(row):
            raise ValueError("Identity mapping row is missing required fields")
        if set(row) - allowed:
            raise ValueError("Identity mapping has unexpected fields; only answer-free identities are allowed")
        for key in ("qid", "video_id", "sample_uid", "original_video_id"):
            if key in row and (not isinstance(row[key], str) or not row[key]
                               or row[key] != row[key].strip()):
                raise ValueError(f"Identity mapping {key} must be a nonempty string without edge whitespace")
        qid, ordinal = row["qid"], row["ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal not in range(300):
            raise ValueError("Identity mapping ordinal must be an integer from 0 through 299")
        if qid in qids or ordinal in ordinals:
            raise ValueError("Duplicate qid or ordinal in identity mapping")
        if not isinstance(row["question_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["question_sha256"]):
            raise ValueError("Identity mapping question_sha256 must be a lowercase SHA256 hex string")
        for alias in {qid, row.get("sample_uid", qid)}:
            if alias in aliases and aliases[alias] != qid:
                raise ValueError("Ambiguous UID alias in identity mapping")
            aliases[alias] = qid
        for video in {row["video_id"], row.get("original_video_id", row["video_id"])}:
            if video in video_aliases and video_aliases[video] != row["video_id"]:
                raise ValueError("Ambiguous video alias in identity mapping")
            video_aliases[video] = row["video_id"]
        qids.add(qid)
        ordinals.add(ordinal)
    if ordinals != set(range(300)):
        raise ValueError("Identity mapping ordinals must cover 0 through 299")
    return IdentityMapping(path, hashlib.sha256(content).hexdigest(), tuple(rows))


def bound_mapping(mapping: Path | IdentityMapping) -> IdentityMapping:
    return mapping if isinstance(mapping, IdentityMapping) else resolve_identity_mapping(mapping)
