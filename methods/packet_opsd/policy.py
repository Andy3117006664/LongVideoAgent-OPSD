"""Qwen3 LoRA student and a stop-gradient EMA of the same adapter.

The frozen base is shared. No externally trained teacher or answer model is
introduced. Generation and sequence scoring use exactly the same temperature,
without top-k/top-p filtering, so the packet score-function estimator is valid.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class Completion:
    text: str
    token_ids: list[int]
    prompt_ids: list[int]
    complete: bool


class ContextOverflow(ValueError):
    pass


class HFPolicy:
    def __init__(self, model, tokenizer, *, max_context: int = 40960):
        self.model, self.tokenizer = model, tokenizer
        self.max_context = int(max_context)
        self.device = model.get_input_embeddings().weight.device
        self.select("student")

    @classmethod
    def load(cls, path: str, *, rank: int = 8, device: str = "cuda:0",
             max_context: int = 40960, student_adapter: str | None = None):
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        base = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, local_files_only=True,
            attn_implementation="sdpa", device_map={"": device},
        )
        if student_adapter:
            model = PeftModel.from_pretrained(base, student_adapter,
                                             adapter_name="student", is_trainable=True)
            config = model.peft_config["student"]
        else:
            config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                target_modules=["q_proj", "v_proj"],
                                task_type="CAUSAL_LM", bias="none")
            model = get_peft_model(base, config, adapter_name="student")
        model.add_adapter("teacher", config)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy = cls(model, tokenizer, max_context=max_context)
        policy.update_teacher(0.0)
        return policy

    def select(self, name: str):
        if name not in {"student", "teacher"}:
            raise ValueError(name)
        self.model.set_adapter(name)
        # Some PEFT versions make the active teacher trainable in set_adapter.
        for key, parameter in self.model.named_parameters():
            parameter.requires_grad_(name == "student" and ".student." in key)

    @contextmanager
    def adapter(self, name: str):
        self.select(name)
        try:
            yield
        finally:
            self.select("student")

    def parameters(self):
        return [p for key, p in self.model.named_parameters() if ".student." in key]

    @torch.no_grad()
    def update_teacher(self, decay: float):
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        params = dict(self.model.named_parameters())
        pairs = [(key, key.replace(".student.", ".teacher."))
                 for key in params if ".student." in key]
        if not pairs:
            raise RuntimeError("no student adapter parameters")
        for key, target in pairs:
            if target not in params:
                raise RuntimeError(f"missing EMA counterpart: {target}")
            params[target].mul_(decay).add_(params[key], alpha=1 - decay)
        self.select("student")

    def prompt_ids(self, messages: list[dict], prefix: str = "") -> list[int]:
        # Match the original chat endpoint: a system message followed by the
        # legacy tools/history user message. No training-only no_think switch.
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        return self.tokenizer.encode(text + prefix, add_special_tokens=False)

    @torch.no_grad()
    def sample(self, messages: list[dict], *, adapter: str = "student",
               prefix: str = "", max_new_tokens: int = 4096,
               temperature: float = 0.7) -> Completion:
        if temperature < 0 or max_new_tokens < 1:
            raise ValueError("invalid generation configuration")
        ids = self.prompt_ids(messages, prefix)
        if len(ids) + max_new_tokens > self.max_context:
            raise ContextOverflow(f"prompt {len(ids)} + generation {max_new_tokens} > {self.max_context}")
        inputs = torch.tensor([ids], device=self.device)
        kwargs: dict[str, Any] = dict(
            max_new_tokens=int(max_new_tokens), do_sample=temperature > 0,
            top_k=0, top_p=1.0, repetition_penalty=1.0, use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            kwargs["temperature"] = float(temperature)
        if prefix:
            from transformers import StoppingCriteria, StoppingCriteriaList
            tokenizer = self.tokenizer
            offset = len(ids)

            class EndToolCall(StoppingCriteria):
                def __call__(self, input_ids, scores, **_kwargs):
                    text = tokenizer.decode(input_ids[0, offset:], skip_special_tokens=True)
                    return "</tool_call>" in text

            kwargs["stopping_criteria"] = StoppingCriteriaList([EndToolCall()])
        self.model.eval()
        with self.adapter(adapter):
            generated = self.model.generate(input_ids=inputs,
                                             attention_mask=torch.ones_like(inputs), **kwargs)
        tokens = generated[0, len(ids):].tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        complete = bool(tokens and tokens[-1] == self.tokenizer.eos_token_id)
        return Completion(text, tokens, ids, complete or (bool(prefix) and "</tool_call>" in text))

    def hidden_for_response(self, prompt_ids: list[int], tokens: list[int]):
        if not prompt_ids or not tokens or len(prompt_ids) + len(tokens) > self.max_context:
            raise ContextOverflow("empty or oversized scoring sequence")
        # Avoid allocating context_length x vocabulary logits. Qwen3 is the
        # baseline architecture; fail explicitly for an unsupported backbone.
        base = self.model.get_base_model()
        if not hasattr(base, "model") or not hasattr(base, "lm_head"):
            raise TypeError("packet OPSD requires a decoder backbone with model/lm_head")
        ids = torch.tensor([prompt_ids + tokens], device=self.device)
        output = base.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            use_cache=False, return_dict=True)
        start = len(prompt_ids) - 1
        return output.last_hidden_state[0, start:start + len(tokens)]

    def sequence_log_prob(self, completion: Completion, *, temperature: float):
        if temperature <= 0:
            raise ValueError("score-function sampling requires positive temperature")
        self.select("student")
        self._scoring_mode()
        hidden = self.hidden_for_response(completion.prompt_ids, completion.token_ids)
        head = self.model.get_base_model().lm_head
        value = hidden.sum() * 0.0
        for start in range(0, len(completion.token_ids), 32):
            logits = head(hidden[start:start + 32]).float() / temperature
            targets = torch.tensor(completion.token_ids[start:start + 32], device=self.device)
            value = value + torch.log_softmax(logits, -1).gather(1, targets[:, None]).sum()
        return value

    def hindsight_kl(self, completion: Completion, teacher_messages: list[dict],
                     *, temperature: float):
        """Dense reverse KL on the *same* sampled student action prefixes."""
        if temperature <= 0:
            raise ValueError("training temperature must be positive")
        teacher_ids = self.prompt_ids(teacher_messages)
        self.model.eval()
        with self.adapter("teacher"), torch.no_grad():
            teacher_hidden = self.hidden_for_response(teacher_ids, completion.token_ids).detach()
        self.select("student")
        self._scoring_mode()
        student_hidden = self.hidden_for_response(completion.prompt_ids, completion.token_ids)
        head = self.model.get_base_model().lm_head
        loss = student_hidden.sum() * 0.0
        for start in range(0, len(completion.token_ids), 32):
            student = torch.log_softmax(head(student_hidden[start:start + 32]).float() / temperature, -1)
            with torch.no_grad():
                teacher = torch.log_softmax(head(teacher_hidden[start:start + 32]).float() / temperature, -1)
            loss = loss + (student.exp() * (student - teacher)).sum()
        return loss / len(completion.token_ids)

    def _scoring_mode(self):
        # Qwen3 and our LoRA have zero dropout. Training mode enables gradient
        # checkpointing without changing the generation distribution.
        if any(isinstance(module, torch.nn.Dropout) and module.p != 0
               for module in self.model.modules()):
            raise ValueError("nonzero dropout would change the score-function distribution")
        if float(getattr(self.model.config, "attention_dropout", 0.0)) != 0.0:
            raise ValueError("nonzero attention dropout would change the score distribution")
        self.model.train()

    def save(self, root: Path, *, optimizer=None, step: int, config: dict):
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        self.select("student")
        self.model.save_pretrained(root, selected_adapters=["student", "teacher"])
        self.tokenizer.save_pretrained(root / "tokenizer")
        if optimizer is not None:
            torch.save({"step": step, "optimizer": optimizer.state_dict(),
                        "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                        "config": config}, root / "training_state.pt")
