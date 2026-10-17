"""Optional tiny Qwen3/LoRA checks. No downloaded weights or tokenizer."""

import importlib.util
import unittest


AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("torch", "transformers", "peft"))

if AVAILABLE:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from methods.packet_opsd.policy import Completion, ContextOverflow, HFPolicy


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "|".join(message["content"] for message in messages) + "|assistant:"

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [3 + ord(character) % 61 for character in text]

    def decode(self, token_ids, *, skip_special_tokens=True):
        return " ".join(str(value) for value in token_ids if value not in (0, 2))


@unittest.skipUnless(AVAILABLE, "torch/transformers/peft are not installed")
class PolicyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        config = Qwen3Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, max_position_embeddings=128, attention_dropout=0.0,
            pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        base = Qwen3ForCausalLM(config)
        adapter = LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                             target_modules=["q_proj", "v_proj"],
                             task_type="CAUSAL_LM", bias="none")
        model = get_peft_model(base, adapter, adapter_name="student")
        model.add_adapter("teacher", adapter)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.policy = HFPolicy(model, TinyTokenizer(), max_context=128)
        self.policy.update_teacher(0.0)
        self.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]

    def test_sequence_log_probability_matches_full_logits_and_generation(self):
        temperature = 0.7
        torch.manual_seed(22)
        completion = self.policy.sample(self.messages, max_new_tokens=6, temperature=temperature)
        actual = self.policy.sequence_log_prob(completion, temperature=temperature)
        self.policy.model.eval()
        ids = torch.tensor([completion.prompt_ids + completion.token_ids])
        with torch.no_grad():
            logits = self.policy.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       use_cache=False).logits[0]
            start = len(completion.prompt_ids) - 1
            logits = logits[start:start + len(completion.token_ids)].float() / temperature
            targets = torch.tensor(completion.token_ids)
            expected = logits.log_softmax(-1).gather(1, targets[:, None]).sum()
        torch.testing.assert_close(actual.detach(), expected, rtol=1e-5, atol=1e-5)

        # The model's own generation scores include its temperature processor.
        # Compare their exact path probability with our differentiable scorer.
        torch.manual_seed(22)
        prompt = torch.tensor([completion.prompt_ids])
        with torch.no_grad():
            generated = self.policy.model.generate(
                input_ids=prompt, attention_mask=torch.ones_like(prompt),
                max_new_tokens=6, do_sample=True, temperature=temperature,
                top_k=0, top_p=1.0, repetition_penalty=1.0, use_cache=True,
                pad_token_id=0, eos_token_id=2,
                return_dict_in_generate=True, output_scores=True,
            )
        self.assertEqual(generated.sequences[0, len(completion.prompt_ids):].tolist(), completion.token_ids)
        generated_score = sum(
            logits[0].float().log_softmax(-1)[token]
            for logits, token in zip(generated.scores, completion.token_ids)
        )
        torch.testing.assert_close(actual.detach(), generated_score, rtol=1e-5, atol=1e-5)

    def test_optimizer_updates_only_student_then_ema_updates_frozen_teacher(self):
        parameters = dict(self.policy.model.named_parameters())
        before = {name: value.detach().clone() for name, value in parameters.items()}
        completion = Completion("", [7, 11, 13], self.policy.prompt_ids(self.messages), True)
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=0.05, weight_decay=0)
        optimizer.zero_grad(set_to_none=True)
        loss = -self.policy.sequence_log_prob(completion, temperature=0.7)
        self.assertTrue(self.policy.model.training)  # checkpointing must stay enabled
        loss.backward()
        self.assertTrue(any(value.grad is not None for name, value in parameters.items() if ".student." in name))
        self.assertTrue(all(value.grad is None for name, value in parameters.items() if ".student." not in name))
        optimizer.step()
        self.assertTrue(any(not torch.equal(value, before[name]) for name, value in parameters.items()
                            if ".student." in name))
        for name, value in parameters.items():
            if ".student." not in name:
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.policy.update_teacher(0.8)
        for name, value in parameters.items():
            if ".teacher." in name:
                student = parameters[name.replace(".teacher.", ".student.")]
                torch.testing.assert_close(value, before[name] * 0.8 + student.detach() * 0.2)
                self.assertFalse(value.requires_grad)
            elif ".student." in name:
                self.assertTrue(value.requires_grad)

    def test_hindsight_kl_has_student_gradient_and_no_teacher_gradient(self):
        completion = Completion("", [7, 11, 13], self.policy.prompt_ids(self.messages), True)
        teacher_messages = self.messages + [{"role": "user", "content": "natural feedback"}]
        loss = self.policy.hindsight_kl(completion, teacher_messages, temperature=0.7)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.detach()), -1e-6)
        self.assertTrue(self.policy.model.training)
        loss.backward()
        parameters = dict(self.policy.model.named_parameters())
        self.assertTrue(any(value.grad is not None for name, value in parameters.items() if ".student." in name))
        self.assertTrue(all(value.grad is None and not value.requires_grad
                            for name, value in parameters.items() if ".teacher." in name))

    def test_context_and_stochastic_score_guards(self):
        with self.assertRaises(ContextOverflow):
            self.policy.sample(self.messages, max_new_tokens=129)
        completion = Completion("", [7], self.policy.prompt_ids(self.messages), True)
        with self.assertRaises(ValueError):
            self.policy.sequence_log_prob(completion, temperature=0)
        self.policy.model.config.attention_dropout = 0.1
        with self.assertRaises(ValueError):
            self.policy.sequence_log_prob(completion, temperature=0.7)


if __name__ == "__main__":
    unittest.main()
