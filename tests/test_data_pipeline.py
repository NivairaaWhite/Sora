import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import DEFAULT_MODEL, SYSTEM_PROMPT, legacy_serialize, make_sft_examples, normalize_messages
from sft_format import as_list_ids, assistant_only_features


class DataPipelineTests(unittest.TestCase):
    def test_default_model_is_instruction_tuned_family(self):
        self.assertEqual(DEFAULT_MODEL, "Qwen/Qwen2.5-0.5B-Instruct")

    def test_normalize_filters_invalid_roles_and_blank_content(self):
        result = normalize_messages([
            {"role": "user", "content": " hello "},
            {"role": "system", "content": "drop"},
            {"role": "assistant", "content": "  "},
            {"role": "assistant", "content": "world"},
        ])
        self.assertEqual(result, [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ])

    def test_one_example_per_assistant_turn_and_context_is_kept(self):
        dialogues = [{"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "follow-up"},
            {"role": "assistant", "content": "answer two"},
        ]}]
        examples, warnings = make_sft_examples(dialogues)
        self.assertEqual(len(examples), 2)
        self.assertFalse(warnings)
        self.assertEqual(examples[0]["messages"][-1]["content"], "answer one")
        self.assertEqual(examples[1]["messages"][-2]["content"], "follow-up")
        self.assertEqual(examples[1]["messages"][-1]["content"], "answer two")

    def test_legacy_serializer_is_consistent_and_does_not_add_special_tokens(self):
        text = legacy_serialize([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        self.assertTrue(text.startswith(SYSTEM_PROMPT))
        self.assertIn("<|user|>\nhello\n", text)
        self.assertIn("<|assistant|>\nhi\n", text)

    def test_source_dataset_has_valid_examples_and_varied_prompts(self):
        data = json.loads((ROOT / "data" / "sister_conversations.json").read_text(encoding="utf-8"))
        examples, warnings = make_sft_examples(data)
        self.assertGreaterEqual(len(data), 100)
        self.assertGreaterEqual(len(examples), 100)
        prompts = {m["content"].casefold() for row in examples for m in row["messages"] if m["role"] == "user"}
        self.assertIn("who created you?", prompts)
        self.assertIn("tell me about death.", prompts)
        self.assertIn("what is an epistle?", prompts)
        self.assertEqual(warnings, [])

    def test_tokenizer_output_dict_is_normalized(self):
        self.assertEqual(as_list_ids({"input_ids": [11, 12, 13]}), [11, 12, 13])
        self.assertEqual(as_list_ids({"input_ids": [[11, 12, 13]]}), [11, 12, 13])

    def test_assistant_only_format_masks_prompt_tokens(self):
        class FakeTokenizer:
            chat_template = "fake-template"
            eos_token = "<eos>"
            pad_token_id = 0

            def __init__(self):
                self.vocab = {}

            def _word_id(self, word):
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab) + 20
                return self.vocab[word]

            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
                ids = [1]
                role_ids = {"system": 2, "user": 3, "assistant": 4}
                for message in messages:
                    ids.append(role_ids[message["role"]])
                    ids.extend(self._word_id(w) for w in message["content"].split())
                    ids.append(5)
                if add_generation_prompt:
                    ids.append(role_ids["assistant"])
                return ids

        tokenizer = FakeTokenizer()
        features = assistant_only_features([
            {"role": "user", "content": "PLEASE_MASK_THIS"},
            {"role": "assistant", "content": "TRAIN_THIS_RESPONSE"},
        ], tokenizer, max_length=512)
        self.assertIsNotNone(features)
        user_id = tokenizer.vocab["PLEASE_MASK_THIS"]
        answer_id = tokenizer.vocab["TRAIN_THIS_RESPONSE"]
        self.assertIn(user_id, features["input_ids"])
        self.assertEqual(features["labels"][features["input_ids"].index(user_id)], -100)
        self.assertEqual(features["labels"][features["input_ids"].index(answer_id)], answer_id)


if __name__ == "__main__":
    unittest.main()
