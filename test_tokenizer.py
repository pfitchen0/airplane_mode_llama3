import unittest

from tokenizer import Tokenizer
from utils import Message


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_special_tokens(self):
        text = "<|begin_of_text|>This is a test<|end_of_text|>"
        encoded = self.tokenizer.encode(text)
        expected_encoding = [128000, 2028, 374, 264, 1296, 128001]
        self.assertEqual(encoded, expected_encoding)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_utf8_characters(self):
        text = "hello123!!!? (안녕하세요!) 😉"
        encoded = self.tokenizer.encode(text)
        expected_encoding = [15339, 4513, 12340, 30, 320, 101193, 124409, 16715, 57037]
        self.assertEqual(encoded, expected_encoding)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_apply_chat_template_no_tokenize(self):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        text = self.tokenizer.apply_chat_template(
            messages=messages, tokenize=False, add_generation_prompt=False
        )
        expected_text = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\nHi there!<|eot_id|>"
        )
        self.assertEqual(text, expected_text)

    def test_apply_chat_template_tokenize(self):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages=messages, tokenize=True, add_generation_prompt=False
        )
        expected_encoding = [
            128000,
            128006,
            882,
            128007,
            271,
            9906,
            128009,
            128006,
            78191,
            128007,
            271,
            13347,
            1070,
            0,
            128009,
        ]
        self.assertAlmostEqual(encoded, expected_encoding)

    def test_apply_chat_template_with_generation_prompt(self):
        messages = [Message(role="user", content="Hello")]
        result = self.tokenizer.apply_chat_template(
            messages=messages, tokenize=False, add_generation_prompt=True
        )
        expected_result = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        self.assertEqual(result, expected_result)

    def test_eos_token_id(self):
        self.assertEqual(
            self.tokenizer.eos_token_id, self.tokenizer.special_tokens["<|eot_id|>"]
        )

    def test_eom_token_id(self):
        self.assertEqual(
            self.tokenizer.eom_token_id, self.tokenizer.special_tokens["<|eom_id|>"]
        )


if __name__ == "__main__":
    unittest.main()
