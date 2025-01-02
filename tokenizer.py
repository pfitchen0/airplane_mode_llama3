"""Pure Python implementation of a Llama3 tiktoken tokenizer.

This module takes very heavy inspiration from:

1. OpenAI's educational tiktoken implementation:
https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py

2. Andrej Karpathy's tokenizer video and minbpe regex.py tokenizer implementation:
https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py

3. Meta's Llama3 tiktoken reference implementation and prompt format documentation:
https://github.com/meta-llama/llama-models/blob/main/models/llama3/api/tokenizer.py
https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3

This code is basically a conglomeration of these above references!
I just want to make sure all credit goes where it is due.
"""

import os
import regex
from pathlib import Path

# For loading the model only:
from tiktoken.load import load_tiktoken_bpe

from utils import Message, Role

_DEFAULT_MODEL_PATH = "./"


class Tokenizer:
    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH) -> None:
        """Creates a Tokenizer object."""

        model_path = Path(model_path)
        tokenizer_model = model_path / "tokenizer.model"
        assert os.path.isfile(
            tokenizer_model
        ), f"{tokenizer_model} not found, download it from https://github.com/meta-llama/llama-models/blob/main/models/llama3/api/tokenizer.model"

        special_tokens = {}
        num_reserved_special_tokens = 256
        pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

        mergeable_ranks = load_tiktoken_bpe(str(tokenizer_model))
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",
            "<|eot_id|>",
            "<|python_tag|>",
            "<|image|>",
        ]
        reserved_tokens = [
            f"<|reserved_special_token_{2 + i}|>"
            for i in range(num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens
        special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        self.mergeable_ranks = mergeable_ranks
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        self._special_pattern = regex.compile(
            "(" + "|".join(regex.escape(k) for k in self.special_tokens) + ")"
        )

        self._decoder = {
            token: token_bytes for token_bytes, token in mergeable_ranks.items()
        }
        self._pat = regex.compile(pat_str)

    def apply_chat_template(
        self,
        messages: list[Message],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        def format_header(role: Role) -> str:
            return f"<|start_header_id|>{role}<|end_header_id|>\n\n"

        def format_message(message: Message) -> str:
            return f"{format_header(message.role)}{message.content}<|eot_id|>"

        text = "<|begin_of_text|>"

        for message in messages:
            text += format_message(message)

        if add_generation_prompt:
            text += format_header("assistant")

        if tokenize:
            return self.encode(text=text)
        else:
            return text

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<|eot_id|>"]

    @property
    def eom_token_id(self) -> int:
        return self.special_tokens["<|eom_id|>"]

    def encode(self, text: str) -> list[int]:
        """Encodes a string into tokens including special tokens."""
        # From Karpathy's minbpe repo:
        # https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py#L123
        # We have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use regex.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_chunks = regex.split(self._special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in self.special_tokens:
                # this is a special token, encode it separately as a special case
                ids.append(self.special_tokens[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of tokens into a string.

        Decoded bytes are not guaranteed to be valid UTF-8. In that case, we replace
        the invalid bytes with the replacement character "�"."""
        # From Karpathy's minbpe repo:
        # https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py#L78
        part_bytes = []
        for token in tokens:
            if token in self._decoder:
                part_bytes.append(self._decoder[token])
            elif token in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[token].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token: {token}")
        return b"".join(part_bytes).decode("utf-8", errors="replace")

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encodes a string into tokens ignoring special tokens."""
        # Use the regex to split the text into (approximately) words
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.encode("utf-8")
            word_tokens = self._bpe_encode(word_bytes)
            tokens.extend(word_tokens)
        return tokens

    def _bpe_encode(self, input: bytes) -> list[int]:
        parts = [bytes([b]) for b in input]
        while True:
            # Iterate over all pairs and find the pair we want to merge the most
            min_idx = None
            min_rank = None
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                rank = self.mergeable_ranks.get(pair[0] + pair[1])
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_idx = i
                    min_rank = rank

            # If there were no pairs we could merge, we're done!
            if min_rank is None:
                break
            assert min_idx is not None

            # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
            parts = (
                parts[:min_idx]
                + [parts[min_idx] + parts[min_idx + 1]]
                + parts[min_idx + 2 :]
            )

        tokens = [self.mergeable_ranks[part] for part in parts]
        return tokens


if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(tokenizer.decode(tokenizer.encode("hello123!!!? (안녕하세요!) 😉")))

    messages = [Message(role="user", content="Hi there!")]
    print(
        tokenizer.apply_chat_template(
            messages=messages, tokenize=False, add_generation_prompt=True
        )
    )
