import mlx.core as mx
import mlx.nn as nn
import unittest


from airplane_mode_llama3.model import (
    Attention,
    Config,
    Mlp,
    Model,
    TransformerBlock,
)


class TestModel(unittest.TestCase):

    def setUp(self):
        self.config = Config(
            hidden_size=128,
            intermediate_size=512,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            rope_theta=10000,
            vocab_size=1000,
            tie_word_embeddings=False,
        )

    def test_attention(self):
        attention = Attention(self.config)
        x = mx.random.normal(
            (2, 10, self.config.hidden_size)
        )  # Batch size 2, sequence length 10
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        output, cache = attention(x, mask)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(
            cache[0].shape,
            (
                2,
                self.config.num_attention_heads,
                10,
                self.config.hidden_size // self.config.num_key_value_heads,
            ),
        )
        self.assertEqual(
            cache[1].shape,
            (
                2,
                self.config.num_attention_heads,
                10,
                self.config.hidden_size // self.config.num_key_value_heads,
            ),
        )

    def test_mlp(self):
        mlp = Mlp(self.config)
        x = mx.random.normal((2, 10, self.config.hidden_size))
        output = mlp(x)
        self.assertEqual(output.shape, x.shape)

    def test_transformer_block(self):
        block = TransformerBlock(self.config)
        x = mx.random.normal((2, 10, self.config.hidden_size))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        output, _ = block(x, mask)
        self.assertEqual(output.shape, x.shape)

    def test_model_generate(self):
        model = Model(self.config)
        tokens = [1, 2, 3]
        generator = model.generate(tokens, temperature=0.8)
        self.assertEqual(len(list(next(generator) for _ in range(5))), 5)


if __name__ == "__main__":
    unittest.main()
