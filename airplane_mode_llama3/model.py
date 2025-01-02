import dataclasses
from glob import glob
import json
from pathlib import Path
from typing import Generator
from huggingface_hub import snapshot_download
import mlx.core as mx
import mlx.nn as nn


@dataclasses.dataclass
class Config:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    tie_word_embeddings: bool

    def __init__(self, **kwargs):
        # Only copy items whose key matches a parameter in Config.
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


class Attention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.rope = nn.RoPE(dims=self.head_dim, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_attention_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Flash attention.
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.head_dim**-0.5, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output), (keys, values)


class Mlp(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.self_attn = Attention(config)
        self.mlp = Mlp(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache


class Transformer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.tie_word_embeddings = config.tie_word_embeddings
        self.model = Transformer(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def generate(
        self, tokens: list[int], temperature: float
    ) -> Generator[int, None, None]:
        # 2 step process:
        # 1. Process the prompt with a causal mask, saving keys and values into per layer caches.
        # 2. Generate new tokens one at a time with no mask, updating the per layer caches each time.
        # Also, explicitly clear metal cache memory every so often (not to be confused with the per layer caches).

        # 1. Process the prompt and populate per layer caches.

        x = mx.array([tokens])  # add a batch dimension
        per_layer_caches = []
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.model.embed_tokens.weight.dtype)

        x = self.model.embed_tokens(x)
        for layer in self.model.layers:
            x, cache = layer(x, mask=mask)
            per_layer_caches.append(cache)
        x = self.model.norm(x)
        y = self._lm_head_sampler(x, temperature)
        mx.metal.clear_cache()  # clear metal cache memory the first time
        yield y.item()

        # 2. Generate new tokens one at a time.

        new_token_count = 0
        while True:
            x = y[:, None]  # add a seq len dim
            x = self.model.embed_tokens(x)
            for i, layer in enumerate(self.model.layers):
                x, per_layer_caches[i] = layer(x, mask=None, cache=per_layer_caches[i])
            x = self.model.norm(x)
            y = self._lm_head_sampler(x, temperature)

            # Explicitly clear metal cache memory every so often.
            new_token_count += 1
            if new_token_count % 128 == 0:
                mx.metal.clear_cache()

            yield y.item()

    def _lm_head_sampler(self, x: mx.array, temperature: float) -> mx.array:
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(x[:, -1])
        else:
            logits = self.lm_head(x[:, -1])

        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temperature))

    @staticmethod
    def from_huggingface(hf_repo_id: str):
        model_path = Path(
            snapshot_download(
                repo_id=hf_repo_id,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                ],
            )
        )
        with open(model_path / "config.json", "r") as f:
            config_json = json.load(f)
        config = Config(**config_json)
        model = Model(config=config)

        # If the model needs to be quantized, do it before loading the weights.
        if (quantization := config_json.get("quantization", None)) is not None:
            nn.quantize(
                model,
                **quantization,
            )

        weight_files = glob(str(model_path / "model*.safetensors"))
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        model.eval()
        return model
