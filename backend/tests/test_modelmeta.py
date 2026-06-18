import unittest

from ..app import modelmeta
from ..app.params import build_schema
from ..app.schemas import Capabilities, HardwareInfo


class ModelMetaNestedConfigTests(unittest.TestCase):
    def test_diffusion_gemma_uses_nested_text_config_for_core_dimensions(self) -> None:
        config = {
            "model_type": "diffusion_gemma",
            "architectures": ["DiffusionGemmaForBlockDiffusion"],
            "dtype": "bfloat16",
            "canvas_length": 256,
            "text_config": {
                "model_type": "diffusion_gemma_text",
                "vocab_size": 262144,
                "hidden_size": 2816,
                "intermediate_size": 2112,
                "num_hidden_layers": 30,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 256,
                "max_position_embeddings": 262144,
                "sliding_window": 1024,
                "num_experts": 128,
                "moe_intermediate_size": 704,
            },
        }

        meta = modelmeta._meta_from_config(
            repo="google/diffusiongemma-26B-A4B-it",
            revision="main",
            quant="none",
            config=config,
            weight_bytes_known=51_647_701_024,
            warnings=[],
        )

        self.assertEqual(meta.family, "diffusion")
        self.assertEqual(meta.model_type, "diffusion_gemma")
        self.assertEqual(meta.hidden_size, 2816)
        self.assertEqual(meta.num_hidden_layers, 30)
        self.assertEqual(meta.num_attention_heads, 16)
        self.assertEqual(meta.num_key_value_heads, 8)
        self.assertEqual(meta.head_dim, 256)
        self.assertEqual(meta.max_position_embeddings, 262144)
        self.assertEqual(meta.num_experts, 128)
        self.assertGreater(meta.param_count, 0)
        self.assertFalse(meta.warnings)

    def test_diffusion_gemma_schema_uses_canvas_length_default(self) -> None:
        meta = modelmeta._meta_from_config(
            repo="google/diffusiongemma-26B-A4B-it",
            revision="main",
            quant="none",
            config={
                "model_type": "diffusion_gemma",
                "architectures": ["DiffusionGemmaForBlockDiffusion"],
                "canvas_length": 256,
                "text_config": {
                    "hidden_size": 2816,
                    "num_hidden_layers": 30,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 256,
                    "max_position_embeddings": 262144,
                },
            },
            weight_bytes_known=None,
            warnings=[],
        )
        caps = Capabilities(
            supports_bf16=False,
            supports_fp8=False,
            supports_marlin=False,
            supported_quantization=[],
            supported_dtypes=["float16"],
            recommended_dtype="float16",
        )
        hw = HardwareInfo(capabilities=caps)

        schema = build_schema(meta, caps, hw)
        diffusion = next(group for group in schema.groups if group.key == "diffusion")
        block_length = next(param for param in diffusion.params if param.key == "block_length")

        self.assertEqual(block_length.default, 256)


if __name__ == "__main__":
    unittest.main()
