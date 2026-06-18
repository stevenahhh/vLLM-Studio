import unittest
from unittest import mock

from ..app import config
from ..app.schemas import LoadRequest
from ..app.vllm_manager import VLLMManager


def _load_request() -> LoadRequest:
    return LoadRequest(
        repo="Qwen/Qwen2.5-0.5B-Instruct",
        revision="main",
        quant="awq",
        dtype="auto",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.91,
        max_model_len=2048,
        max_num_seqs=8,
        kv_cache_dtype="fp8",
        enforce_eager=True,
        trust_remote_code=True,
        extra_args=["--attention-backend", "TRITON_ATTN"],
    )


class VLLMManagerRunnerTests(unittest.TestCase):
    def test_process_runner_preserves_python_vllm_command(self) -> None:
        with (
            mock.patch.object(config, "VLLM_ENGINE_RUNNER", "process"),
            mock.patch.object(config, "VLLM_TURBOQUANT", False),
        ):
            manager = VLLMManager()

            argv = manager._build_argv(_load_request(), "/models/qwen", "qwen")

        self.assertEqual(argv[1:3], ["-m", "vllm.entrypoints.openai.api_server"])
        self.assertIn("--model", argv)
        self.assertIn("/models/qwen", argv)
        self.assertIn("--served-model-name", argv)
        self.assertIn("qwen", argv)

    def test_process_runner_uses_turboquant_wrapper_when_enabled(self) -> None:
        with (
            mock.patch.object(config, "VLLM_ENGINE_RUNNER", "process"),
            mock.patch.object(config, "VLLM_TURBOQUANT", True),
        ):
            manager = VLLMManager()

            argv = manager._build_argv(_load_request(), "/models/qwen", "qwen")

        self.assertEqual(argv[1:3], ["-m", "app.turboquant_entrypoint"])
        self.assertIn("--model", argv)
        self.assertIn("/models/qwen", argv)
        self.assertIn("--served-model-name", argv)
        self.assertIn("qwen", argv)

    def test_docker_runner_ignores_turboquant_wrapper(self) -> None:
        with (
            mock.patch.object(config, "VLLM_ENGINE_RUNNER", "docker"),
            mock.patch.object(config, "VLLM_TURBOQUANT", True),
            mock.patch.object(config, "VLLM_DOCKER_IMAGE", "vllm/vllm-openai:v-test"),
            mock.patch.object(config, "HF_TOKEN", None),
        ):
            manager = VLLMManager()
            argv = manager._build_argv(_load_request(), "Qwen/Qwen2.5-0.5B-Instruct", "qwen")

        self.assertNotIn("app.turboquant_entrypoint", argv)
        self.assertIn("vllm/vllm-openai:v-test", argv)

    def test_docker_runner_builds_gpu_container_command(self) -> None:
        with (
            mock.patch.object(config, "VLLM_ENGINE_RUNNER", "docker"),
            mock.patch.object(config, "VLLM_DOCKER_IMAGE", "vllm/vllm-openai:v-test"),
            mock.patch.object(config, "VLLM_CONTAINER_NAME", "vllm-studio-test"),
            mock.patch.object(config, "HF_HOME", "/cache/hf"),
            mock.patch.object(config, "HF_HUB_CACHE", "/cache/hf/hub"),
            mock.patch.object(config, "HF_TOKEN", "hf_test"),
            mock.patch.object(config, "VLLM_PORT", 8001),
        ):
            manager = VLLMManager()
            argv = manager._build_argv(_load_request(), "/cache/hf/hub/models--qwen/snapshots/abc", "qwen")

        self.assertEqual(argv[:3], ["docker", "run", "--rm"])
        self.assertIn("--runtime", argv)
        self.assertIn("nvidia", argv)
        self.assertIn("--gpus", argv)
        self.assertIn("all", argv)
        self.assertIn("--ipc=host", argv)
        self.assertIn("8001:8001", argv)
        self.assertIn("/cache/hf:/cache/hf", argv)
        self.assertIn("HF_HOME=/cache/hf", argv)
        self.assertIn("HF_HUB_CACHE=/cache/hf/hub", argv)
        self.assertIn("HF_TOKEN", argv)
        self.assertFalse(any("hf_test" in part for part in argv))
        self.assertIn("vllm/vllm-openai:v-test", argv)
        self.assertIn("--host", argv)
        self.assertEqual(argv[argv.index("--host") + 1], "0.0.0.0")
        self.assertIn("--model", argv)
        self.assertIn("/cache/hf/hub/models--qwen/snapshots/abc", argv)

    def test_invalid_runner_fails_before_spawn(self) -> None:
        with mock.patch.object(config, "VLLM_ENGINE_RUNNER", "podman"):
            manager = VLLMManager()

            with self.assertRaisesRegex(ValueError, "unsupported VLLM_ENGINE_RUNNER"):
                manager._build_argv(_load_request(), "Qwen/Qwen2.5-0.5B-Instruct", "qwen")

    def test_docker_log_redacts_token(self) -> None:
        with (
            mock.patch.object(config, "VLLM_ENGINE_RUNNER", "docker"),
            mock.patch.object(config, "VLLM_DOCKER_IMAGE", "vllm/vllm-openai:v-test"),
            mock.patch.object(config, "HF_TOKEN", "hf_secret_token"),
        ):
            manager = VLLMManager()
            argv = manager._build_argv(_load_request(), "Qwen/Qwen2.5-0.5B-Instruct", "qwen")
            manager._append_log("[vllm-studio] launching: " + " ".join(manager._argv_for_log(argv)))

        log_text = "\n".join(manager.log_lines())
        self.assertIn("HF_TOKEN", log_text)
        self.assertNotIn("hf_secret_token", log_text)

    def test_docker_runner_mounts_hub_cache_outside_hf_home(self) -> None:
        with (
            mock.patch.object(config, "VLLM_ENGINE_RUNNER", "docker"),
            mock.patch.object(config, "HF_HOME", "/cache/hf"),
            mock.patch.object(config, "HF_HUB_CACHE", "/mnt/models/hub"),
            mock.patch.object(config, "HF_TOKEN", None),
        ):
            manager = VLLMManager()
            argv = manager._build_argv(_load_request(), "/mnt/models/hub/models--qwen/snapshots/abc", "qwen")

        self.assertIn("/cache/hf:/cache/hf", argv)
        self.assertIn("/mnt/models/hub:/mnt/models/hub", argv)

    def test_docker_runner_mounts_external_local_model_read_only(self) -> None:
        with (
            mock.patch.object(config, "VLLM_ENGINE_RUNNER", "docker"),
            mock.patch.object(config, "HF_HOME", "/cache/hf"),
            mock.patch.object(config, "HF_HUB_CACHE", "/cache/hf/hub"),
            mock.patch.object(config, "HF_TOKEN", None),
        ):
            manager = VLLMManager()
            argv = manager._build_argv(_load_request(), "/models/local-qwen", "qwen")

        self.assertIn("/models/local-qwen:/models/local-qwen:ro", argv)


if __name__ == "__main__":
    unittest.main()
