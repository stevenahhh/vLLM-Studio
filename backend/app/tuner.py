"""Bayesian vLLM parameter tuner using Optuna.

Runs a multi-objective optimisation study in a background thread:
  - Loads the target model via VLLMManager with each trial's params
  - Sends concurrent HTTP requests to the live engine and measures
    throughput / latency / memory
  - Returns ranked results + best-params recommendation

Public API (used by main.py):
    tuner = VLLMTuner(manager)
    tuner.start(config)     -> TunerStatus
    tuner.stop()            -> TunerStatus
    tuner.status()          -> TunerStatus
"""
from __future__ import annotations

import logging
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import httpx
import optuna
from pydantic import BaseModel, Field

from . import config as cfg

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TunerConfig(BaseModel):
    repo: str
    revision: str = "main"
    quant: str = "none"
    dtype: str = "float16"
    tensor_parallel_size: int = 8
    extra_args: list[str] = Field(default_factory=list)

    # Objectives (weights must sum to 100)
    throughput_weight: int = 60
    latency_weight: int = 30
    memory_weight: int = 10

    # Search space bounds
    gpu_memory_utilization_min: float = 0.70
    gpu_memory_utilization_max: float = 0.95
    max_num_seqs_min: int = 4
    max_num_seqs_max: int = 128
    max_num_batched_tokens_min: int = 2048
    max_num_batched_tokens_max: int = 32768
    max_model_len: int = 8192

    # Study settings
    n_trials: int = 20
    timeout_minutes: int = 120

    # Benchmark workload
    concurrent_requests: int = 10
    requests_per_trial: int = 50
    max_tokens: int = 256
    prompts: list[str] = Field(default_factory=list)


class TrialResult(BaseModel):
    trial_id: int
    state: str          # completed / failed / pruned
    params: dict
    throughput: float   # req/s
    avg_latency_ms: float
    p99_latency_ms: float
    memory_util: float  # fraction 0-1
    score: float        # composite (higher = better)
    error: str = ""


class TunerStatus(BaseModel):
    state: str          # idle / running / done / stopped / error
    current_trial: int
    total_trials: int
    trials: list[TrialResult]
    best_params: Optional[dict]
    best_score: float
    elapsed_seconds: float
    error: str


# ---------------------------------------------------------------------------
# Default prompts used when the user supplies none
# ---------------------------------------------------------------------------

_DEFAULT_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the main causes of the French Revolution?",
    "Describe the process of photosynthesis.",
    "How does a transformer neural network work?",
    "Write a SQL query to find the top 10 customers by revenue.",
    "What is the difference between TCP and UDP?",
    "Explain gradient descent in machine learning.",
    "Write a bash script to monitor CPU usage every 5 seconds.",
    "What are the SOLID principles in software engineering?",
    "Translate 'Hello, how are you?' into French, Spanish, and Japanese.",
    "What is the time complexity of quicksort?",
    "Explain the CAP theorem in distributed systems.",
    "Write a React component that fetches data from an API.",
    "What are the differences between Docker and Kubernetes?",
    "Describe the Turing test and its implications.",
    "How does HTTPS encryption work?",
    "Write a function to reverse a linked list in Python.",
    "What is the difference between supervised and unsupervised learning?",
    "Explain what a neural network activation function does.",
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _run_benchmark(
    base_url: str,
    model_name: str,
    prompts: list[str],
    concurrent: int,
    max_tokens: int,
    n_requests: int,
) -> dict:
    """Send n_requests to the engine concurrently, return timing metrics."""
    # Cycle through prompts
    prompt_list = [prompts[i % len(prompts)] for i in range(n_requests)]
    latencies: list[float] = []
    errors = 0

    def send_one(prompt: str) -> Optional[float]:
        t0 = time.perf_counter()
        try:
            resp = httpx.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 1.0,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            return (time.perf_counter() - t0) * 1000  # ms
        except Exception as exc:
            logger.debug("bench request failed: %s", exc)
            return None

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        futs = [pool.submit(send_one, p) for p in prompt_list]
        for f in as_completed(futs):
            lat = f.result()
            if lat is None:
                errors += 1
            else:
                latencies.append(lat)
    t_elapsed = time.perf_counter() - t_start

    if not latencies:
        return {
            "throughput": 0.0,
            "avg_latency_ms": float("inf"),
            "p99_latency_ms": float("inf"),
            "error_rate": 1.0,
        }

    latencies.sort()
    p99_idx = max(0, int(len(latencies) * 0.99) - 1)
    return {
        "throughput": len(latencies) / t_elapsed,
        "avg_latency_ms": statistics.mean(latencies),
        "p99_latency_ms": latencies[p99_idx],
        "error_rate": errors / n_requests,
    }


# ---------------------------------------------------------------------------
# Main tuner class
# ---------------------------------------------------------------------------

class VLLMTuner:
    """Run an Optuna study against the managed vLLM engine."""

    def __init__(self, manager) -> None:
        self._manager = manager
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._state = "idle"
        self._trials: list[TrialResult] = []
        self._current_trial = 0
        self._total_trials = 0
        self._best_params: Optional[dict] = None
        self._best_score = 0.0
        self._start_ts = 0.0
        self._error = ""
        self._config: Optional[TunerConfig] = None

    # -- public API -----------------------------------------------------------

    def start(self, config: TunerConfig) -> TunerStatus:
        with self._lock:
            if self._state == "running":
                raise RuntimeError("Tuner is already running")
            self._config = config
            self._state = "running"
            self._trials = []
            self._current_trial = 0
            self._total_trials = config.n_trials
            self._best_params = None
            self._best_score = 0.0
            self._start_ts = time.monotonic()
            self._error = ""
            self._stop_event.clear()

        self._thread = threading.Thread(target=self._run, args=(config,), daemon=True)
        self._thread.start()
        return self.status()

    def stop(self) -> TunerStatus:
        self._stop_event.set()
        with self._lock:
            if self._state == "running":
                self._state = "stopped"
        return self.status()

    def status(self) -> TunerStatus:
        with self._lock:
            return TunerStatus(
                state=self._state,
                current_trial=self._current_trial,
                total_trials=self._total_trials,
                trials=list(self._trials),
                best_params=self._best_params,
                best_score=round(self._best_score, 4),
                elapsed_seconds=round(time.monotonic() - self._start_ts, 1)
                if self._start_ts else 0.0,
                error=self._error,
            )

    # -- internals ------------------------------------------------------------

    def _composite_score(self, throughput: float, avg_lat: float, mem_util: float, cfg_: TunerConfig) -> float:
        """Weighted composite score in [0, 1] (higher = better)."""
        # Normalise each objective to [0, 1]
        thr_score = min(1.0, throughput / 50.0)          # 50 req/s = perfect
        lat_score = max(0.0, 1.0 - avg_lat / 10000.0)    # 10 s = 0
        mem_score = max(0.0, 1.0 - mem_util)             # lower util = better

        w = cfg_.throughput_weight + cfg_.latency_weight + cfg_.memory_weight
        if w == 0:
            return 0.0
        return (
            cfg_.throughput_weight * thr_score
            + cfg_.latency_weight * lat_score
            + cfg_.memory_weight * mem_score
        ) / w

    def _wait_engine_ready(self, timeout: int = 600) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                return False
            st = self._manager.status()
            if st.state == "ready":
                return True
            if st.state == "error":
                return False
            time.sleep(2.0)
        return False

    def _get_memory_util(self) -> float:
        """Best-effort: average used/total across all GPUs via pynvml."""
        try:
            import pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            vals = []
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                vals.append(info.used / info.total)
            return statistics.mean(vals) if vals else 0.0
        except Exception:
            return 0.0

    def _run(self, tuner_cfg: TunerConfig) -> None:
        prompts = tuner_cfg.prompts or _DEFAULT_PROMPTS
        model_name = tuner_cfg.repo.split("/")[-1] or tuner_cfg.repo
        base_url = cfg.VLLM_BASE_URL

        def objective(trial: optuna.Trial) -> tuple[float, float, float]:
            if self._stop_event.is_set():
                raise optuna.exceptions.TrialPruned()

            trial_id = trial.number
            with self._lock:
                self._current_trial = trial_id + 1

            params = {
                "gpu_memory_utilization": trial.suggest_float(
                    "gpu_memory_utilization",
                    tuner_cfg.gpu_memory_utilization_min,
                    tuner_cfg.gpu_memory_utilization_max,
                    step=0.01,
                ),
                "max_num_seqs": trial.suggest_int(
                    "max_num_seqs",
                    tuner_cfg.max_num_seqs_min,
                    tuner_cfg.max_num_seqs_max,
                    log=True,
                ),
                "max_num_batched_tokens": trial.suggest_int(
                    "max_num_batched_tokens",
                    tuner_cfg.max_num_batched_tokens_min,
                    tuner_cfg.max_num_batched_tokens_max,
                    step=512,
                ),
            }

            self._append_log(f"[tuner] trial {trial_id}: {params}")

            # Build a LoadRequest-compatible dict
            from .schemas import LoadRequest
            req = LoadRequest(
                repo=tuner_cfg.repo,
                revision=tuner_cfg.revision,
                quant=tuner_cfg.quant,
                dtype=tuner_cfg.dtype,
                tensor_parallel_size=tuner_cfg.tensor_parallel_size,
                gpu_memory_utilization=params["gpu_memory_utilization"],
                max_model_len=tuner_cfg.max_model_len,
                max_num_seqs=params["max_num_seqs"],
                enforce_eager=False,
                extra_args=[
                    "--max-num-batched-tokens",
                    str(params["max_num_batched_tokens"]),
                ] + list(tuner_cfg.extra_args),
            )
            self._manager.load(req)

            ready = self._wait_engine_ready(timeout=600)
            if not ready:
                err = self._manager.status().error or "engine did not become ready"
                self._record_trial(trial_id, params, 0.0, float("inf"), float("inf"), 1.0, 0.0, err)
                raise optuna.exceptions.TrialPruned()

            # Warm-up (5 requests)
            _run_benchmark(base_url, model_name, prompts, 2, tuner_cfg.max_tokens, 5)

            mem_util = self._get_memory_util()
            metrics = _run_benchmark(
                base_url,
                model_name,
                prompts,
                tuner_cfg.concurrent_requests,
                tuner_cfg.max_tokens,
                tuner_cfg.requests_per_trial,
            )

            throughput = metrics["throughput"]
            avg_lat = metrics["avg_latency_ms"]
            p99_lat = metrics["p99_latency_ms"]
            score = self._composite_score(throughput, avg_lat, mem_util, tuner_cfg)

            self._record_trial(trial_id, params, throughput, avg_lat, p99_lat, mem_util, score, "")
            self._append_log(
                f"[tuner] trial {trial_id} done: thr={throughput:.2f} lat={avg_lat:.0f}ms score={score:.3f}"
            )

            # Optuna multi-objective: maximise throughput, minimise latency, minimise mem
            return throughput, avg_lat, mem_util

        try:
            sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
            study = optuna.create_study(
                directions=["maximize", "minimize", "minimize"],
                sampler=sampler,
            )
            study.optimize(
                objective,
                n_trials=tuner_cfg.n_trials,
                timeout=tuner_cfg.timeout_minutes * 60,
                catch=(Exception,),
            )
            with self._lock:
                if self._state == "running":
                    self._state = "done"
                self._set_best()
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._error = str(exc)
            logger.exception("tuner study failed")
        finally:
            # Leave engine in stopped state after study
            try:
                self._manager.unload()
            except Exception:
                pass

    def _record_trial(
        self,
        trial_id: int,
        params: dict,
        throughput: float,
        avg_lat: float,
        p99_lat: float,
        mem_util: float,
        score: float,
        error: str,
    ) -> None:
        result = TrialResult(
            trial_id=trial_id,
            state="failed" if error else "completed",
            params=params,
            throughput=round(throughput, 3),
            avg_latency_ms=round(avg_lat, 1),
            p99_latency_ms=round(p99_lat, 1),
            memory_util=round(mem_util, 3),
            score=round(score, 4),
            error=error,
        )
        with self._lock:
            self._trials.append(result)
            if score > self._best_score:
                self._best_score = score
                self._best_params = params

    def _set_best(self) -> None:
        if not self._trials:
            return
        best = max((t for t in self._trials if t.state == "completed"), key=lambda t: t.score, default=None)
        if best:
            self._best_params = best.params
            self._best_score = best.score

    def _append_log(self, line: str) -> None:
        logger.info(line)


tuner: Optional[VLLMTuner] = None


def get_tuner(manager) -> VLLMTuner:
    global tuner
    if tuner is None:
        tuner = VLLMTuner(manager)
    return tuner
