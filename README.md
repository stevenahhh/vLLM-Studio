# vLLM Studio

vLLM Studio는 vLLM으로 로컬 LLM을 실행하기 위한 셀프 호스팅 풀스택 앱입니다.
FastAPI 제어 서버, Next.js/shadcn 대시보드, 모델 다운로드 도구, GPU 텔레메트리,
엔진 제어, 채팅, 그리고 모델 로드 전 KV 캐시 기반 VRAM 예측을 제공합니다.

```text
Next.js dashboard (:3000)
        |
        | HTTP / SSE
        v
FastAPI control plane (:8000)
        |
        | starts and monitors
        v
vLLM OpenAI-compatible engine (:8001)
```

## Linux 요구사항

대상 플랫폼은 NVIDIA GPU가 있는 Linux입니다. Ubuntu 22.04 또는 24.04를 가장
안전한 기준으로 봅니다.

필수 항목:

- `nvidia-smi`에서 보이는 NVIDIA 드라이버
- Python 3.10 또는 3.11
- Bun 1.1 이상
- Git
- vLLM/Torch 스택이 런타임 CUDA 컴파일을 요구하는 경우 CUDA 12.x toolkit
- Hugging Face 모델 가중치를 저장할 충분한 디스크 공간

운영 또는 원격 호스트에서 권장하는 항목:

- 백엔드와 프론트엔드를 유지하기 위한 `tmux` 또는 `systemd`
- 대용량 비모델 자산을 추가할 가능성이 있다면 `git-lfs`
- 선택적 Docker vLLM runner를 사용할 경우 Docker 및 NVIDIA Container Toolkit

모델 가중치, 런타임 JSON 상태, 로컬 `.env` 파일, Hugging Face 토큰은 커밋하지
마세요. 루트 `.gitignore`가 기본적으로 이를 제외하도록 설정되어 있습니다.

## Linux 새 설치

프로젝트를 clone하고 루트 디렉터리로 이동합니다.

```bash
git clone <your-repo-url> vllm-studio
cd vllm-studio
```

백엔드용 Python 환경을 만듭니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
```

프론트엔드 의존성을 설치합니다.

```bash
cd frontend
bun install
cd ..
```

커밋된 예제 파일에서 로컬 환경 파일을 만듭니다.

```bash
cp .env.example .env.local
cp frontend/.env.example frontend/.env.local
```

호스트 환경에 맞게 로컬 파일을 수정합니다. 다른 머신에서 UI를 열 경우 최소한
다음 값을 설정하세요.

```bash
NEXT_PUBLIC_API_BASE=http://<linux-host>:8000
NEXT_ALLOWED_DEV_ORIGINS=<linux-host>,localhost,127.0.0.1
```

gated 또는 private Hugging Face 저장소를 사용하려면 토큰을 shell에 export하거나
커밋되지 않는 로컬 env 파일에만 보관하세요.

```bash
export HF_TOKEN=hf_...
```

## 로컬 실행

백엔드 제어 서버를 시작합니다.

```bash
scripts/run-backend.sh
```

두 번째 터미널에서 대시보드를 시작합니다.

```bash
scripts/run-frontend.sh dev
```

브라우저에서 `http://localhost:3000`을 엽니다. 백엔드 API 문서는
`http://localhost:8000/docs`에서 확인할 수 있습니다.

두 계층을 한 번에 시작할 수도 있습니다.

```bash
scripts/run-all.sh
```

제어 서버가 vLLM 엔진을 관리합니다. 앱의 엔진 설정을 바꾸지 않았다면 같은 엔진
포트에서 별도의 vLLM OpenAI 서버를 직접 띄우지 마세요.

## 프로덕션 빌드

프론트엔드를 빌드합니다.

```bash
cd frontend
bun run build
cd ..
```

백엔드와 프로덕션 프론트엔드를 각각 별도의 supervised process로 실행합니다.

```bash
scripts/run-backend.sh
scripts/run-frontend.sh start
```

원격 장기 실행 환경에서는 `systemd`, `supervisord`, `tmux` 같은 프로세스
매니저를 사용하세요. 브라우저가 접근할 수 있는 백엔드 주소를
`NEXT_PUBLIC_API_BASE`로 유지해야 합니다.

## 모델 저장소

백엔드는 `HF_HOME`이 설정되어 있으면 그 값을 사용하고, 없으면
`~/.cache/huggingface`를 사용합니다.

모델용 대용량 디스크가 있는 머신에서는 다음처럼 설정합니다.

```bash
export HF_HOME=/mnt/data/hf-cache
mkdir -p "$HF_HOME"
```

다운로드, 로컬 모델 스캔, 선택된 vLLM runner는 모두 `HF_HOME`을 따릅니다. 모델
가중치를 repository 내부에 저장하지 마세요.

## Smoke Test

작은 모델을 다운로드합니다.

```bash
scripts/pull-smoke-model.sh
```

그 다음 대시보드에서 **Load model**을 열고 다운로드된 모델을 선택해 로드합니다.
API 레벨 테스트가 필요하면 백엔드 `/docs`를 사용하거나 UI가 보내는 것과 같은
모델 설정으로 `/api/engine/load` endpoint를 호출하세요.

## 선택적 Docker vLLM Runner

기본 runner는 vLLM을 host process로 시작합니다. FastAPI 제어 서버는 host에 두고
vLLM 엔진만 Docker에서 실행하려면 다음을 설정합니다.

```bash
export VLLM_ENGINE_RUNNER=docker
export VLLM_DOCKER_IMAGE=vllm/vllm-openai:latest
scripts/run-backend.sh
```

Docker mode는 NVIDIA GPU, host IPC, `8001` 포트, 설정된 Hugging Face cache를
사용합니다. `HF_TOKEN`이 설정되어 있으면 launch log에 값을 쓰지 않고 environment
name으로만 전달합니다.

## 선택적 TurboQuant

TurboQuant 지원은 host process runner에서만 사용할 수 있습니다.

백엔드 Python 환경에 설치합니다.

```bash
pip install "turboquant[vllm,triton] @ git+https://github.com/0xSero/turboquant.git"
```

백엔드를 시작하기 전에 활성화합니다.

```bash
export VLLM_ENGINE_RUNNER=process
export VLLM_TURBOQUANT=1
scripts/run-backend.sh
```

기본 TurboQuant 설정:

```bash
VLLM_TURBOQUANT_KEY_BITS=3
VLLM_TURBOQUANT_VALUE_BITS=2
VLLM_TURBOQUANT_BUFFER_SIZE=128
VLLM_TURBOQUANT_INITIAL_LAYERS=4
```

라이선스 참고: TurboQuant는 `https://github.com/0xSero/turboquant`의 선택적
third-party dependency입니다. upstream repository는 GPL-3.0으로 라이선스되어
있습니다. 이 프로젝트는 TurboQuant를 vendoring하지 않고 기본 설치하지도 않습니다.
TurboQuant를 포함한 bundle 또는 image를 배포한다면 해당 라이선스를 검토하고
준수하세요. 자세한 내용은 `THIRD_PARTY_NOTICES.md`를 참고하세요.

## 주요 기능

- vLLM OpenAI-compatible API를 통한 로드된 모델과의 채팅
- 대시보드에서 Hugging Face 모델 및 quantized variant 다운로드
- 실시간 GPU 메모리와 총 VRAM 사용량 모니터링
- 로드 전 VRAM, KV cache 크기, OOM 위험 예측
- quantization, dtype, context length, tensor parallelism, GPU memory utilization,
  max sequences, KV-cache dtype, eager mode 설정
- load-time system prompt 저장 및 optional custom prompt override
- vLLM `extra_body`를 통한 표준 LLM parameter와 diffusion model parameter 지원

## 하드웨어 감지

Dashboard의 Hardware 페이지는 README에 하드코딩된 내용이 아니라 백엔드 감지
결과로 생성됩니다. 백엔드는 가능하면 NVML을 사용하고, 하드웨어 정보를 감지하지
못하면 보수적인 fallback을 사용합니다. Recommendations는 시작 가이드로 봐야 하며,
모든 모델이 반드시 맞는다는 보장은 아닙니다.

## VRAM 예측

Estimator는 모델 가중치, KV cache, activation, CUDA graph overhead, runtime
overhead, tensor parallelism, context length, 설정된 GPU memory utilization을
반영합니다.

단순화한 GPU별 계산 형태:

```text
weights_per_gpu   = weights_total / tensor_parallel_size
kv_per_token      = 2 * layers * kv_heads * head_dim * kv_bytes
kv_per_gpu        = kv_per_token * max_model_len * concurrency / tensor_parallel_size
required_per_gpu  = weights_per_gpu + kv_per_gpu + activations + overhead
budget_per_gpu    = gpu_total * gpu_memory_utilization
```

전체 API와 estimation contract는 `CONTRACT.md`를 참고하세요.

## 프로젝트 구조

```text
backend/app/      FastAPI control plane
backend/tests/    Backend unit tests
frontend/         Next.js 16 + shadcn dashboard
scripts/          Local run and smoke-test helpers
CONTRACT.md       API and behavior contract
```

중요한 포트:

- `3000`: Next.js dashboard
- `8000`: FastAPI control plane
- `8001`: managed vLLM engine

## API

모든 control-plane route는 `/api` 아래에 있습니다.

주요 route:

- `GET /api/hardware`
- `GET /api/gpu/stats`
- `GET /api/gpu/stream`
- `GET /api/models/downloaded`
- `GET /api/models/search`
- `GET /api/models/variants`
- `GET /api/models/meta`
- `POST /api/estimate`
- `GET|POST /api/downloads`
- `GET|POST /api/engine`
- `POST /api/engine/load`
- `POST /api/engine/unload`
- `GET /api/engine/logs`
- `GET /api/params/schema`
- `GET|PUT /api/settings`
- `POST /api/chat/completions`

전체 schema: `http://localhost:8000/docs`.

## 검증

백엔드 command-building test:

```bash
PYTHONPATH="$PWD" python -m unittest backend.tests.test_vllm_manager -v
```

프론트엔드 check:

```bash
cd frontend
bun run typecheck
bun run lint
cd ..
```

GitHub Actions는 FastAPI, Next.js, vLLM 서버를 시작하지 않고 같은 lightweight
backend test와 frontend check를 실행합니다.

## GitHub 배포 체크리스트

이 프로젝트는 full-stack app이므로 root-level monorepo로 배포하는 것이 맞습니다.

첫 root commit 전 확인할 항목:

1. 기존 nested `frontend/.git` directory를 정리합니다. 그대로 두면 Git이
   `frontend/`를 일반 project file이 아니라 embedded repository로 취급합니다.
2. 실제 runtime state를 Git에 넣지 않습니다: `data/*.json`, `.env.local`,
   `frontend/.env.local`, `.omo/`, `.codegraph`, model weights, caches.
3. root `LICENSE`와 `THIRD_PARTY_NOTICES.md`를 유지합니다. 이 프로젝트는 MIT
   license이며, optional TurboQuant integration은 TurboQuant 자체가 GPL-3.0이므로
   별도로 문서화되어 있습니다.
4. 위 검증 명령을 실행합니다.
5. commit 전 최종 add set을 확인합니다.

```bash
git status --short
git diff --cached --stat
```

추천 첫 commit:

```text
chore(repo): prepare publishable monorepo
feat(engine): add vllm studio runtime
docs(readme): document linux installation
ci(github): add backend and frontend checks
```

remote URL, repository visibility, branch name이 의도한 값인지 확인한 뒤 push하세요.

## 라이선스

vLLM Studio는 MIT License로 배포됩니다. `LICENSE`를 참고하세요.

TurboQuant는 선택적으로 별도 설치되는 third-party integration이며 upstream은
GPL-3.0으로 라이선스되어 있습니다. `THIRD_PARTY_NOTICES.md`를 참고하세요.
