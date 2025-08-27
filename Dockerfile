# CUDA 11.8 런타임 (Ubuntu 22.04)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ----- OS 패키지 -----
# - python3.10, pip: Ubuntu 22.04 기본 제공
# - ffmpeg: decord/moviepy 등에서 사용
# - git-lfs: 모델/가중치 LFS 받기용
# - libsndfile1: soundfile용
# - OpenCV/영상 디코딩 관련 X libs: decord/일부 시각화 안전장치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git git-lfs ffmpeg \
    libsndfile1 \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# python 최신 pip로 업그레이드
RUN python3 -m pip install --upgrade pip wheel setuptools

# ----- 작업 폴더 -----
WORKDIR /app

# (옵션) 프로젝트 소스 복사: 필요 시 주석 해제
# COPY . /app

## ----- Python 패키지 설치 (두 가지 모드) -----
## BUILD ARG: INSTALL_MODE
##   original (기본) : 기존 Dockerfile 패키지 세트 설치
##   lock            : requirements/requirements-current.txt (현재 실행 환경 freeze) 설치
## 사용 예:
##   docker build -t foley:orig   --build-arg INSTALL_MODE=original .
##   docker build -t foley:lock   --build-arg INSTALL_MODE=lock .

ARG INSTALL_MODE=original
ENV INSTALL_MODE=${INSTALL_MODE}

COPY requirements/requirements-current.txt requirements/requirements-current.txt

### original 모드: 기존 수동 나열 설치
RUN if [ "$INSTALL_MODE" = "original" ]; then \
        echo "[Build] INSTALL_MODE=original: curated list install" && \
        python3 -m pip install --no-cache-dir \
                torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
                --index-url https://download.pytorch.org/whl/cu118 && \
        python3 -m pip install --no-cache-dir \
                "numpy<2.0" protobuf \
                diffusers==0.25.1 \
                transformers==4.30.2 \
                huggingface_hub==0.23.0 \
                xformers==0.0.23.post1 \
                imageio==2.33.1 \
                decord==0.6.0 \
                einops omegaconf safetensors gradio==5.0.0 \
                tqdm==4.66.1 soundfile==0.12.1 wandb \
                moviepy==1.0.3 kornia==0.7.1 h5py==3.7.0 \
                matplotlib scipy accelerate pydub ; \
    else \
        echo "[Build] INSTALL_MODE=lock: installing from requirements-current.txt" && \
        python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements/requirements-current.txt ; \
    fi

# 간단한 설치/CUDA 체크 (실패하더라도 빌드 진행)
RUN python3 - <<'PY'
try:
    import torch, torchvision, torchaudio, transformers, diffusers, xformers
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| available:", torch.cuda.is_available())
    print("torchvision:", torchvision.__version__, "torchaudio:", torchaudio.__version__)
    print("transformers:", transformers.__version__, "diffusers:", diffusers.__version__, "xformers:", xformers.__version__)
    import huggingface_hub, matplotlib, scipy, accelerate
    print("huggingface_hub:", huggingface_hub.__version__, "matplotlib:", matplotlib.__version__, "scipy:", scipy.__version__, "accelerate:", accelerate.__version__)
except Exception as e:
    print("Sanity check error:", e)
PY

# Gradio/Web UI 등 노출 포트가 있으면 열어두기 (예: 7860)
EXPOSE 7860

# 기본 진입점: bash
CMD ["/bin/bash"]
