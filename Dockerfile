FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Maintainer / metadata
LABEL org.opencontainers.image.source="https://github.com/cdcd09/FoleyCrafter-1" \
      org.opencontainers.image.description="Reproducible runtime for FoleyCrafter (PyTorch 2.2.0 CUDA 11.8)" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    MPLCONFIGDIR=/workspace/.cache/matplotlib

WORKDIR /workspace

# System dependencies (runtime only; add build-essential if compiling extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy requirement specs first (leverage layer caching)
COPY requirements/requirements.full-freeze.txt requirements/requirements.txt ./requirements/

# Install Python dependencies exactly as frozen
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements/requirements.full-freeze.txt \
 && python -c "import torch, torchvision, torchaudio; print('Torch:', torch.__version__)"

# Copy project source
COPY . .

# (Optional) create a non-root user – commented out for simplicity
# RUN useradd -m foley && chown -R foley:foley /workspace
# USER foley

EXPOSE 7860

# Default entrypoint: Gradio app; override with `docker run ... python inference.py ...`
ENTRYPOINT ["python", "app.py"]
