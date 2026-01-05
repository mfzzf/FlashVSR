# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASHVSR_DEVICE=cuda \
    FLASHVSR_DTYPE=bfloat16 \
    FLASHVSR_V1_DIR=/models/FlashVSR \
    FLASHVSR_V1_1_DIR=/models/FlashVSR-v1.1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    ninja-build \
    cmake \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

COPY . /app

# Install torch wheels from the official CUDA index (required by requirements.txt pins).
RUN pip install --upgrade pip \
  && pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt \
  && pip install -e . --no-deps \
  && pip install -r api/requirements.txt

# Optional but recommended: install Block-Sparse Attention backend.
ARG INSTALL_BLOCK_SPARSE_ATTN=1
RUN if [ "$INSTALL_BLOCK_SPARSE_ATTN" = "1" ]; then \
      pip install packaging ninja && \
      pip install git+https://github.com/mit-han-lab/Block-Sparse-Attention.git ; \
    fi

RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "flashvsr_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
