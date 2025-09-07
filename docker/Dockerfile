# Base image with CUDA 12.9 and cuDNN 9
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       curl \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Env defaults
ENV TORCH_HOME=/app/checkpoints \
    HF_HOME=/app/checkpoints \
    MPLCONFIGDIR=/app/.mplconfig

RUN mkdir -p /app/checkpoints /app/logs /app/outputs

CMD ["python", "src/main.py"]
