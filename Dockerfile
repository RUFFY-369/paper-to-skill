# Dockerfile for Synthesizer Agent in Paper-to-Skill Meta-Compiler Pipeline

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install basic developer tools and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set standard working directory for mounting scripts or code execution
WORKDIR /workspace

# Default command
CMD ["python3", "-c", "import torch; print(f'PyTorch + CUDA ready. GPUs available: {torch.cuda.is_available()}')"]
