# syntax=docker/dockerfile:1
# ============================================================================
# CircuitSynth-SquareWave — HuggingFace Spaces Docker image
# ============================================================================
# Exposes a FastAPI server on port 7860 (HF Spaces default).
# The inference.py LLM agent connects to this server (or runs in-process).
#
# Build:
#   docker build -t circuitsynth .
#
# Run server:
#   docker run -p 7860:7860 circuitsynth
#
# Run inference (in-process, no server needed):
#   docker run -e API_BASE_URL=... -e MODEL_NAME=... -e HF_TOKEN=... \
#     circuitsynth python inference.py
# ============================================================================

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    ngspice \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip     pip     /usr/bin/pip3       1

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies (cached layer)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn openai openenv-core

# ---------------------------------------------------------------------------
# Copy source
# ---------------------------------------------------------------------------
COPY . .
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Environment variables (set these at runtime)
# ---------------------------------------------------------------------------
ENV PORT=7860
ENV MOCK_SIM=true
ENV API_BASE_URL=https://router.huggingface.co/hf-inference/v1/
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV HF_TOKEN=hf_placeholder
ENV TASK_NAME=squarewave-easy

# ---------------------------------------------------------------------------
# Expose HF Spaces port
# ---------------------------------------------------------------------------
EXPOSE 7860

# ---------------------------------------------------------------------------
# Healthcheck
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ---------------------------------------------------------------------------
# Entrypoint: Run inference first, then keep server alive for HF Spaces
# ---------------------------------------------------------------------------
CMD ["bash", "-c", "python inference.py && python -m uvicorn server.app:app --host 0.0.0.0 --port 7860"]
