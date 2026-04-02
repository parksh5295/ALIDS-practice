# ALIDS-practice: IDS + WGAN / NIDSGAN (CPU by default; use GPU image + CUDA wheels if needed)
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Optional: OpenMP for sklearn/scipy BLAS on some platforms (wheels usually bundle enough)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Application code (mount ./data and ./saved_models at runtime for datasets and checkpoints)
COPY . .

# Typical workflow (run from host):
#   docker compose run --rm app python train_ids.py --config configs/multi_layer_perceptron.yaml
#   docker compose run --rm app python train_nidsgan.py --name exp1 --attack Probe --normalize \
#       --surrogate_path saved_models/multi_layer_perceptron.pt --save_model saved_models
# Override with: docker compose run --rm app python train_nidsgan.py ...
CMD ["bash"]
