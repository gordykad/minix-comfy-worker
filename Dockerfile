# MiniX Lane A worker: ComfyUI + ComfyUI-GGUF on a RunPod serverless GPU.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip git wget libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ComfyUI (recent master carries Qwen-Image 2.1 / Qwen3-VL support) + GGUF nodes
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /comfyui && \
    cd /comfyui && pip install --no-cache-dir -r requirements.txt && \
    git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF custom_nodes/ComfyUI-GGUF && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-GGUF/requirements.txt

RUN pip install --no-cache-dir runpod==1.7.9 requests

# Pin the triton that matches torch 2.5.1. A newer triton dragged in by the
# requirements above breaks `import torch` itself (infer_schema rejects a
# list[int] stride annotation) - ComfyUI died on import on RunPod
# 2026-11-03 before ever binding port 8188. Smoke-gate both imports here
# so a bad combination fails THIS build, never the worker.
RUN pip install --no-cache-dir "triton==3.1.0" && \
    python3 -c "import torch; print('torch import OK', torch.__version__)" && \
    cd /comfyui && python3 -c "import comfy.model_management; print('comfy import OK')"

WORKDIR /worker
COPY handler.py workflow_api.json start.sh ./
RUN chmod +x start.sh
CMD ["/worker/start.sh"]
