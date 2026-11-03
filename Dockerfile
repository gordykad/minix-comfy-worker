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
    git clone --depth 1 https://github.com/city96/ComfyUI-GGUF custom_nodes/ComfyUI-GGUF && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-GGUF/requirements.txt

RUN pip install --no-cache-dir runpod==1.7.9 requests

WORKDIR /worker
COPY handler.py workflow_api.json start.sh ./
RUN chmod +x start.sh
CMD ["/worker/start.sh"]
