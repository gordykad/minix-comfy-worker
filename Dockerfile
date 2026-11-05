# MiniX Lane A worker: ComfyUI + ComfyUI-GGUF on a RunPod serverless GPU.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip git wget libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128

# ComfyUI (recent master carries Qwen-Image 2.1 / Qwen3-VL support) + GGUF nodes
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /comfyui && \
    cd /comfyui && pip install --no-cache-dir -r requirements.txt && \
    git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF custom_nodes/ComfyUI-GGUF && \
    pip install --no-cache-dir -r custom_nodes/ComfyUI-GGUF/requirements.txt

RUN pip install --no-cache-dir runpod==1.7.9 requests

# Current ComfyUI needs a modern torch: its comfy_kitchen package registers
# custom ops annotated with builtin list[int], which torch 2.5.1's
# infer_schema rejects - ComfyUI died on import on RunPod 2026-11-03
# before ever binding port 8188 (an earlier triton drift broke plain
# `import torch` the same night). Smoke-gate both imports here so a bad
# combination fails THIS build, never the worker. The gate targets
# comfy_kitchen itself (the package whose op registration crashed):
# importing comfy.model_management wholesale probes CUDA at module level
# and cannot run on a GPU-less build runner.
RUN python3 -c "import torch; print('torch import OK', torch.__version__)" && \
    cd /comfyui && python3 -c "import comfy_kitchen; print('comfy-kitchen import OK')"

WORKDIR /worker
COPY handler.py workflow_api.json workflow_sdxl_api.json workflow_wan_t2v_api.json workflow_wan_i2v_api.json start.sh ./
# Every workflow graph must exist and parse, or the build fails here instead
# of the worker failing a paid job at runtime (2026-11-04: the SDXL graph
# was never COPYed, so every SDXL job died with ENOENT after its download).
RUN python3 -c "import json; [json.load(open(f)) for f in ('workflow_api.json', 'workflow_sdxl_api.json', 'workflow_wan_t2v_api.json', 'workflow_wan_i2v_api.json')]; print('workflow graphs OK')"
RUN chmod +x start.sh
CMD ["/worker/start.sh"]
