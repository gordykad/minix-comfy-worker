#!/bin/bash
# MiniX Lane A worker start: link the volume's model cache into ComfyUI,
# boot ComfyUI headless, then run the RunPod handler.
set -e
VOLUME="${VOLUME_PATH:-/runpod-volume}"
if [ -d "$VOLUME/models" ]; then
  for sub in diffusion_models text_encoders vae loras; do
    mkdir -p "$VOLUME/models/$sub" /comfyui/models/$sub
    cp -rs "$VOLUME/models/$sub/"* /comfyui/models/$sub/ 2>/dev/null || true
  done
fi
cd /comfyui
python3 main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch &
cd /worker
python3 handler.py
