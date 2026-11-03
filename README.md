# MiniX Lane A worker (ticket 28)

RunPod serverless worker for Astro MiniX's Image Gen tab, Lane A.
ComfyUI + ComfyUI-GGUF running Qwen-Image 2.1 Uncensored (GGUF).

- Input contract (from the app): `{prompt, width, height, steps, seed, model}`
  where `model` is a Hugging Face GGUF repo id.
- Output: `{"images": ["<base64 png>"], "seed": n}`.
- Model files cache on the attached network volume (`VOLUME_PATH`,
  default `/runpod-volume`) under `models/{diffusion_models,
  text_encoders,vae}`; first job on a cold volume downloads ~13 GB.
- Fits a 24 GB GPU (Q4_K_M diffusion + int8 text encoder).
- Image builds on push via GitHub Actions to
  `ghcr.io/gordykad/minix-comfy-worker:latest`.

Deploy: create a RunPod serverless endpoint from that image with a
network volume attached and `VOLUME_PATH` set to its mount; paste the
endpoint ID + a RunPod API key into the app's lane settings (owner's
device only - Amendment 9).
