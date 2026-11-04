# MiniX Lane A worker (ticket 28): RunPod serverless + ComfyUI-GGUF.
# Contract (matches the app client): job input
#   {prompt, width, height, steps, seed, model}
#     model = a Hugging Face GGUF repo id (the app entry's link)
#   -> {"images": ["<base64 png>"]}
# Model files cache on the network volume (VOLUME_PATH, default
# /runpod-volume) so cold starts after the first are fast. The text
# encoder + VAE always come from the seed repo (they are family parts);
# the diffusion GGUF comes from the requested repo (Q4_K_M preferred).
import base64
import json
import os
import random
import time
import urllib.parse
import urllib.request

import requests
import runpod

COMFY = "http://127.0.0.1:8188"
VOLUME = os.environ.get("VOLUME_PATH", "/runpod-volume")
MODELS = os.path.join(VOLUME, "models")
SEED_REPO = "KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF"
# The original repo (abenzerps/Qwen-Image-2.1-Uncensored-GGUF) was renamed
# by its author to a Turbo-quant repo on 2026-11-03; the name now resolves
# to different weights. The mirror above preserves the original UC files.
ALIASES = {"abenzerps/Qwen-Image-2.1-Uncensored-GGUF": SEED_REPO}
ENCODER = "text_encoders/qwen3vl_8b_int8_convrot.safetensors"
VAE = "vae/qwen_image_2.1_vae_bf16.safetensors"


def hf_file(repo, path, dest_dir):
    dest = os.path.join(dest_dir, os.path.basename(path))
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    os.makedirs(dest_dir, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    os.rename(tmp, dest)
    return dest


def repo_files(repo):
    with urllib.request.urlopen(
        f"https://huggingface.co/api/models/{repo}", timeout=30
    ) as r:
        d = json.load(r)
    return [s["rfilename"] for s in d.get("siblings", [])]


def ensure_models(repo):
    """Download (once) everything the graph needs; return unet file name."""
    hf_file(SEED_REPO, ENCODER, os.path.join(MODELS, "text_encoders"))
    hf_file(SEED_REPO, VAE, os.path.join(MODELS, "vae"))
    files = repo_files(repo)
    ggufs = [f for f in files if f.endswith(".gguf") and "/" not in f]
    if not ggufs:
        raise RuntimeError(f"repo {repo} has no top-level .gguf")
    pick = next((f for f in ggufs if "Q4_K_M" in f), ggufs[0])
    hf_file(repo, pick, os.path.join(MODELS, "diffusion_models"))
    return pick


COMFY_MODELS = "/comfyui/models"


def link_into_comfy():
    """ComfyUI only sees its own models tree. start.sh links the volume
    cache in at container boot, but files downloaded afterwards (the
    first job on a fresh volume) never got linked - the exact failure of
    2026-11-03 21:50, when ComfyUI rejected the prompt with every model
    list empty while the files sat complete on the volume. Link at job
    time, right before prompting."""
    for sub in ("diffusion_models", "text_encoders", "vae", "loras"):
        src_dir = os.path.join(MODELS, sub)
        dst_dir = os.path.join(COMFY_MODELS, sub)
        if not os.path.isdir(src_dir):
            continue
        os.makedirs(dst_dir, exist_ok=True)
        for name in os.listdir(src_dir):
            if name.endswith(".part"):
                continue
            dst = os.path.join(dst_dir, name)
            if not os.path.exists(dst):
                os.symlink(os.path.join(src_dir, name), dst)


def wait_for_comfy(timeout=420):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(COMFY + "/system_stats",
                            timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    tail = ""
    try:
        with open("/tmp/comfy.log") as f:
            tail = f.read()[-1500:]
    except OSError:
        pass
    raise RuntimeError(f"ComfyUI did not come up. Boot log tail: {tail}")


def run_workflow(job_input):
    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        raise RuntimeError("prompt is required")
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    steps = max(8, min(50, int(job_input.get("steps", 28))))
    seed = int(job_input.get("seed", 0)) or random.randint(1, 2**31 - 1)
    repo = job_input.get("model") or SEED_REPO
    repo = ALIASES.get(repo, repo)

    unet_name = ensure_models(repo)
    link_into_comfy()

    with open(os.path.join(os.path.dirname(__file__), "workflow_api.json")) as f:
        wf = json.load(f)
    wf["1"]["inputs"]["unet_name"] = unet_name
    wf["4"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["width"] = width
    wf["7"]["inputs"]["height"] = height
    wf["9"]["inputs"]["steps"] = steps
    wf["9"]["inputs"]["seed"] = seed

    r = requests.post(COMFY + "/prompt", json={"prompt": wf}, timeout=30)
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]

    deadline = time.time() + 600
    while time.time() < deadline:
        h = requests.get(COMFY + f"/history/{prompt_id}", timeout=15).json()
        if prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"comfy execution error: {status}")
            for node_out in entry.get("outputs", {}).values():
                for img in node_out.get("images", []):
                    q = urllib.parse.urlencode(
                        {"filename": img["filename"],
                         "subfolder": img.get("subfolder", ""),
                         "type": img.get("type", "output")})
                    data = requests.get(COMFY + "/view?" + q, timeout=60).content
                    return {"images": [base64.b64encode(data).decode()],
                            "seed": seed}
            raise RuntimeError("workflow finished with no image output")
        time.sleep(2)
    raise RuntimeError("generation timed out after 600s")


def handler(job):
    try:
        return run_workflow(job.get("input", {}))
    except Exception as e:  # honest failure back to the app
        return {"error": str(e)}


if __name__ == "__main__":
    wait_for_comfy()
    runpod.serverless.start({"handler": handler})
