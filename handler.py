# MiniX Lane A worker (ticket 28): RunPod serverless + ComfyUI.
# Contract (matches the app client): job input
#   {prompt, width, height, steps, seed, model}
#     model = a Hugging Face repo id (the app entry's link)
#   -> {"images": ["<base64 png>"]}
# Two model families, routed by the repo's contents:
#   - GGUF repos (top-level .gguf): the Qwen-Image graph. The text
#     encoder + VAE always come from the seed repo (family parts); the
#     diffusion GGUF comes from the requested repo (Q4_K_M preferred).
#   - SDXL repos (a standalone .safetensors checkpoint): the classic
#     CheckpointLoaderSimple graph (workflow_sdxl_api.json), tuned for
#     the Big Lust line (DPM2 ancestral + Karras, CFG 4.5).
#   Optional, SDXL stills only: loras = [{repo, file, strength_model,
#     strength_clip}, ...] chains LoraLoader nodes after the checkpoint;
#     negative_extra appends to the built-in SDXL negative.
# Model files cache on the network volume (VOLUME_PATH, default
# /runpod-volume) so cold starts after the first are fast.
#
# Video (ticket 32): job input {kind: "video", prompt, image?, width,
# height, frames, seed, lora?, lora_strength?} -> {"videos": [b64 mp4]}.
# Wan 2.2 FP8 dual experts (T2V or I2V by whether an image is attached)
# + lightx2v 4-step Lightning LoRAs, graphs mirrored node-for-node from
# Comfy-Org's official workflow templates. The stock template negative
# bans nudity outright; this lane's negative is quality-only.
import base64
import io
import json
import os
import random
import shutil
import time
import urllib.parse
import urllib.request
import zipfile

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
SDXL_NEGATIVE = (
    "lowres, bad anatomy, bad hands, missing fingers, extra fingers, "
    "blurry, deformed, watermark, text, logo, jpeg artifacts"
)
WAN_REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
WAN_NEGATIVE = (
    "blurry, distorted, warped anatomy, extra limbs, missing limbs, "
    "static frame, frozen motion, low quality, worst quality, "
    "jpeg artifacts, watermark, text, subtitles, flicker, morphing"
)


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


def hf_file_as(repo, path, dest):
    """hf_file with an explicit destination path (the FaceID stack
    hunts its files by exact names in specific model folders)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    os.rename(tmp, dest)
    return dest


FACEID_REPO = "h94/IP-Adapter-FaceID"
CLIPVISION_NAME = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
BUFFALO_URL = ("https://github.com/deepinsight/insightface/releases/"
               "download/v0.7/buffalo_l.zip")


def ensure_faceid():
    """Cache the SDXL FaceID stack on the volume (IP-Adapter FaceID
    Plus V2 + its LoRA, the CLIP-ViT-H vision encoder, and the
    InsightFace buffalo_l face-embedding pack, ~4.7 GB total). The
    loader nodes hunt these by canonical file names, so destinations
    are exact. Volume room comes from the Qwen GGUF cache if needed -
    that family is dropped and its files are re-downloadable."""
    need = 6_500_000_000
    if shutil.disk_usage(VOLUME).free < need:
        gguf_dir = os.path.join(MODELS, "diffusion_models")
        if os.path.isdir(gguf_dir):
            for name in os.listdir(gguf_dir):
                if name.endswith(".gguf"):
                    print(f"evicting cached GGUF {name} for FaceID room")
                    try:
                        os.remove(os.path.join(gguf_dir, name))
                    except OSError:
                        pass
    hf_file_as(FACEID_REPO, "ip-adapter-faceid-plusv2_sdxl.bin",
               os.path.join(MODELS, "ipadapter",
                            "ip-adapter-faceid-plusv2_sdxl.bin"))
    hf_file_as(FACEID_REPO, "ip-adapter-faceid-plusv2_sdxl_lora.safetensors",
               os.path.join(MODELS, "loras",
                            "ip-adapter-faceid-plusv2_sdxl_lora.safetensors"))
    hf_file_as("h94/IP-Adapter", "models/image_encoder/model.safetensors",
               os.path.join(MODELS, "clip_vision", CLIPVISION_NAME))
    pack_dir = os.path.join(MODELS, "insightface", "models", "buffalo_l")
    marker = os.path.join(pack_dir, "det_10g.onnx")
    if not (os.path.exists(marker)
            and os.path.getsize(marker) > 1_000_000):
        os.makedirs(pack_dir, exist_ok=True)
        with requests.get(BUFFALO_URL, stream=True, timeout=180) as r:
            r.raise_for_status()
            blob = io.BytesIO()
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                blob.write(chunk)
        blob.seek(0)
        with zipfile.ZipFile(blob) as z:
            for info in z.infolist():
                base = os.path.basename(info.filename)
                if base.endswith(".onnx"):
                    with z.open(info) as src, \
                            open(os.path.join(pack_dir, base),
                                 "wb") as dst:
                        shutil.copyfileobj(src, dst)


def repo_files(repo):
    with urllib.request.urlopen(
        f"https://huggingface.co/api/models/{repo}", timeout=30
    ) as r:
        d = json.load(r)
    return [s["rfilename"] for s in d.get("siblings", [])]


def ensure_models(repo, files):
    """Download (once) everything the GGUF graph needs; return unet name."""
    hf_file(SEED_REPO, ENCODER, os.path.join(MODELS, "text_encoders"))
    hf_file(SEED_REPO, VAE, os.path.join(MODELS, "vae"))
    ggufs = [f for f in files if f.endswith(".gguf") and "/" not in f]
    pick = next((f for f in ggufs if "Q4_K_M" in f), ggufs[0])
    hf_file(repo, pick, os.path.join(MODELS, "diffusion_models"))
    return pick


def sdxl_checkpoint(files):
    """A standalone SDXL checkpoint is a .safetensors that is not one of
    the diffusers-layout parts (unet/, text_encoder*/, vae/)."""
    skip = ("text_encoders/", "text_encoder/", "text_encoder_2/",
            "vae/", "unet/", "loras/")
    cands = [f for f in files
             if f.endswith(".safetensors") and not f.startswith(skip)]
    return cands[0] if cands else None


def ensure_sdxl(repo, files):
    """Download (once) the SDXL checkpoint; return its file name."""
    pick = sdxl_checkpoint(files)
    if not pick:
        raise RuntimeError(
            f"repo {repo} has no top-level .gguf and no standalone "
            ".safetensors checkpoint")
    dest_dir = os.path.join(MODELS, "checkpoints")
    dest = os.path.join(dest_dir, os.path.basename(pick))
    if not (os.path.exists(dest) and os.path.getsize(dest) > 1_000_000):
        if shutil.disk_usage(VOLUME).free < 7_500_000_000:
            # The 20 GB volume cannot hold the Qwen cache and an SDXL
            # checkpoint at once. The GGUF is a re-downloadable cache,
            # so evict it to make room (it returns on the next GGUF job).
            gguf_dir = os.path.join(MODELS, "diffusion_models")
            if os.path.isdir(gguf_dir):
                for name in os.listdir(gguf_dir):
                    if name.endswith(".gguf"):
                        print(f"evicting cached GGUF {name} for volume room")
                        os.remove(os.path.join(gguf_dir, name))
    hf_file(repo, pick, dest_dir)
    return os.path.basename(pick)


def ensure_wan(flavor):
    """Cache the Wan 2.2 stack for flavor ('t2v' or 'i2v').

    Each flavor's two FP8 experts are ~14.3 GB apiece; the 50 GB
    volume holds ONE flavor beside the image caches, so a flavor
    switch evicts the other flavor's experts (plus Qwen GGUFs if
    still tight) and re-downloads. Shared parts (umt5, VAE, and both
    flavors' small Lightning LoRAs) stay cached."""
    files = repo_files(WAN_REPO)

    def find(base):
        hits = [f for f in files if os.path.basename(f) == base]
        if not hits:
            raise RuntimeError(f"Wan repo is missing {base}")
        return hits[0]

    lora_v = "v1.1" if flavor == "t2v" else "v1"
    want = [
        (f"wan2.2_{flavor}_high_noise_14B_fp8_scaled.safetensors",
         "diffusion_models"),
        (f"wan2.2_{flavor}_low_noise_14B_fp8_scaled.safetensors",
         "diffusion_models"),
        (f"wan2.2_{flavor}_lightx2v_4steps_lora_{lora_v}_high_noise"
         ".safetensors", "loras"),
        (f"wan2.2_{flavor}_lightx2v_4steps_lora_{lora_v}_low_noise"
         ".safetensors", "loras"),
        ("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
        ("wan_2.1_vae.safetensors", "vae"),
    ]
    experts = [os.path.join(MODELS, sub, base) for base, sub in want[:2]]
    cached = all(os.path.exists(p) and os.path.getsize(p) > 1_000_000
                 for p in experts)
    if not cached and shutil.disk_usage(VOLUME).free < 32_000_000_000:
        other = "i2v" if flavor == "t2v" else "t2v"
        dm = os.path.join(MODELS, "diffusion_models")
        if os.path.isdir(dm):
            for name in os.listdir(dm):
                if name.startswith(f"wan2.2_{other}_") or \
                        name.endswith(".gguf"):
                    print(f"evicting {name} for Wan volume room")
                    try:
                        os.remove(os.path.join(dm, name))
                    except OSError:
                        pass
    for base, sub in want:
        hf_file(WAN_REPO, find(base), os.path.join(MODELS, sub))


def run_video(job_input):
    prompt = job_input.get("prompt", "").strip()
    if not prompt:
        raise RuntimeError("prompt is required")
    image_b64 = job_input.get("image")
    flavor = "i2v" if image_b64 else "t2v"
    width = int(job_input.get("width", 832))
    height = int(job_input.get("height", 480))
    frames = int(job_input.get("frames", 81))
    seed = int(job_input.get("seed", 0)) or random.randint(1, 2**31 - 1)

    ensure_wan(flavor)

    # Optional community LoRA (the explicit-content tuning layer): an
    # HF repo id or "repo:filename". Wan NSFW LoRAs ship as expert
    # PAIRS (a high-noise file and a low-noise file - e.g. lopi999's
    # NSFW-22-H-e8 / NSFW-22-L-e8); when the repo names a pair, each
    # file chains into its own expert. A lone file chains into both.
    lora_high = lora_low = None
    lora_ref = (job_input.get("lora") or "").strip()
    if lora_ref:
        if ":" in lora_ref and "/" in lora_ref.split(":")[0]:
            lrepo, lfile = lora_ref.split(":", 1)
        else:
            lrepo, lfile = lora_ref, None
        lfiles = repo_files(lrepo)
        tensors = [f for f in lfiles if f.endswith(".safetensors")]
        if lfile:
            if lfile not in tensors:
                raise RuntimeError(f"lora file {lfile} not in {lrepo}")
            hf_file(lrepo, lfile, os.path.join(MODELS, "loras"))
            lora_high = lora_low = os.path.basename(lfile)
        else:
            def _is_high(n):
                n = n.lower()
                return "high" in n or "-h-" in n or "_h_" in n \
                    or n.endswith("-h.safetensors")
            def _is_low(n):
                n = n.lower()
                return "low" in n or "-l-" in n or "_l_" in n \
                    or n.endswith("-l.safetensors")
            hi = next((f for f in tensors if _is_high(f)), None)
            lo = next((f for f in tensors if _is_low(f)), None)
            if hi and lo and hi != lo:
                hf_file(lrepo, hi, os.path.join(MODELS, "loras"))
                hf_file(lrepo, lo, os.path.join(MODELS, "loras"))
                lora_high = os.path.basename(hi)
                lora_low = os.path.basename(lo)
            elif tensors:
                hf_file(lrepo, tensors[0],
                        os.path.join(MODELS, "loras"))
                lora_high = lora_low = os.path.basename(tensors[0])
            else:
                raise RuntimeError(f"lora repo {lrepo} has no .safetensors")
    link_into_comfy()

    here = os.path.dirname(__file__)
    fname = ("workflow_wan_i2v_api.json" if flavor == "i2v"
             else "workflow_wan_t2v_api.json")
    with open(os.path.join(here, fname)) as f:
        wf = json.load(f)
    wf["8"]["inputs"]["text"] = prompt
    wf["9"]["inputs"]["text"] = job_input.get("negative") or WAN_NEGATIVE
    wf["12"]["inputs"]["noise_seed"] = seed
    if flavor == "i2v":
        up = requests.post(
            COMFY + "/upload/image",
            files={"image": ("minix_i2v_input.png",
                             base64.b64decode(image_b64), "image/png")},
            data={"overwrite": "true"}, timeout=60)
        up.raise_for_status()
        wf["17"]["inputs"]["width"] = width
        wf["17"]["inputs"]["height"] = height
        wf["17"]["inputs"]["length"] = frames
    else:
        wf["11"]["inputs"]["width"] = width
        wf["11"]["inputs"]["height"] = height
        wf["11"]["inputs"]["length"] = frames
    if lora_high:
        strength = float(job_input.get("lora_strength", 0.85))
        for new_id, prev, nxt, lname in (
                ("18", "2", "3", lora_high), ("19", "5", "6", lora_low)):
            wf[new_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": [prev, 0], "lora_name": lname,
                           "strength_model": strength},
            }
            wf[nxt]["inputs"]["model"] = [new_id, 0]

    r = requests.post(COMFY + "/prompt", json={"prompt": wf}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(
            f"comfy /prompt {r.status_code}: {r.text[:600]}")
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]

    deadline = time.time() + 1200
    while time.time() < deadline:
        h = requests.get(COMFY + f"/history/{prompt_id}",
                         timeout=15).json()
        if prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"comfy execution error: {status}")
            for node_out in entry.get("outputs", {}).values():
                for key in ("images", "gifs", "videos"):
                    for vid in node_out.get(key, []):
                        fn = vid.get("filename", "")
                        if fn.endswith((".mp4", ".webm")):
                            q = urllib.parse.urlencode(
                                {"filename": fn,
                                 "subfolder": vid.get("subfolder", ""),
                                 "type": vid.get("type", "output")})
                            data = requests.get(
                                COMFY + "/view?" + q, timeout=120).content
                            return {"videos": [
                                base64.b64encode(data).decode()],
                                "seed": seed}
            raise RuntimeError("workflow finished with no video output")
        time.sleep(3)
    raise RuntimeError("video generation timed out after 1200s")


def ensure_lora(spec):
    """Download (once) one LoRA file into the volume loras cache.

    spec: {"repo": <hf repo>, "file": <path in repo>,
           "strength_model": float, "strength_clip": float}
    Returns the cached file's basename for the LoraLoader node.
    hf_file returns the full destination path; LoraLoader validates
    lora_name against the bare file names in its loras folder, so the
    basename is mandatory (the full path 400s every LoRA job).
    """
    dest = hf_file(spec["repo"], spec["file"],
                   os.path.join(MODELS, "loras"))
    return os.path.basename(dest)


def build_sdxl_lora_workflow(ckpt_name, lora_names, prompt, negative,
                             width, height, steps, seed, specs):
    """The SDXL graph with a LoraLoader chain after the checkpoint.

    Same sampler recipe as workflow_sdxl_api.json (DPM 2 ancestral,
    Karras, CFG 4.5) and the same node ids for the shared slots; the
    only change is model+clip threading through one LoraLoader per
    requested LoRA, in order.
    """
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"batch_size": 1, "height": height, "width": width}},
        "5": {"class_type": "KSampler",
              "inputs": {"cfg": 4.5, "denoise": 1.0,
                         "latent_image": ["4", 0],
                         "negative": ["3", 0], "positive": ["2", 0],
                         "sampler_name": "dpm_2_ancestral",
                         "scheduler": "karras", "seed": seed,
                         "steps": steps}},
        "6": {"class_type": "VAEDecode",
              "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "minix_sdxl",
                         "images": ["6", 0]}},
    }
    prev = "1"
    for i, (name, spec) in enumerate(zip(lora_names, specs)):
        node = str(10 + i)
        wf[node] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": [prev, 0],
                "clip": [prev, 1],
                "lora_name": name,
                "strength_model": float(spec.get("strength_model", 0.8)),
                "strength_clip": float(spec.get("strength_clip", 0.8)),
            },
        }
        prev = node
    wf["2"]["inputs"]["clip"] = [prev, 1]
    wf["3"]["inputs"]["clip"] = [prev, 1]
    wf["5"]["inputs"]["model"] = [prev, 0]
    return wf


COMFY_MODELS = "/comfyui/models"


def link_into_comfy():
    """ComfyUI only sees its own models tree. start.sh links the volume
    cache in at container boot, but files downloaded afterwards (the
    first job on a fresh volume) never got linked - the exact failure of
    2026-11-03 21:50, when ComfyUI rejected the prompt with every model
    list empty while the files sat complete on the volume. Link at job
    time, right before prompting."""
    for sub in ("diffusion_models", "text_encoders", "vae", "loras",
                "checkpoints", "ipadapter", "clip_vision", "insightface"):
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

    files = repo_files(repo)
    here = os.path.dirname(__file__)
    if any(f.endswith(".gguf") and "/" not in f for f in files):
        unet_name = ensure_models(repo, files)
        link_into_comfy()
        with open(os.path.join(here, "workflow_api.json")) as f:
            wf = json.load(f)
        wf["1"]["inputs"]["unet_name"] = unet_name
        wf["4"]["inputs"]["text"] = prompt
        wf["7"]["inputs"]["width"] = width
        wf["7"]["inputs"]["height"] = height
        wf["9"]["inputs"]["steps"] = steps
        wf["9"]["inputs"]["seed"] = seed
    else:
        ckpt_name = ensure_sdxl(repo, files)
        ref_b64 = job_input.get("image")
        if ref_b64:
            # Face-lock path: a base64 reference portrait anchors the
            # identity (IP-Adapter FaceID Plus V2) over the SDXL graph.
            ensure_faceid()
            link_into_comfy()
            up = requests.post(
                COMFY + "/upload/image",
                files={"image": ("minix_face_ref.png",
                                 base64.b64decode(ref_b64), "image/png")},
                data={"overwrite": "true"}, timeout=60)
            up.raise_for_status()
            with open(os.path.join(here,
                                   "workflow_sdxl_faceid_api.json")) as f:
                wf = json.load(f)
            wf["11"]["inputs"]["lora_strength"] = float(
                job_input.get("face_lora_strength", 0.6))
            wf["12"]["inputs"]["weight"] = float(
                job_input.get("face_weight", 0.85))
            wf["12"]["inputs"]["weight_faceidv2"] = float(
                job_input.get("face_weight_v2", 1.2))
            negative = SDXL_NEGATIVE
        else:
            negative = SDXL_NEGATIVE
            extra_neg = str(job_input.get("negative_extra") or "").strip()
            if extra_neg:
                negative = negative + ", " + extra_neg
            lora_specs = job_input.get("loras") or []
            if lora_specs:
                lora_names = [ensure_lora(spec) for spec in lora_specs]
                link_into_comfy()
                wf = build_sdxl_lora_workflow(
                    ckpt_name, lora_names, prompt, negative,
                    width, height, steps, seed, lora_specs)
            else:
                link_into_comfy()
                with open(os.path.join(here, "workflow_sdxl_api.json")) as f:
                    wf = json.load(f)
        wf["1"]["inputs"]["ckpt_name"] = ckpt_name
        wf["2"]["inputs"]["text"] = prompt
        wf["3"]["inputs"]["text"] = negative
        wf["4"]["inputs"]["width"] = width
        wf["4"]["inputs"]["height"] = height
        wf["5"]["inputs"]["steps"] = steps
        wf["5"]["inputs"]["seed"] = seed

    r = requests.post(COMFY + "/prompt", json={"prompt": wf}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(
            f"comfy /prompt {r.status_code}: {r.text[:600]}")
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]

    deadline = time.time() + 1200
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
    raise RuntimeError("generation timed out after 1200s")


def handler(job):
    try:
        job_input = job.get("input", {})
        if job_input.get("kind") == "video":
            return run_video(job_input)
        return run_workflow(job_input)
    except Exception as e:  # honest failure back to the app
        return {"error": str(e)}


if __name__ == "__main__":
    wait_for_comfy()
    runpod.serverless.start({"handler": handler})
