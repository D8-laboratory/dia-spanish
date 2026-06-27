"""
Modal inference script for Dia Spanish — generate audio from fine-tuned checkpoints.

Usage:
  # Generate with default prompts using latest checkpoint
  modal run infer_dia_es.py

  # Use a specific checkpoint
  modal run infer_dia_es.py --ckpt /vol/data/checkpoints/dia_es/ckpt_step1001.pth

  # Compare base model vs fine-tuned
  modal run infer_dia_es.py --compare

  # Custom generation params
  modal run infer_dia_es.py --temperature 1.5 --cfg-scale 3.0

Output WAVs land on the volume at /vol/output/es_samples/ and are also downloaded locally.
"""

import modal
import os

MINUTES = 60

app = modal.App("dia-es-infer")

VOL_NAME = "dia-spanish-vol"
VOL_MOUNT = "/vol"
CKPT_DIR = f"{VOL_MOUNT}/data/checkpoints/dia_es"
OUTPUT_DIR = f"{VOL_MOUNT}/output/es_samples"
CONFIG_PATH = "/root/dia/config.json"

# Reuse the training image definition (must match training for DAC compat)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "descript-audio-codec>=1.0.0",
        "bitsandbytes>=0.43.0",
        "datasets>=2.13.0,<3.0",
        "soundfile>=0.13.1",
        "tensorboard>=2.12.0",
        "transformers>=4.35.0",
        "huggingface-hub>=0.30.2",
        "safetensors>=0.4.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "pydantic>=2.11.3",
    )
    .pip_install(
        "protobuf>=5.26.1",
        "numpy>=1.24.0,<2.0",
    )
    .env({"PYTHONPATH": "/root:$PYTHONPATH"})
    .add_local_dir("dia", "/root/dia")
)

volume = modal.Volume.from_name(VOL_NAME)


# ── Spanish test prompts ─────────────────────────────────────────────
# Mix of domains — casual, tech, narrative — to probe what the model learned.
DEFAULT_PROMPTS = [
    {
        "id": "es_casual_01",
        "text": "[S1] ¡Hola! ¿Cómo te fue hoy? [S2] Bastante bien, la verdad. Terminé el trabajo temprano y pude descansar. [S1] Qué bueno. ¿Querés salir a caminar más tarde? [S2] Dale, perfecto. Pasa por mi casa a las seis.",
    },
    {
        "id": "es_tech_01",
        "text": "[S1] ¿Viste la nueva versión del modelo de lenguaje? [S2] Sí, es impresionante. Genera texto en español casi perfecto. [S1] ¿Y funciona para diálogos? [S2] Sí, de hecho puede mantener una conversación bastante natural.",
    },
    {
        "id": "es_narrativa_01",
        "text": "[S1] La historia de España está llena de momentos cruciales que cambiaron el rumbo del mundo. [S2] Así es. Desde la reconquista hasta la conquista de América, cada época dejó su huella. [S1] Y la cultura hispánica se expandió por todo el continente.",
    },
    {
        "id": "es_podcast_01",
        "text": "[S1] Bienvenidos a un nuevo episodio. Hoy vamos a hablar sobre inteligencia artificial. [S2] Tema fascinante. ¿Por dónde empezamos? [S1] Pues por lo básico. ¿Qué es realmente la inteligencia artificial?",
    },
]


@app.function(
    image=image,
    volumes={VOL_MOUNT: volume},
    gpu="l40s",
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
def generate(
    ckpt_path: str = "",
    prompts_json: str = "",
    temperature: float = 1.3,
    cfg_scale: float = 3.0,
    top_p: float = 0.95,
    max_tokens: int = 3072,
    use_finetuned: bool = True,
) -> dict:
    """Generate Spanish audio samples from a Dia checkpoint."""
    import json
    import torch
    import numpy as np
    import soundfile as sf
    from pathlib import Path

    volume.reload()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Resolve checkpoint ──
    if use_finetuned:
        if not ckpt_path:
            # Default: latest epoch checkpoint
            ckpts = sorted(Path(CKPT_DIR).glob("ckpt_epoch*.pth"))
            if not ckpts:
                ckpts = sorted(Path(CKPT_DIR).glob("ckpt_step*.pth"))
            if not ckpts:
                return {"error": f"No checkpoints found in {CKPT_DIR}"}
            ckpt_path = str(ckpts[-1])
        print(f"Using fine-tuned checkpoint: {ckpt_path}")
    else:
        ckpt_path = ""
        print("Using base Dia-1.6B model (nari-labs/Dia-1.6B)")

    # ── Parse prompts ──
    if prompts_json:
        prompts = json.loads(prompts_json)
    else:
        prompts = DEFAULT_PROMPTS

    print(f"Generating {len(prompts)} samples...")
    print(f"  temperature={temperature}, cfg_scale={cfg_scale}, top_p={top_p}")

    # ── Load model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from dia.model import Dia

    if use_finetuned:
        dia = Dia.from_local(CONFIG_PATH, ckpt_path, device=device)
    else:
        # Load base model from HF, but using our dia package
        import json as _json
        from dia.config import DiaConfig
        from dia.layers import DiaModel
        from huggingface_hub import hf_hub_download

        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config_dict = _json.load(f)
        dia_cfg = DiaConfig(**config_dict)
        dia = Dia(dia_cfg, device)
        base_ckpt = hf_hub_download("nari-labs/Dia-1.6B", filename="dia-v0_1.pth")
        ckpt = torch.load(base_ckpt, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        dia.model.load_state_dict(state_dict, strict=False)

    results = []
    for i, prompt in enumerate(prompts, 1):
        pid = prompt["id"]
        text = prompt["text"]
        print(f"\n[{i}/{len(prompts)}] {pid}")
        print(f"  text: {text[:80]}...")

        try:
            audio = dia.generate(
                text=text,
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
            )
            # audio is np.ndarray float32
            out_path = f"{OUTPUT_DIR}/{pid}.wav"
            sf.write(out_path, audio, 44100, subtype="PCM_16")
            duration = len(audio) / 44100
            print(f"  ✓ saved {out_path} ({duration:.1f}s)")
            results.append({
                "id": pid,
                "text": text,
                "audio_path": out_path,
                "duration_sec": round(duration, 2),
                "sample_rate": 44100,
            })
        except Exception as e:
            print(f"  ✗ failed: {e}")
            results.append({"id": pid, "error": str(e), "text": text})

    volume.commit()
    return {"checkpoint": ckpt_path or "base", "samples": results}


@app.local_entrypoint()
def main(
    ckpt: str = "",
    temperature: float = 1.3,
    cfg_scale: float = 3.0,
    top_p: float = 0.95,
    max_tokens: int = 3072,
    base: bool = False,
    text: str = "",
    sample_id: str = "custom",
):
    """Generate Spanish samples. Use --base for the un-fine-tuned model.

    Pass --text "..." to generate a single custom prompt.
    Otherwise generates the 4 built-in default prompts.
    """
    import json

    if text:
        prompts = [{"id": sample_id, "text": text}]
    else:
        prompts = []  # use defaults on the remote side

    result = generate.remote(
        ckpt_path=ckpt,
        prompts_json=json.dumps(prompts) if prompts else "",
        temperature=temperature,
        cfg_scale=cfg_scale,
        top_p=top_p,
        max_tokens=max_tokens,
        use_finetuned=not base,
    )

    print(f"\n{'='*60}")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"{'='*60}")
    for s in result["samples"]:
        if "error" in s:
            print(f"  ✗ {s['id']}: {s['error']}")
        else:
            print(f"  ✓ {s['id']}: {s['duration_sec']}s → {s['audio_path']}")
    print(f"\n{'='*60}")
    print(f"Download locally with:")
    print(f"  modal volume get dia-spanish-vol output/es_samples ./results/es_samples")
    print(f"{'='*60}")
