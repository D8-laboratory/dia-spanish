"""Generate Spanish dialogue audio samples with the base Dia model on Modal.

Usage:
  modal run scripts/generate_es_samples.py
  modal run scripts/generate_es_samples.py --num-samples 3
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "dia-spanish-generate"
OUTPUT_DIR = Path("/vol/output/spanish_samples")

SPANISH_SAMPLES = [
    {
        "id": "es_casual_01",
        "domain": "casual",
        "transcript": "[S1] ¡Hola! ¿Cómo te fue en la reunión? [S2] ¡Uf, súper bien! Estaba nerviosa pero al final todo salió perfecto. (risas) [S1] ¡Qué bueno! Me alegro mucho. ¿Y qué te dijeron del proyecto? [S2] Pues, les encantó la idea. Quieren que lo presentemos la próxima semana ante la junta directiva. [S1] Wow, eso es genial. ¿Necesitas ayuda preparando la presentación? [S2] La verdad sí, me haría falta una mano. ¿Te viene bien mañana por la tarde?",
    },
    {
        "id": "es_familia_01",
        "domain": "familia",
        "transcript": "[S1] Mami, ¿puedo ir al cine con mis amigos este sábado? [S2] ¿Qué películas están dando? [S1] Hay una de superhéroes que está súper buena. Todos mis amigos ya la vieron. (risas) [S2] Está bien, pero con condición de que llegues antes de las nueve. [S1] ¡Gracias, mami! Prometo portarme bien. (risas)",
    },
    {
        "id": "es_tecnologia_01",
        "domain": "tecnología",
        "transcript": "[S1] ¿Ya probaste la nueva actualización de la aplicación? [S2] ¡Sí! Está increíble. Ahora tiene modo oscuro y todo. [S1] ¿En serio? Tengo que actualizarla ya mismo. [S2] Y también agregaron un asistente de voz que entiende español perfecto. (risas) Por fin, ¿no? [S1] Jaja, era hora. Las versiones anteriores eran un dolor de cabeza.",
    },
]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("dia-spanish-vol", create_if_missing=True)
hf_cache = modal.Volume.from_name("dia-spanish-hf-cache", create_if_missing=True)

# Pin protobuf last — Modal's runner breaks if pip pulls protobuf 5.x.
dia_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.52.0",
        "descript-audio-codec>=1.0.0",
        "huggingface-hub>=0.30.2",
        "numpy>=2.2.4",
        "safetensors>=0.5.3",
        "soundfile>=0.13.1",
        "accelerate",
    )
    .pip_install("protobuf>=4.25.0,<5.0")
)


@app.function(
    image=dia_image,
    gpu="T4",
    timeout=45 * 60,
    volumes={"/vol": volume, "/root/.cache/huggingface": hf_cache},
)
def generate_samples(samples: list[dict], model_name: str = "nari-labs/Dia-1.6B-0626") -> list[dict]:
    import torch
    from transformers import AutoProcessor, DiaForConditionalGeneration

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    print(f"Loading model {model_name} on {device}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = DiaForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] Generating {sample['id']} ({sample['domain']})...")

        inputs = processor(text=[sample["transcript"]], padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3072,
                guidance_scale=3.0,
                temperature=1.8,
                top_p=0.90,
                top_k=45,
            )

        decoded = processor.batch_decode(outputs)
        audio_path = OUTPUT_DIR / f"{sample['id']}.mp3"
        processor.save_audio(decoded, str(audio_path))

        manifest = {
            "id": sample["id"],
            "domain": sample["domain"],
            "transcript": sample["transcript"],
            "audio_path": str(audio_path),
            "model": model_name,
            "note": "Base English Dia model — zero-shot Spanish test",
        }
        manifest_path = OUTPUT_DIR / f"{sample['id']}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        results.append(manifest)
        print(f"  Saved → {audio_path}")

    volume.commit()
    hf_cache.commit()
    return results


@app.local_entrypoint()
def main(num_samples: int = 3, model: str = "nari-labs/Dia-1.6B-0626"):
    samples = SPANISH_SAMPLES[:num_samples]
    print(f"Generating {len(samples)} Spanish dialogue samples with {model}...\n")

    results = generate_samples.remote(samples, model_name=model)

    print("\n--- Results ---")
    for r in results:
        print(f"  [{r['id']}] ({r['domain']})")
        print(f"    Audio: {r['audio_path']}")
        print(f"    Transcript: {r['transcript'][:80]}...")
        print()

    print("Download with:")
    print("  modal volume get dia-spanish-vol output/spanish_samples ./output/spanish_samples")
