"""
Modal training script for Dia Spanish fine-tuning.

Usage:
  # Smoke test (3,000 samples, 1 epoch, ~1 hour on L40S)
  modal run train_dia_es.py --max-samples 3000 --epochs 1

  # Full run (43,219 samples, 2 epochs, ~24h on A100-80GB)
  modal run train_dia_es.py --max-samples 0 --epochs 2 --gpu a100-80gb

  # Resume from checkpoint
  modal run train_dia_es.py --resume /vol/data/checkpoints/dia_es/ckpt_epoch1.pth
"""

import modal
import sys
import os

MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App("dia-es-training")

VOL_NAME = "dia-spanish-vol"
VOL_MOUNT = "/vol"
DATASET_PATH = f"{VOL_MOUNT}/data/hf_dataset"
CKPT_DIR = f"{VOL_MOUNT}/data/checkpoints/dia_es"
RUNS_DIR = f"{VOL_MOUNT}/data/runs"
CONFIG_PATH = "/root/dia/config.json"

# ── Image ────────────────────────────────────────────────────────────
# CUDA 12.6 + Python 3.11 + torch 2.6 + DAC + bitsandbytes
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "libsndfile1", "git")
    # Step 1: PyTorch with CUDA (separate to avoid resolution-too-deep)
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        index_url="https://download.pytorch.org/whl/cu126",
    )
    # Step 2: ML ecosystem (torch already satisfied → bitsandbytes won't re-resolve)
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
        "pydantic>=2.11.3",   # needed by dia/config.py (BeforeValidator)
        "wandb>=0.18.0",      # experiment tracking
    )
    # Step 3: Force compatible versions LAST (override any transitive upgrades)
    .pip_install(
        "protobuf>=4.21,<5.0",   # <5 to avoid breaking Modal runtime
        "numpy>=1.24.0,<2.0",    # <2 for numba/DAC compat
    )
    .env({"PYTHONPATH": "/root:$PYTHONPATH"})
    # Deploy our dia/ package into the image (MUST be last step in build)
    .add_local_dir("dia", "/root/dia")
)

volume = modal.Volume.from_name(VOL_NAME)


@app.function(
    image=image,
    volumes={VOL_MOUNT: volume},
    gpu="l40s",
    timeout=2 * HOURS,
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
        modal.Secret.from_name("my-wandb-secret"),
    ],
)
def train(
    max_samples: int = 3000,
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 2,
    learning_rate: float = 1e-5,
    eval_step: int = 100,
    save_step: int = 500,
    half: bool = True,
    compile: bool = False,
    resume: str = "",
    run_name: str = "",
    wandb_project: str = "dia-spanish",
    wandb_enabled: bool = True,
):
    """Run Dia Spanish fine-tuning on Modal."""
    volume.reload()

    # Ensure checkpoint + runs dirs exist on the volume
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    rn = run_name or ("dia_es_smoke" if max_samples > 0 else "dia_es_full")

    # Set wandb env vars for the training process
    if wandb_enabled:
        os.environ["WANDB_PROJECT"] = wandb_project
        # WANDB_API_KEY comes from the modal secret
        os.environ.setdefault("WANDB_MODE", "online")
    else:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ.pop("WANDB_API_KEY", None)

    # Build sys.argv for finetune.main()
    argv = [
        "finetune.py",
        "--config", CONFIG_PATH,
        "--dataset", DATASET_PATH,
        "--output_dir", CKPT_DIR,
        "--runs_dir", RUNS_DIR,
        "--run_name", rn,
        "--max_samples", str(max_samples),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--grad_accum", str(grad_accum),
        "--learning_rate", str(learning_rate),
        "--eval_step", str(eval_step),
        "--save_step", str(save_step),
    ]
    if half:
        argv.append("--half")
    if compile:
        argv.append("--compile")
    if resume:
        argv.extend(["--resume_from", resume])

    sys.argv = argv

    print("=" * 60)
    print(f"Dia Spanish Fine-tuning")
    print(f"  max_samples: {max_samples or 'ALL'}")
    print(f"  epochs: {epochs}")
    print(f"  batch_size: {batch_size} × grad_accum {grad_accum}")
    print(f"  lr: {learning_rate}")
    print(f"  eval_every: {eval_step} | save_every: {save_step}")
    print(f"  half: {half} | compile: {compile}")
    print(f"  dataset: {DATASET_PATH}")
    print(f"  checkpoints: {CKPT_DIR}")
    print(f"  run_name: {rn}")
    print("=" * 60)

    from dia import finetune
    finetune.main()

    # List checkpoints
    volume.reload()
    ckpts = sorted(os.listdir(CKPT_DIR)) if os.path.exists(CKPT_DIR) else []
    print(f"\n{'='*60}")
    print(f"Training complete. Checkpoints in {CKPT_DIR}:")
    for c in ckpts:
        sz = os.path.getsize(os.path.join(CKPT_DIR, c)) / 1e9
        print(f"  {c} ({sz:.1f} GB)")
    print(f"{'='*60}")

    return {
        "run_name": rn,
        "checkpoints": ckpts,
        "ckpt_dir": CKPT_DIR,
    }


@app.local_entrypoint()
def main(
    max_samples: int = 3000,
    epochs: int = 1,
    batch_size: int = 2,
    grad_accum: int = 2,
    learning_rate: float = 1e-5,
    eval_step: int = 100,
    save_step: int = 500,
    half: bool = True,
    compile: bool = False,
    resume: str = "",
    run_name: str = "",
    wandb_project: str = "dia-spanish",
    wandb_enabled: bool = True,
):
    """Local entrypoint — delegates to train.remote()."""
    result = train.remote(
        max_samples=max_samples,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        eval_step=eval_step,
        save_step=save_step,
        half=half,
        compile=compile,
        resume=resume,
        run_name=run_name,
        wandb_project=wandb_project,
        wandb_enabled=wandb_enabled,
    )
    print(f"\n✅ Done: {result}")
