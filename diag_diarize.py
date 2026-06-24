"""Diagnostic: surface the real diarization failure (no try/except hiding it)."""
import modal

VOL_MOUNT = "/vol"
app = modal.App("dia-diag")
volume = modal.Volume.from_name("dia-spanish-vol")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "build-essential")
    .pip_install("whisperx>=3.1.1", "torch>=2.1", "torchaudio>=2.1",
                 "faster-whisper>=1.0", "pyannote.audio>=3.1")
)


@app.function(
    image=image,
    volumes={VOL_MOUNT: volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
    gpu="A10G",
    timeout=10 * 60,
)
def diag():
    import os
    import inspect
    import whisperx

    print("=== DIAGNOSTIC ===")
    print(f"whisperx version: {getattr(whisperx, '__version__', '?')}")
    print(f"HF_TOKEN in env: {'HF_TOKEN' in os.environ}")
    print(f"HUGGING_FACE_HUB_TOKEN in env: {'HUGGING_FACE_HUB_TOKEN' in os.environ}")
    print(f"All HF-ish env keys: {[k for k in os.environ if 'HF' in k or 'HUGGING' in k or 'TOKEN' in k]}")

    from whisperx.diarize import DiarizationPipeline
    sig = inspect.signature(DiarizationPipeline.__init__)
    print(f"DiarizationPipeline.__init__ params: {list(sig.parameters)}")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Token resolved: {bool(hf_token)} (len={len(hf_token) if hf_token else 0})")

    # Instantiate WITHOUT try/except — let it crash loudly
    print("\n--- Attempting DiarizationPipeline instantiation ---")
    kw = {"device": "cuda"}
    if "token" in sig.parameters:
        kw["token"] = hf_token
        print("Using kwarg 'token'")
    elif "use_auth_token" in sig.parameters:
        kw["use_auth_token"] = hf_token
        print("Using kwarg 'use_auth_token'")

    diarize_model = DiarizationPipeline(**kw)
    print("✅ DiarizationPipeline instantiated successfully")

    # Run on a real audio file from the volume
    import glob
    wavs = sorted(glob.glob(f"{VOL_MOUNT}/data/podcasts_rss/audio/*.wav"))[:1]
    if wavs:
        print(f"\n--- Running diarization on: {wavs[0]} ---")
        segs = diarize_model(wavs[0], min_speakers=2, max_speakers=4)
        print(f"✅ Diarization produced {len(segs)} segments")
        speakers = sorted(set(s.speaker for s in segs)) if hasattr(segs[0], "speaker") else "N/A"
        print(f"Speaker labels found: {speakers}")
        print("First 3 segments:")
        for s in segs[:3]:
            print(f"  {s}")


@app.local_entrypoint()
def main():
    diag.remote()
