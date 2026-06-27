"""
Quick Modal script to:
1. Transcribe voicespeaker1.ogg with Whisper
2. Run voice cloning inference on the fine-tuned Spanish Dia model

Usage:
  # Step 1: transcribe only
  modal run transcribe_and_clone.py::transcribe

  # Step 2: run voice cloning (uses transcript from step 1)
  modal run transcribe_and_clone.py --new-lines "[S1] Hola, esto es una prueba de clonación de voz."
"""

import modal
import os
import base64
from pathlib import Path

MINUTES = 60
VOL_NAME = "dia-spanish-vol"
VOL_MOUNT = "/vol"
CKPT_PATH = f"{VOL_MOUNT}/data/checkpoints/dia_es/ckpt_epoch1.pth"
OUTPUT_DIR = f"{VOL_MOUNT}/output/voice_cloning"
CONFIG_PATH = "/root/dia/config.json"

app = modal.App("dia-es-voice-clone")

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
        # Install DAC first — descript-audiotools declares protobuf<3.20 but doesn't use it
        "descript-audio-codec>=1.0.0",
        "soundfile>=0.13.1",
        "numpy>=1.24.0,<2.0",
        "safetensors>=0.4.0",
        "tqdm>=4.65.0",
        "huggingface-hub>=0.30.2",
        "pydantic>=2.11.3",
    )
    .pip_install(
        # Upgrade protobuf to >=5 (overrides descript-audiotools stale constraint)
        # and install whisper separately to avoid resolver conflict
        "openai-whisper==20250625",
        "protobuf>=5.26.1",
    )
    .env({"PYTHONPATH": "/root:$PYTHONPATH"})
    .add_local_dir("dia", "/root/dia")
)

volume = modal.Volume.from_name(VOL_NAME)


@app.function(image=image, volumes={VOL_MOUNT: volume}, timeout=10 * MINUTES)
def transcribe(audio_b64: str, filename: str = "voice.ogg") -> str:
    import whisper
    import tempfile, base64

    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    model = whisper.load_model("medium")
    result = model.transcribe(tmp_path, language="es", task="transcribe")
    transcript = result["text"].strip()
    print(f"Transcript: {transcript}")
    return transcript


@app.function(
    image=image,
    volumes={VOL_MOUNT: volume},
    gpu="a10g",
    timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
def voice_clone(audio_b64: str, transcript: str, new_lines: str, filename: str = "voice.ogg") -> dict:
    import subprocess
    import tempfile
    import base64
    import torch
    import soundfile as sf
    from pathlib import Path as P

    volume.reload()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save audio prompt and convert to WAV for DAC compatibility
    audio_bytes = base64.b64decode(audio_b64)
    suffix = P(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_src = f.name

    tmp_wav = tmp_src.replace(suffix, ".wav")
    subprocess.run(["ffmpeg", "-i", tmp_src, "-ar", "44100", "-ac", "1", tmp_wav, "-y"], check=True)

    # Build full text: transcript prefix + new content
    # Dia voice cloning: prepend transcript of the audio sample, then new lines
    transcript_tagged = f"[S1] {transcript.strip()}"
    new_lines_clean = new_lines.strip()
    full_text = f"{transcript_tagged}\n{new_lines_clean}"
    print(f"\nFull prompt:\n{full_text}\n")

    # Load fine-tuned model
    device = torch.device("cuda")
    from dia.model import Dia
    dia = Dia.from_local(CONFIG_PATH, CKPT_PATH, device=device)

    with torch.inference_mode():
        audio = dia.generate(
            text=full_text,
            max_tokens=3072,
            cfg_scale=3.0,
            temperature=1.3,
            top_p=0.95,
            audio_prompt_path=tmp_wav,
        )

    out_path = f"{OUTPUT_DIR}/voice_clone_result.wav"
    sf.write(out_path, audio, 44100, subtype="PCM_16")
    duration = len(audio) / 44100
    print(f"\nSaved: {out_path} ({duration:.1f}s)")

    volume.commit()
    return {"audio_path": out_path, "duration_sec": round(duration, 2), "transcript": transcript, "full_text": full_text}


CHUNKED_DIR = f"{VOL_MOUNT}/output/voice_cloning/chunked"

# DAC runs ~86 tokens/sec. These encode model/codec assumptions, not arbitrary values:
CHARS_TO_TOKENS = 1.8        # Spanish chars → audio tokens, for the min-length EOS floor
ROUND_CEILING_TOKENS = 900   # ~10.5s — a full round fits; cuts noise before it spirals

# A natural 8-turn conversation — tortilla-adjacent but more varied
DEFAULT_DIALOGUE = (
    "[S1] Oye, ¿ya escuchaste que van a abrir un restaurante nuevo en el barrio? "
    "[S2] Sí, me contaron. Dicen que es de cocina fusión, algo entre mexicano y japonés. "
    "[S1] ¿En serio? Eso suena rarísimo, pero también me da mucha curiosidad. "
    "[S2] A mí también. Mi prima fue a uno parecido en Madrid y dice que estaba buenísimo. "
    "[S1] Pues habrá que ir a probarlo cuando abra. ¿El fin de semana que viene tienes algo? "
    "[S2] Creo que estoy libre el sábado por la tarde. ¿A qué hora quedamos? "
    "[S1] Podríamos ir sobre las dos, comemos y luego damos una vuelta por el centro. "
    "[S2] Me parece perfecto. Yo reservo la mesa para no tener que esperar."
)


def parse_turns(text: str) -> list:
    """Split a tagged dialogue string into individual speaker turns."""
    import re
    parts = re.split(r"(\[S[12]\])", text.strip())
    turns = []
    for i in range(1, len(parts), 2):
        tag = parts[i]
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            turns.append(f"{tag} {content}")
    return turns


def _ogg_to_wav(audio_b64: str, label: str, max_sec: float = 5.0) -> str:
    """Decode b64 ogg, trim to max_sec, convert to 44100Hz mono WAV, return temp path."""
    import subprocess, tempfile, base64
    audio_bytes = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        f.write(audio_bytes)
        tmp_src = f.name
    tmp_wav = tmp_src.replace(".ogg", f"_{label}.wav")
    subprocess.run(
        ["ffmpeg", "-i", tmp_src, "-t", str(max_sec), "-ar", "44100", "-ac", "1", tmp_wav, "-y"],
        check=True, capture_output=True,
    )
    return tmp_wav


@app.function(
    image=image,
    volumes={VOL_MOUNT: volume},
    gpu="a10g",
    timeout=30 * MINUTES,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
def generate_chunked(s1_audio_b64: str, s1_transcript: str, dialogue: str) -> dict:
    """
    Generate a dialogue in [S1]+[S2] rounds, conditioned on a single 5s S1 prompt.
    Model is loaded once; the prompt is pre-converted to WAV.
    """
    import subprocess
    import torch
    import soundfile as sf

    volume.reload()
    os.makedirs(CHUNKED_DIR, exist_ok=True)

    # Single clean 5s S1 prompt — the known-good regime. The spliced 2-speaker prompt
    # is too OOD for this audio-conditioning-free fine-tune and degenerates into noise.
    # S2 becomes a consistent model-invented contrast voice anchored by the S1 prompt.
    s1_wav = _ogg_to_wav(s1_audio_b64, "s1")
    clone_prefix = f"[S1] {s1_transcript.strip()}"
    print(f"S1 prompt (5s): {s1_wav}\nClone prefix: {clone_prefix}")

    # Load model once
    device = torch.device("cuda")
    from dia.model import Dia
    print("Loading model...")
    dia = Dia.from_local(CONFIG_PATH, CKPT_PATH, device=device)

    # Group turns into [S1]+[S2] rounds — each chunk = one full round, both speakers.
    turns = parse_turns(dialogue)
    pairs = [turns[i:i + 2] for i in range(0, len(turns), 2)]
    print(f"\n{len(pairs)} rounds to generate:")
    for idx, pair in enumerate(pairs):
        print(f"  [{idx}] {' '.join(pair)[:80]}")

    chunk_meta = []

    for idx, pair in enumerate(pairs):
        # One round: "[S1] ... [S2] ...". No trailing tag — it invited the model to
        # ramble into hallucinated extra turns → noise.
        gen_text = " ".join(pair)
        full_text = f"{clone_prefix} {gen_text}"

        # Floor to prevent premature EOS before S2 speaks; ceiling is a tight safety net.
        body_chars = sum(len(t.split("] ", 1)[-1]) for t in pair)
        min_eos = int(body_chars * CHARS_TO_TOKENS)
        print(f"\n--- Round {idx} ---\n{full_text}\n  (min_eos={min_eos}, force_eos_at={ROUND_CEILING_TOKENS})\n")

        with torch.inference_mode():
            audio = dia.generate(
                text=full_text,
                max_tokens=1536,  # KV cache holds 5s prompt (~431) + generation
                cfg_scale=3.0,
                temperature=1.3,
                top_p=0.95,
                audio_prompt_path=s1_wav,
                extra_steps_after_eos=18,
                min_steps_before_eos=min_eos,
                force_eos_at=ROUND_CEILING_TOKENS,
            )

        chunk_path = f"{CHUNKED_DIR}/chunk_{idx:02d}.wav"
        sf.write(chunk_path, audio, 44100, subtype="PCM_16")
        duration = len(audio) / 44100
        print(f"  → saved {chunk_path} ({duration:.1f}s)")
        chunk_meta.append({"idx": idx, "turn": gen_text, "duration_sec": round(duration, 2), "path": chunk_path})

    # Concatenate all chunks with ffmpeg
    concat_list = f"{CHUNKED_DIR}/concat.txt"
    with open(concat_list, "w") as f:
        for c in chunk_meta:
            f.write(f"file '{c['path']}'\n")
    final_path = f"{CHUNKED_DIR}/full_conversation.wav"
    subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-ar", "44100", final_path, "-y"],
        check=True, capture_output=True,
    )
    total = sum(c["duration_sec"] for c in chunk_meta)
    print(f"\nFinal conversation: {final_path} ({total:.1f}s)")

    volume.commit()
    return {"chunks": chunk_meta, "final_path": final_path, "total_sec": round(total, 2)}


@app.local_entrypoint()
def main(
    audio_path: str = "voice-cloning-experiments/voicespeaker1.ogg",
    new_lines: str = "[S1] Bueno, la verdad es que esto es una prueba bastante interesante. [S1] Vamos a ver si el modelo logra copiar mi voz en español.",
    transcript: str = "",
    transcribe_only: bool = False,
):
    audio_bytes = Path(audio_path).read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    filename = Path(audio_path).name

    # Step 1: transcribe if no transcript provided
    if not transcript:
        print("Transcribing audio sample on Modal...")
        transcript = transcribe.remote(audio_b64, filename)
        print(f"\nTranscript: {transcript}")

    if transcribe_only:
        return

    # Step 2: voice clone
    print(f"\nRunning voice cloning with new lines:\n{new_lines}\n")
    result = voice_clone.remote(audio_b64, transcript, new_lines, filename)

    print(f"\n{'='*60}")
    print(f"Transcript used: {result['transcript']}")
    print(f"Full prompt:\n{result['full_text']}")
    print(f"Output: {result['audio_path']} ({result['duration_sec']}s)")
    print(f"{'='*60}")
    print(f"\nDownload with:")
    print(f"  modal volume get {VOL_NAME} output/voice_cloning/voice_clone_result.wav ./results/")


_SCRIPT_DIR = Path(__file__).parent


@app.local_entrypoint()
def chunked(
    s1_audio: str = "",
    s1_transcript: str = "muy salas por esas tortillas 10 de 10 compralas de verdad son lo máximo lo máximo lo máximo",
    dialogue: str = "",
):
    """Generate a dialogue in [S1]+[S2] rounds, conditioned on a single 5s S1 prompt."""
    import subprocess as sp

    s1_path = Path(s1_audio) if s1_audio else _SCRIPT_DIR / "voice-cloning-experiments/speaker1sample.ogg"
    s1_b64 = base64.b64encode(s1_path.read_bytes()).decode()

    dlg = dialogue.strip() if dialogue.strip() else DEFAULT_DIALOGUE
    print(f"\nGenerating {len(parse_turns(dlg))} turns chunked...")
    print(f"S1: {s1_path.name}\n")

    result = generate_chunked.remote(s1_b64, s1_transcript, dlg)

    out_dir = _SCRIPT_DIR / "voice-cloning-experiments/results/chunked"
    out_dir.mkdir(parents=True, exist_ok=True)

    modal_out = "output/voice_cloning/chunked"
    print("\nDownloading results...")
    sp.run(["modal", "volume", "get", "--force", VOL_NAME, modal_out, str(out_dir)], check=True)

    print(f"\n{'='*60}")
    for c in result["chunks"]:
        print(f"  chunk {c['idx']:02d} ({c['duration_sec']}s): {c['turn'][:60]}")
    print(f"\n  FULL: {result['total_sec']}s → {out_dir}/chunked/full_conversation.wav")
    print(f"{'='*60}")
