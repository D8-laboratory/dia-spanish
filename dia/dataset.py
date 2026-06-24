from pathlib import Path

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

import dac
from .config import DiaConfig




class LocalDiaDataset(Dataset):
    """Dataset loader from a local CSV (pipe-separated) and audio folder."""
    def __init__(self, csv_path: Path, audio_root: Path, config: DiaConfig, dac_model: dac.DAC):
        self.df = pd.read_csv(csv_path, sep=r"\s*\|\s*", engine="python", names=["audio", "text", "channel"])
        self.audio_root = audio_root
        self.config = config
        self.dac_model = dac_model

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = row["text"]
        channel = row.get("channel", None)
        if channel and pd.notna(channel):
            text = f"[{channel}]{text}"

        audio_path = self.audio_root / row["audio"]
        waveform, sr = torchaudio.load(audio_path)

        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform[:1]  # only take 1 channel if stereo

        waveform = waveform.unsqueeze(0)  # (1, 1, T)
        with torch.no_grad():
            audio_tensor = self.dac_model.preprocess(waveform, 44100).to(
                next(self.dac_model.parameters()).device
            )
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            encoded = encoded.squeeze(0).transpose(0, 1)  # (T, C)

        return text, encoded, waveform


class HFDiaDataset(Dataset):
    """Dataset loader từ Hugging Face Datasets object."""

    def __init__(self, hf_dataset, config: DiaConfig, dac_model: dac.DAC):
        self.dataset = hf_dataset
        self.config = config
        self.dac_model = dac_model

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        # Xử lý text tag
        text = sample["text"]
        channel = sample.get("channel", None)
        lang = sample.get("language", None)

        if channel and isinstance(channel, str) and channel.strip():
            text = f"[{channel}]{text}"
        elif lang and isinstance(lang, str):
            text = f"[{lang}]{text}"

        # Xử lý audio
        audio_info = sample["audio"]
        waveform = torch.tensor(audio_info["array"], dtype=torch.float32)

        # Đảm bảo waveform shape (1, 1, T)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform[:1].unsqueeze(0)  # lấy 1 channel đầu

        # Resample nếu không phải 44100 Hz
        sr = audio_info.get("sampling_rate", 44100)
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)

        with torch.no_grad():
            audio_tensor = self.dac_model.preprocess(waveform, 44100).to(next(self.dac_model.parameters()).device)
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            encoded = encoded.squeeze(0).transpose(0, 1)  # (T, C)

        return text, encoded, waveform



class HFDiaIterDataset(torch.utils.data.IterableDataset):
    """Iterable wrapper for a HF streaming Dataset that has `audio.array` & `text`."""
    def __init__(self, hf_iterable, config: DiaConfig, dac_model: dac.DAC):
        super().__init__()
        self.dataset = hf_iterable
        self.config = config
        self.dac_model = dac_model

    def __iter__(self):
        for sample in self.dataset:
            lang = sample.get("language", None)
            # Lấy thông tin channel và chuẩn hóa
            channel = sample.get("channel", "").replace("@", "").lower()
            speaker_tag = f"[{channel}]" if channel else "[unk]"   
            # Ghép tag speaker + text
            text = speaker_tag + sample["text"]
            audio_info = sample['audio']
            waveform = torch.tensor(audio_info['array'], dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            sr = audio_info.get('sampling_rate', 44100)
            if sr != 44100:
                waveform = torchaudio.functional.resample(waveform, sr, 44100)
            with torch.no_grad():
                audio_tensor = (
                    self.dac_model.preprocess(waveform, 44100)
                    .to(next(self.dac_model.parameters()).device)
                )
                _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
                encoded = encoded.squeeze(0).transpose(0, 1)
            yield text, encoded, waveform

class VietnameseDiaDataset(HFDiaIterDataset):
    def __init__(self, dataset, dia_cfg, dac_model):
        super().__init__(dataset, dia_cfg, dac_model)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 1) Thêm tag ngôn ngữ [vi]
        text = item["text"]
        if not text.startswith("[vi]"):
            text = f"[vi]{text}"

        # 2) Xử lý audio về 44.1 kHz
        audio_array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        if sr != 44100:
            audio_array = torchaudio.functional.resample(
                torch.tensor(audio_array),
                orig_freq=sr,
                new_freq=44100
            ).numpy()

        # 3) Mã hoá DAC (tần số codec) từ đoạn audio
        encoding = self.get_dac_encoding(audio_array)

        return text, encoding, audio_array


class SpanishDiaDataset(Dataset):
    """Dataset loader for Spanish podcast data stored as HF dataset with audio_path.

    Each sample has:
      - audio_path: str  (path to WAV file on disk, 44.1kHz mono)
      - text: str         (already contains [S1]/[S2] speaker tags)
      - language: str     ("es")

    Prepends [es] language tag. Audio loaded from path via torchaudio.
    """

    def __init__(self, hf_dataset, config: DiaConfig, dac_model: dac.DAC):
        self.dataset = hf_dataset
        self.config = config
        self.dac_model = dac_model

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]

        # Text: prepend [es] language tag if not already present
        text = sample["text"]
        if not text.startswith("[es]"):
            text = f"[es]{text}"

        # Load audio from path
        audio_path = sample["audio_path"]
        waveform, sr = torchaudio.load(audio_path)

        # Ensure mono
        if waveform.ndim == 2:
            waveform = waveform[:1]  # take first channel if stereo

        # Resample to 44100 if needed
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)

        # Shape: (1, 1, T) for DAC
        waveform = waveform.unsqueeze(0)  # (1, 1, T)

        with torch.no_grad():
            audio_tensor = self.dac_model.preprocess(waveform, 44100).to(
                next(self.dac_model.parameters()).device
            )
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            encoded = encoded.squeeze(0).transpose(0, 1)  # (T, C)

        return text, encoded, waveform
