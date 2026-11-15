"""Local replica of upstream finetune/prepare_dataset.py main(), bypassing the
vieneu_lora package __init__ (which imports torch via data/model/lora) because
prepare itself is torch-free: it only needs safe_path + the vieneu ONNX SDK.
Logic (phonemize, _encode_ref_wav, speaker embedding, parquet schema, issue #198
check) is copied verbatim from upstream prepare_dataset.py."""
from __future__ import annotations
import argparse, importlib.util, os, sys, time
from pathlib import Path
import numpy as np
# load safe_path directly from file, bypassing package __init__
# Upstream VieNeu-TTS checkout (only utils.safe_path is needed).
# Set VIENEU_TTS_SRC to its path, or clone it next to this repo:
#   git clone https://github.com/pnnbao97/VieNeu-TTS.git ../vieneu-tts
VIENEU_TTS_SRC = Path(os.environ.get("VIENEU_TTS_SRC", "") or (Path(__file__).resolve().parent.parent / "vieneu-tts"))
spec = importlib.util.spec_from_file_location("vieneu_lora_utils", str(VIENEU_TTS_SRC / "finetune/vieneu_lora/utils.py"))
_utils = importlib.util.module_from_spec(spec); spec.loader.exec_module(_utils)
safe_path = _utils.safe_path

def read_metadata(path: Path):
    rows, extra = [], 0
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or (ln == 1 and line.lower().startswith("file_name|")):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                print(f"  skip line {ln}: expected file_name|text"); continue
            if len(parts) > 2 and parts[2].strip():
                extra += 1
            rows.append({"file_name": parts[0].strip(), "text": parts[1].strip()})
    return rows

def phonemize(text: str) -> str:
    from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions
    chunks, _ = normalize_to_chunks_v3_with_gaps(text, max_chars=1_000_000)
    return " ".join(phonemize_text_with_emotions(c) for c in chunks).strip()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default=str(Path(__file__).resolve().parent / "dataset"))
    ap.add_argument("--base", default="pnnbao-ump/VieNeu-TTS-v3-Turbo")
    args = ap.parse_args()
    ds_dir = safe_path(args.dataset_dir)
    meta = ds_dir / "metadata.csv"
    audio_dir = ds_dir / "raw_audio"
    out = ds_dir / "train.parquet"
    rows = read_metadata(meta)
    print(f"{len(rows)} clips listed in {meta}", flush=True)
    import soundfile as sf
    from vieneu import Vieneu
    tts = Vieneu(mode="v3turbo", backend="onnx", precision="fp32", backbone_repo=args.base)
    eng = tts.engine
    spk_enc = eng._ensure_speaker_encoder()
    recs, skipped, t0 = [], 0, time.perf_counter()
    for i, r in enumerate(rows, 1):
        p = safe_path(r["file_name"], base=audio_dir)
        if not p.is_file():
            print(f"  missing audio: {p}"); skipped += 1; continue
        wav, sr = sf.read(str(p), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        dur = len(wav) / sr
        if not (3.0 <= dur <= 15.2):
            print(f"  skip {r['file_name']}: {dur:.1f}s outside [3, 15.2]"); skipped += 1; continue
        phones = phonemize(r["text"])
        if not phones:
            print(f"  skip {r['file_name']}: empty phonemes"); skipped += 1; continue
        codes = eng._encode_ref_wav(wav, sr)
        # EMPIRICAL FIX (verified 02:07 15/11): the encoder's LAST frame is a
        # content-independent pad artifact [993, 419, ...] — cutting the wav by
        # 80/160/240ms leaves the last row unchanged, while pure silence encodes
        # as 482, not 993. Upstream [:n_frames] does not remove it in vieneu
        # 3.8.4, so drop the final frame; it carries no speech (<=80ms).
        codes = codes[:-1]
        emb = np.asarray(spk_enc.embed(wav, sr), dtype=np.float32)
        recs.append({"file_name": r["file_name"], "text": r["text"],
                     "phones": phones, "codes": codes.astype(np.int64).tolist(),
                     "speaker_embedding": emb.tolist(), "duration": float(dur)})
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    if not recs:
        raise SystemExit("No usable clips.")
    from collections import Counter
    last = Counter(x["codes"][-1][0] for x in recs)
    code, n_same = last.most_common(1)[0]
    if len(recs) >= 20 and n_same / len(recs) > 0.5:
        raise SystemExit(f"{n_same}/{len(recs)} rows end with codebook-0 = {code}: pad-frame artifact (issue #198).")
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(recs), str(out))
    total = sum(x["duration"] for x in recs)
    print(f"\nwrote {out}: {len(recs)} rows, {total/60:.1f} min of audio, skipped={skipped}", flush=True)

if __name__ == "__main__":
    main()
