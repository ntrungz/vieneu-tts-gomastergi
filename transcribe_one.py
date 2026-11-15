"""Generic single-file transcribe with faster-whisper small, same settings as
transcribe_k7_rest.py (beam 5, VAD, int8 CPU). Usage:
  ./venv/bin/python transcribe_one.py <audio_path> <out_id> [offset_s]
Writes transcripts/<out_id>.json with start/end in absolute seconds."""
import subprocess, numpy as np, json, time, sys
from pathlib import Path
from faster_whisper import WhisperModel

base = Path(__file__).resolve().parent
AUDIO = base / "audio"
OUT = base / "transcripts"
OUT.mkdir(parents=True, exist_ok=True)

audio_path = Path(sys.argv[1])
vid = sys.argv[2]
OFFSET = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

def decode(path, ss):
    cmd = ["ffmpeg", "-v", "error"]
    if ss:
        cmd += ["-ss", str(ss)]
    cmd += ["-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    return np.frombuffer(subprocess.check_output(cmd), dtype=np.float32)

t0 = time.time()
model = WhisperModel("small", device="cpu", compute_type="int8")
print(f"model loaded {time.time()-t0:.1f}s", flush=True)
audio = decode(audio_path, OFFSET)
print(f"{vid}: {len(audio)/16000:.0f}s audio (offset {OFFSET}s)", flush=True)
t0 = time.time()
segments, info = model.transcribe(audio, beam_size=5, vad_filter=True)
segs = []
for s in segments:
    segs.append({"start": round(float(s.start) + OFFSET, 2),
                 "end": round(float(s.end) + OFFSET, 2),
                 "text": s.text.strip(),
                 "avg_logprob": round(float(s.avg_logprob), 3),
                 "no_speech_prob": round(float(s.no_speech_prob), 3)})
(OUT / f"{vid}.json").write_text(json.dumps(
    {"id": vid, "language": info.language,
     "lang_prob": round(float(info.language_probability), 3),
     "segments": segs}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{vid}: {len(segs)} segments, {info.language} "
      f"{info.language_probability:.2f}, took {time.time()-t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
