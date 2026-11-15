import subprocess, numpy as np, json, time
from pathlib import Path
from faster_whisper import WhisperModel
AUDIO = Path(__file__).resolve().parent / "audio"
OUT = Path(__file__).resolve().parent / "transcripts"
def decode(path):
    cmd = ["ffmpeg","-v","error","-i",str(path),"-ac","1","-ar","16000","-f","f32le","-"]
    return np.frombuffer(subprocess.check_output(cmd), dtype=np.float32)
t0=time.time()
model = WhisperModel("small", device="cpu", compute_type="int8")
print(f"model loaded {time.time()-t0:.1f}s", flush=True)
vid="k7-part1-1800s"
audio = decode(AUDIO / f"{vid}.mp3")
print(f"{vid}: {len(audio)/16000:.0f}s audio", flush=True)
t0=time.time()
segments, info = model.transcribe(audio, beam_size=5, vad_filter=True)
segs = [{"start": round(float(s.start),2), "end": round(float(s.end),2), "text": s.text.strip()} for s in segments]
(OUT / f"{vid}.json").write_text(json.dumps({"id": vid, "language": info.language, "lang_prob": round(float(info.language_probability),3), "segments": segs}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{vid}: {len(segs)} segments, {info.language} {info.language_probability:.2f}, took {time.time()-t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
