"""Build dataset v2 (~120 min, clean-first) for VieNeu-TTS LoRA gomastergi.

- Keeps all 249 v1 clips (re-cleaned with the extended fix map), copies wav.
- Adds new clips from transcripts/k7-rest-1800s.json (+ any top-up transcript
  present in TOPUP, e.g. sample-Y_d_o7czcDM).
- Strong clean filters (brief 15/11): segment quality (avg_logprob /
  no_speech_prob), hallucination markers, chars/s, RMS / clipping.
- Writes dataset-v2/raw_audio/*.wav + dataset-v2/metadata.csv.
  Does NOT touch dataset/ (v1) or build_dataset.py.
"""
import json, pathlib, re, shutil, subprocess
import numpy as np

base = pathlib.Path(__file__).resolve().parent
AUDIO = base / "audio"
TRANS = base / "transcripts"
V1 = base / "dataset"
OUT = base / "dataset-v2"
RAW = OUT / "raw_audio"
RAW.mkdir(parents=True, exist_ok=True)

# ---------- text cleaning (v1 map + v2 additions, only when certain) ----------
def clean(t):
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    # --- v1 systematic fixes ---
    t = t.replace("Jenshin", "Genshin").replace("jenshin", "Genshin")
    t = re.sub(r"\btần\b", "tuần", t, flags=re.I)
    t = t.replace("Tần thứ", "Tuần thứ").replace("tần thứ", "tuần thứ")
    t = t.replace("Hành trình 100.000 chơi", "Hành trình 100 ngày chơi")
    t = t.replace("chuyến hành trình 100.000 này", "chuyến hành trình 100 ngày này")
    t = t.replace("Sandron", "Sandrone")
    # --- v2 additions ---
    # Genshin 6.x region Nod-Krai, consistently misheard by Whisper small:
    t = t.replace("Dương North Cry", "vùng Nod-Krai")
    t = t.replace("North Cry", "Nod-Krai")
    t = t.replace("non cry", "Nod-Krai").replace("Non cry", "Nod-Krai")
    t = t.replace("Hennot cry", "Nod-Krai")
    t = t.replace("not crying", "Nod-Krai")  # "tượng thần Nod-Krai" context
    # "trạy" is not a Vietnamese word; "chạy map / chạy việc" fits context
    t = t.replace("trạy máp", "chạy map").replace("trạy việc", "chạy việc")
    t = t.replace("ấm nhạc", "âm nhạc").replace("Ơm nhạc", "âm nhạc")
    return t.strip()

# ---------- filters ----------
def has_repeated_word(txt):
    w = txt.split()
    return any(w[i] == w[i+1] == w[i+2] for i in range(len(w) - 2))

def has_music_marker(txt):
    if "[" in txt or "]" in txt:
        return True
    # standalone "nhạc" (music hallucination), but NOT legit "âm nhạc"
    if re.search(r"(?<!âm )\bnhạc\b", txt):
        return True
    return False

# ---------- stats ----------
stats = {"kept": 0, "total_s": 0.0}
reject = {}

def note(reason):
    reject[reason] = reject.get(reason, 0) + 1

rows = []
seen = set()

def wav_rms_ok(path):
    """RMS / clipping check on the cut wav (soundfile, float32)."""
    import soundfile as sf
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return False, "unreadable_wav"
    x = wav.mean(axis=1)
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    if rms < 0.02:
        return False, "rms_too_low"
    clip_frac = float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0
    if clip_frac > 0.01:
        return False, "clipping_heavy"
    return True, ""

def cut_wav(src, start, end, dest):
    s = max(0.0, start - 0.05)
    e = end + 0.05
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.2f}", "-to", f"{e:.2f}",
           "-i", str(src), "-ac", "1", "-ar", "24000", str(dest)]
    return subprocess.run(cmd).returncode == 0

def add_clip(fname, txt, dur, src, start, end):
    """Shared accept path for v1 (already-cut) and new clips."""
    ct = clean(txt)
    if len(ct) < 10:
        note("text_too_short"); return False
    if has_repeated_word(ct):
        note("repeated_word"); return False
    if has_music_marker(ct):
        note("music_marker"); return False
    cps = len(ct) / max(dur, 0.1)
    if cps < 5.0:
        note("chars_per_sec_low"); return False
    key = re.sub(r"\W+", "", ct.lower())
    if key in seen:
        note("duplicate_text"); return False
    seen.add(key)
    rows.append((fname, ct, dur))
    return True

# ---------- 1. v1 clips: copy wav, re-clean text ----------
v1_meta = (V1 / "metadata.csv").read_text(encoding="utf-8").splitlines()
v1_kept = 0
for line in v1_meta:
    if "|" not in line:
        continue
    fname, txt = line.split("|", 1)
    src_wav = V1 / "raw_audio" / fname
    if not src_wav.is_file():
        note("v1_wav_missing"); continue
    # duration from wav header
    import soundfile as sf
    try:
        info = sf.info(str(src_wav))
        dur = info.duration
    except Exception:
        note("v1_wav_unreadable"); continue
    if not (3.0 <= dur <= 15.2):
        note("v1_dur_out_of_range"); continue
    # RMS check on the v1 wav itself
    ok, why = wav_rms_ok(src_wav)
    if not ok:
        note(why); continue
    if add_clip(fname, txt, dur, None, 0, 0):
        shutil.copy2(src_wav, RAW / fname)
        v1_kept += 1
print(f"v1: kept {v1_kept}/{len(v1_meta)} clips")

# ---------- 2. new transcripts ----------
# source audio per transcript id
SRC = {
    "k7-rest-1800s": AUDIO / "k7ueemfrIpk.mp3",
    # top-up, only used if transcript json exists:
    "sample-Y_d_o7czcDM": AUDIO / "sample-Y_d_o7czcDM.wav",
    "iKaFFTNAd3w": AUDIO / "iKaFFTNAd3w.mp3",
    "Lnsrs5mscDs": AUDIO / "Lnsrs5mscDs.mp3",
}
for tid, src in SRC.items():
    p = TRANS / f"{tid}.json"
    if not p.exists():
        print(f"skip {tid}: no transcript yet")
        continue
    if not src.is_file():
        print(f"skip {tid}: no audio {src}")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    segs = d["segments"]
    # segment-level quality filter FIRST; dropped segments break merge chains.
    # NOTE (verified 15/11 ~04:00): no_speech_prob is NOT usable on this audio
    # — even nsp 0.9-1.0 segments are real narration with avg_logprob -0.27
    # (Whisper inflates nsp because of background game music; the values even
    # repeat per 30s window: 0.64 / 0.763 / 0.909). VAD already removed pure
    # music (v1: GBpLHAAtA5U gave 0 segments). So filter only on avg_logprob.
    good = []
    for s in segs:
        lp = s.get("avg_logprob")
        if lp is not None and lp < -1.0:
            note("avg_logprob<-1.0"); continue
        good.append(s)
    print(f"{tid}: {len(good)}/{len(segs)} segments pass quality filter")
    i = 0
    idx = 0
    kept_here = 0
    while i < len(good):
        j = i
        end = good[i]["end"]
        txt = good[i]["text"]
        while (j + 1 < len(good) and good[j + 1]["start"] - end < 1.0
               and good[j + 1]["end"] - good[i]["start"] <= 15.0):
            j += 1
            end = good[j]["end"]
            txt += " " + good[j]["text"]
        L = end - good[i]["start"]
        start = good[i]["start"]
        if 3.0 <= L <= 15.0:
            fname = f"{tid}_{idx:03d}.wav"
            dest = RAW / fname
            if cut_wav(src, start, end, dest):
                ok, why = wav_rms_ok(dest)
                if not ok:
                    dest.unlink(missing_ok=True)
                    note(why)
                elif add_clip(fname, txt, L, src, start, end):
                    kept_here += 1
                    idx += 1
                else:
                    dest.unlink(missing_ok=True)
            else:
                note("ffmpeg_fail")
        else:
            note("clip_dur_out_of_range")
        i = j + 1
    print(f"{tid}: kept {kept_here} clips")

# ---------- write ----------
total = sum(r[2] for r in rows)
print(f"\nTOTAL v2: clips={len(rows)} total={total:.0f}s = {total/60:.1f} min")
print("reject reasons:", dict(sorted(reject.items(), key=lambda kv: -kv[1])))
with open(OUT / "metadata.csv", "w", encoding="utf-8") as f:
    for fname, txt, L in rows:
        f.write(f"{fname}|{txt}\n")
print("wrote", OUT / "metadata.csv")
for r in rows[:5]:
    print(r)
