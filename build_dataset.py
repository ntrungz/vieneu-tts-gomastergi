import json, pathlib, re, subprocess
base = pathlib.Path(__file__).resolve().parent
AUDIO = base/"audio"; TRANS = base/"transcripts"
OUT = base/"dataset"; RAW = OUT/"raw_audio"
RAW.mkdir(parents=True, exist_ok=True)
# source file per transcript id
SRC = {
 "OdYsSibmaEA": AUDIO/"OdYsSibmaEA.mp3",
 "maLKp9zJzOg": AUDIO/"maLKp9zJzOg.mp3",
 "JV15Svc5wRE": AUDIO/"JV15Svc5wRE.mp3",
 "k7-part1-1800s": AUDIO/"k7-part1-1800s.mp3",
}
def clean(t):
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace("Jenshin","Genshin").replace("jenshin","Genshin")
    t = re.sub(r"\btần\b", "tuần", t, flags=re.I)
    t = t.replace("Tần thứ","Tuần thứ").replace("tần thứ","tuần thứ")
    t = t.replace("Hành trình 100.000 chơi","Hành trình 100 ngày chơi")
    t = t.replace("chuyến hành trình 100.000 này","chuyến hành trình 100 ngày này")
    t = t.replace("Sandron","Sandrone")
    # remove trailing spaces, ensure ends with punctuation-ish (keep as is)
    return t.strip()
rows=[]
seen=set()
for tid, src in SRC.items():
    p = TRANS/f"{tid}.json"
    if not p.exists(): continue
    d=json.loads(p.read_text())
    segs=d["segments"]
    i=0
    idx=0
    while i < len(segs):
        j=i; end=segs[i]["end"]; txt=segs[i]["text"]
        while j+1 < len(segs) and segs[j+1]["start"]-end < 1.0 and segs[j+1]["end"]-segs[i]["start"] <= 15.0:
            j+=1; end=segs[j]["end"]; txt+=" "+segs[j]["text"]
        L=end-segs[i]["start"]
        start=segs[i]["start"]
        if 3.0 <= L <= 15.0:
            ct=clean(txt)
            key=re.sub(r"\W+","",ct.lower())
            if len(ct)>=10 and key not in seen:
                seen.add(key)
                fname=f"{tid}_{idx:03d}.wav"
                # cut with ffmpeg, 24k mono wav, pad 0.05s each side clipped to source bounds
                s=max(0.0, start-0.05); e=end+0.05
                cmd=["ffmpeg","-y","-v","error","-ss",f"{s:.2f}","-to",f"{e:.2f}","-i",str(src),"-ac","1","-ar","24000",str(RAW/fname)]
                r=subprocess.run(cmd)
                if r.returncode==0:
                    rows.append((fname, ct, L))
                    idx+=1
                else:
                    print("ffmpeg fail", fname)
        i=j+1
total=sum(r[2] for r in rows)
print(f"clips={len(rows)} total={total:.0f}s = {total/60:.1f} min")
with open(OUT/"metadata.csv","w",encoding="utf-8") as f:
    for fname, txt, L in rows:
        f.write(f"{fname}|{txt}\n")
print("wrote", OUT/"metadata.csv")
for r in rows[:5]: print(r)
