"""A/B test v1 r16 (baseline) vs v2 r16 vs v2 r32-all, same 2 sentences.
Short = the exact v1 test sentence. Long = ~40-word audiobook-style
Vietnamese sentence (book clean txt not present locally, so a composed
narration-style sentence is used instead).
Outputs: test-v2ab-<model>-<short|long>.wav in this dir."""
import time
from pathlib import Path

from vieneu import Vieneu

BASE = str(Path(__file__).resolve().parent)
SHORT = "Xin chào các bạn, hôm nay mình sẽ hướng dẫn các bạn cách build đội hình cho người mới chơi Genshin Impact nhé."
LONG = ("Khi bạn hiểu được cách tâm trí con người vận hành, bạn sẽ nhận ra rằng "
        "hầu hết những quyết định tưởng chừng lý trí thật ra lại bị chi phối bởi "
        "cảm xúc và thói quen đã hình thành từ lâu.")

MODELS = [
    ("v1r16", f"{BASE}/final-gomastergi-r16"),
    ("v2r16", f"{BASE}/final-gomastergi-v2-r16"),
    ("v2r32", f"{BASE}/final-gomastergi-v2-r32-all"),
]

for tag, repo in MODELS:
    t0 = time.time()
    print(f"=== loading {tag} ({repo}) ===", flush=True)
    tts = Vieneu(mode="v3turbo", backend="pytorch", backbone_repo=repo)
    for name, text in [("short", SHORT), ("long", LONG)]:
        print(f"--- infer {tag} {name} ---", flush=True)
        audio = tts.infer(text, voice="gomastergi")
        out = f"{BASE}/test-v2ab-{tag}-{name}.wav"
        tts.save(audio, out)
        print(f"saved {out} ({time.time()-t0:.0f}s total)", flush=True)
print("ALL DONE", flush=True)
