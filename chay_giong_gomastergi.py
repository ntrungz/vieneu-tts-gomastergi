# Chạy giọng gomastergi (bản finetune v2 r16 — bản đã chọn) — chỉ cần Python, không cần GPU.
# Cách dùng:
#   1. Giải nén gomastergi-r16.zip, được thư mục final-gomastergi-r16
#   2. Đặt file này CÙNG THƯ MỤC với final-gomastergi-r16 (không đặt trong nó)
#   3. Cài đặt 1 lần:  pip install torch --index-url https://download.pytorch.org/whl/cpu
#                      pip install vieneu transformers
#   4. Chạy:  python chay_giong_gomastergi.py
#   5. Gõ câu cần đọc, Enter -> file output-....wav lưu ngay cạnh file này.
import time
from pathlib import Path

from vieneu import Vieneu

BASE = Path(__file__).resolve().parent / "final-gomastergi-v2-r16"
if not BASE.is_dir():
    print("Không thấy thư mục final-gomastergi-v2-r16 cạnh file này.")
    print("Hãy tải model (xem README) rồi đặt file này cùng thư mục với nó.")
    raise SystemExit(1)

print("Đang tải model, đợi một chút (lần đầu hơi lâu vì còn tải codec)...")
tts = Vieneu(mode="v3turbo", backend="pytorch", backbone_repo=str(BASE))
print("Xong! Gõ câu cần đọc, Enter để tạo giọng nói. Gõ 'thoat' để dừng.")
while True:
    text = input("> ").strip()
    if not text:
        continue
    if text.lower() in ("thoat", "exit", "quit"):
        break
    out = f"output-{int(time.time())}.wav"
    audio = tts.infer(text, voice="gomastergi")
    tts.save(audio, out)
    print("Đã lưu:", out)
