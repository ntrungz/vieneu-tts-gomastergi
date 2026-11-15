# Giọng gomastergi — bản v2 r16

- Train: LoRA r16, target backbone, 3 epoch, 577 clip / 122.5 phút (560 train + 17 eval), 105 steps
- Dataset v2: v1 249 clip (giữ 247) + k7-rest 248 + sample 10 + iKaFF 40 + Lns 32; lọc sạch (loại nhạc/ồn, transcript sửa Nod-Krai...)
- Final eval loss 5.1637 (v1: 5.2424), acc_cb0 0.158 (v1: ~0.13)
- Kernel: ngnhattrung/vieneu-tts-lora-gomastergi-v2-2x-t4-parallel, train 10.5 phút, cả 2 run exit 0
- Giọng đã đóng gói sẵn trong `voices_v3_turbo.json` (tên `gomastergi`, nữ)

## Cách dùng (Python SDK)

```python
from vieneu import Vieneu

tts = Vieneu(mode="v3turbo", backbone_repo="final-gomastergi-v2-r16")
audio = tts.infer("Xin chào các bạn, hôm nay mình sẽ hướng dẫn các bạn cách build đội hình nhé.", voice="gomastergi")
tts.save(audio, "output.wav")
```

Lưu ý: trên máy có GPU thì backend tự dùng PyTorch, trên CPU thì thêm `backend="pytorch"` khi tạo Vieneu.
