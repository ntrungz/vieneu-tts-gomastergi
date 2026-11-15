# Giọng gomastergi — bản r16 (bản đã chọn)

- Train: LoRA r16, target backbone, 3 epoch, 249 clip / 53 phút (242 train + 7 eval)
- Final eval loss 5.2424, acc_cb0 0.125
- Giọng đã đóng gói sẵn trong `voices_v3_turbo.json` (tên `gomastergi`, nữ)

## Cách dùng (Python SDK)

```python
from vieneu import Vieneu

tts = Vieneu(mode="v3turbo", backbone_repo="final-gomastergi-r16")
audio = tts.infer("Xin chào các bạn, hôm nay mình sẽ hướng dẫn các bạn cách build đội hình nhé.", voice="gomastergi")
tts.save(audio, "output.wav")
```

Lưu ý: trên máy có GPU thì backend tự dùng PyTorch, trên CPU thì thêm `backend="pytorch"` khi tạo Vieneu.
