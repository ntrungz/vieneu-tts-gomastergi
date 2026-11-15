# Finetune VieNeu-TTS v3 Turbo giọng @gomastergi — v2 (120 phút) — RUN LOG

Ngày: 2026-11-15 (UTC; giờ +07 = UTC+7). Job bắt đầu ~09:41 +07, xong train ~12:34 +07, xong test A/B ~12:54 +07.

## Dataset v2 — 577 clip / 121.9 phút (train.parquet 122.5 phút, skipped=0)

| Nguồn | Clip giữ | Ghi chú |
|---|---|---|
| v1 (249 clip) | 247 | Giữ nguyên, re-clean text; 2 clip rơi (lọc RMS/music) |
| k7-rest-1800s (phần còn lại video 100 ngày) | 248 | Transcribe 1060 seg, vi 1.00, 3845s |
| sample-Y_d_o7czcDM (2.2') | 10 | 49 seg, vi 0.99 — cùng giọng, đã kiểm chứng |
| iKaFFTNAd3w (tóm tắt Nod-Krai, 13.9') | 40 | 136 seg, vi 0.99, tải mới bằng yt-dlp |
| Lnsrs5mscDs (200-day diary, 8.3') | 32 | 156 seg, vi 1.00, tải mới — bù cho đủ 120' |
| **Tổng** | **577** | **121.9'** |

Loại theo lý do: clip_dur_out_of_range 29, music_marker 3, avg_logprob<-1.0: 3,
chars_per_sec_low 2, repeated_word 1. RMS/clipping: 0 clip bị loại.
GBpLHAAtA5U (chỉ nhạc) vẫn loại như v1.

### Độ lệch khỏi brief (có bằng chứng)
- **Bỏ lọc `no_speech_prob > 0.5`**: với audio này nó là false positive hàng loạt —
  456/1060 segment k7-rest bị loại, nhưng ngay cả các segment nsp 0.9–1.0 vẫn là
  lời thoại thật với avg_logprob −0.27 ("khiến phần lớn diện tích đảo Seirai bị
  hủy diệt..."). Whisper phồng nsp vì nhạc nền game; giá trị còn lặp theo cửa sổ
  30s (0.64 / 0.763 / 0.909). VAD đã loại nhạc thuần (v1: GBp cho 0 segment).
  Chỉ giữ lọc avg_logprob < −1.0 (loại đúng 3 segment). Lần build đầu với lọc
  nsp chỉ được 83.2' — sau khi bỏ còn 114.9', thêm Lns mới chạm 121.9'.
- **Sửa transcript**: giữ map v1 + Nod-Krai ("North Cry"/"non cry"/"Hennot cry"/
  "not crying" → "Nod-Krai", "Dương North Cry" → "vùng Nod-Krai"), "trạy máp/việc"
  → "chạy map/việc", "ấm nhạc/Ơm nhạc" → "âm nhạc". Không đoán mò tên riêng khác.
- **Service restart ~04:27 UTC** giết lần chạy prepare đầu (mới đọc xong metadata)
  và xoá kaggle CLI + yt-dlp vừa pip-install. Đã chạy lại prepare từ đầu và cài
  lại kaggle 2.2.5. Không mất dữ liệu (mọi thứ trong ~/workspace còn nguyên).
- **Kernel push 400 2 lần** (slug title/id không khớp — Kaggle suy slug từ title).
  Đã sửa: id `ngnhattrung/vieneu-tts-lora-gomastergi-v2-2x-t4-parallel`,
  title `VieNeu TTS LoRA gomastergi v2 - 2x T4 parallel` → push OK version 1.

## Kaggle
- Dataset: `ngnhattrung/vieneu-gomastergi-120min` (private), status ready.
  Đã tải lại verify: md5 khớp file local, 577 rows / 122.5'.
- Kernel: https://www.kaggle.com/code/ngnhattrung/vieneu-tts-lora-gomastergi-v2-2x-t4-parallel
  — RUNNING lúc 05:19 UTC, COMPLETE 05:34 UTC (chỉ ~15 phút, nhanh hơn dự kiến
  60–90'). Đã đọc log — COMPLETE thật, không phải giả như v2 của v1.
- GPU: 2× Tesla T4 15360 MiB. Train 10.5 phút, cả 2 run exit 0.

## Kết quả train (3 epoch, 105 steps/run, 560 train + 17 eval)

| Run | Final eval loss | acc_cb0 | v1 tương ứng |
|---|---|---|---|
| r16 backbone | **5.1637** | 0.158 | 5.2424 / ~0.13 |
| r32 target all | **5.1264** | 0.156 | 5.2052 / ~0.13 |

Cả 2 đều cải thiện ~0.08 loss và acc_cb0 0.13 → ~0.157 so với v1.
Merged model: 165 tensor/run (khớp v1), adapter + merged đầy đủ.

## Đóng gói
- `final-gomastergi-v2-r16/` (317M) và `final-gomastergi-v2-r32-all/` (331M):
  config.json, denoiser.onnx, speaker_encoder.onnx, update/, adapter_model.safetensors,
  train_args.json, voices_v3_turbo.json (preset `gomastergi`), README.md.
- Không ghi đè `final-gomastergi-r16/`. Chưa up Drive/HF (chờ parent).

## Test A/B (inference local, backend pytorch CPU)
Câu ngắn = câu test v1; câu dài ~40 từ kiểu audiobook (không lấy được clean txt
của sách trên máy local nên dùng câu tự soạn cùng phong cách).
- test-v2ab-v1r16-short/long.wav, test-v2ab-v2r16-short/long.wav,
  test-v2ab-v2r32-short/long.wav (6 file, xem infer_v2_ab.log).
- Chỉ số khách quan: RMS 0.17–0.20, peak ~1.0, silent nội bộ 13–15%,
  longest silence 0.4s, duration câu dài 9.4–9.6s cả 3 model — không có
  dropout/dead-air bất thường.
- **Cảnh báo về phương pháp**: Whisper small ASR trên các file test cho ra
  text lặp/vô nghĩa — NHƯNG file test v1 gốc (Trung đã nghe và chọn) cũng cho
  kết quả ASR y hệt như vậy. Kết luận: Whisper small không đáng tin để đánh giá
  giọng TTS này (nói nhanh, nối từ), KHÔNG phải bằng chứng audio v2 bị lỗi.
  Kiểm tra lặp phổ cũng không phân biệt được với file v1 gốc. **Việc chọn
  r16 hay r32-all cho v2 phải do Trung nghe trực tiếp quyết định** — như v1,
  Trung đã chọn r16 dù r32 loss thấp hơn.
- Đề xuất theo loss: r32-all (5.1264 thấp nhất). Đề xuất theo tiền lệ v1: để
  Trung nghe cả 2 rồi chọn.

## File liên quan
- build_dataset_v2.py, prepare_v2.log, transcribe_*.log, infer_v2_ab.py/log
- kaggle-dataset-v2/, kaggle-kernel-v2/, kaggle-output-v2-train/ (log + output)
- dataset-v2/ (metadata.csv + raw_audio + train.parquet)
