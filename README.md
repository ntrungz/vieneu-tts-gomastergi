# VieNeu-TTS — Giọng @gomastergi (LoRA finetune)

Finetune giọng đọc tiếng Việt của nhà sáng tạo YouTube **[@gomastergi](https://www.youtube.com/@gomastergi)**
(chuyên video Genshin Impact 100/200 ngày) lên [VieNeu-TTS v3 Turbo](https://github.com/pnnbao97/VieNeu-TTS)
bằng LoRA. Bản chốt: **v2 r16**.

> **Lưu ý về quyền sử dụng giọng nói:** giọng này là giọng của @gomastergi.
> Repo này chỉ chứa code, dữ liệu text và cấu hình để tái tạo lại quá trình finetune —
> **không chứa file audio, clip giọng hay trọng số model**. Việc phát hành lại
> model giọng nói cần được chủ giọng đồng ý. Model đã train chỉ chia sẻ riêng tư
> (Hugging Face + Drive, xem bên dưới), không phát tán công khai.

## Kết quả

| Bản | Dataset | Loss cuối | acc_cb0 | Chọn? |
|---|---|---|---|---|
| v1 r16 | 249 clip / 53 phút | 5.2424 | ~0.13 | — |
| v1 r32-all | 249 clip / 53 phút | 5.2052 | ~0.13 | — |
| **v2 r16** | **577 clip / 121.9 phút** | **5.1637** | **0.158** | ✅ **bản chốt** |
| v2 r32-all | 577 clip / 121.9 phút | 5.1264 | 0.156 | loss thấp nhất, nhưng nghe thực tế thì r16 được chọn |

Train trên Kaggle 2× Tesla T4 song song (mỗi GPU một run), 3 epoch, 105 steps/run,
chỉ ~10.5 phút train cho v2. Chi tiết đầy đủ: [`V2_RUN_LOG.md`](V2_RUN_LOG.md).

## Model đã train ở đâu

Trọng số **không có trong repo này** (file >100 MB, GitHub không nhận). Lấy ở:

- Hugging Face (riêng tư): https://huggingface.co/ntrung/VieNeu-TTS-v3-Turbo-gomastergi-v2
- Drive zip `gomastergi-v2-r16.zip` (~247 MB)

Cấu hình model (không chứa trọng số) có sẵn trong repo:
[`final-gomastergi-v2-r16/config.json`](final-gomastergi-v2-r16/config.json),
[`train_args.json`](final-gomastergi-v2-r16/train_args.json),
[`voices_v3_turbo.json`](final-gomastergi-v2-r16/voices_v3_turbo.json) (preset `gomastergi`).

Dùng thử (cần model ở trên, đặt cạnh script):

```python
from vieneu import Vieneu

tts = Vieneu(mode="v3turbo", backbone_repo="final-gomastergi-v2-r16")
audio = tts.infer("Xin chào các bạn, hôm nay mình sẽ hướng dẫn các bạn cách build đội hình nhé.",
                  voice="gomastergi")
tts.save(audio, "output.wav")
```

Hoặc chạy `python chay_giong_gomastergi.py` rồi gõ câu cần đọc.

## Dataset

Nguồn: video YouTube của @gomastergi (tải bằng `yt-dlp`, transcribe bằng
faster-whisper `small`, beam 5, VAD, CPU int8). File audio gốc **không** kèm
trong repo; `dataset*/metadata.csv` giữ lại toàn bộ text (`file|text`).

| Video (YouTube ID) | Nội dung | Dùng cho |
|---|---|---|
| `OdYsSibmaEA`, `maLKp9zJzOg`, `JV15Svc5wRE` | Genshin vlog | v1 |
| `k7ueemfrIpk` (phần 1 + phần còn lại) | Hành trình 100 ngày chơi | v1 + v2 |
| `GBpLHAAtA5U` | Chỉ nhạc, không lời | loại (0 segment) |
| `sample-Y_d_o7czcDM` | Short 2.2' | v2 |
| `iKaFFTNAd3w` | Tóm tắt Nod-Krai 13.9' | v2 |
| `Lnsrs5mscDs` | Nhật ký 200 ngày 8.3' | v2 |

- **v1:** 249 clip / 53 phút (`dataset/metadata.csv`).
- **v2:** 577 clip / 121.9 phút (`dataset-v2/metadata.csv`) — giữ 247 clip v1
  (re-clean text), thêm 330 clip mới. Lọc: độ dài 3–15s, `avg_logprob >= -1.0`,
  không lặp từ, không marker nhạc, chars/s ≥ 5, RMS/không clip nặng; sửa lỗi
  nghe nhầm hệ thống của Whisper (vd. `North Cry` → `Nod-Krai`).
  Lưu ý đã kiểm chứng: **không lọc theo `no_speech_prob`** — với audio có nhạc
  nền game nó là false positive hàng loạt (xem `V2_RUN_LOG.md`).
- Transcript Whisper gốc: [`transcripts/`](transcripts/) (9 file JSON).

## Tái tạo lại

```bash
# 1. Tải audio nguồn (yt-dlp) vào audio/, transcribe thành transcripts/*.json
python transcribe_one.py audio/<video>.mp3 <video_id>
#    (hoặc transcribe_shorts.py / transcribe_k7.py / transcribe_k7_rest.py)

# 2. Cắt clip + build metadata
python build_dataset.py        # v1: dataset/metadata.csv
python build_dataset_v2.py     # v2: dataset-v2/metadata.csv

# 3. Đóng gói parquet cho Kaggle (cần upstream VieNeu-TTS checkout)
#    git clone https://github.com/pnnbao97/VieNeu-TTS.git ../vieneu-tts
python prepare_local.py --dataset-dir dataset-v2
#    -> dataset-v2/train.parquet (cũng đã up sẵn lên Kaggle, xem dưới)

# 4. Train trên Kaggle (2x T4, xem kaggle-kernel-v2/)
#    - Dataset Kaggle (riêng tư): ngnhattrung/vieneu-gomastergi-120min
#    - Kernel: ngnhattrung/vieneu-tts-lora-gomastergi-v2-2x-t4-parallel
#      (kaggle-kernel-v2/train.py + kernel-metadata.json)
#    - v1 tương ứng: ngnhattrung/vieneu-gomastergi-50min,
#      ngnhattrung/vieneu-tts-lora-gomastergi-2x-t4-parallel

# 5. Test A/B các bản (câu ngắn + câu dài kiểu audiobook)
python infer_v2_ab.py
```

`train.parquet` không kèm trong repo (file nhị phân, và audio clip đã loại) —
tái tạo bằng `prepare_local.py` hoặc lấy từ dataset Kaggle riêng tư ở trên.

## Cấu trúc repo

```
build_dataset.py / build_dataset_v2.py  # cắt clip + metadata.csv
transcribe_*.py                          # Whisper transcribe (faster-whisper small)
prepare_local.py                         # đóng gói train.parquet (bản sao logic upstream)
transcripts/                             # transcript Whisper gốc (JSON)
dataset/ / dataset-v2/                   # metadata.csv (text; audio không kèm)
kaggle-kernel/ / kaggle-kernel-v2/       # train.py + kernel-metadata.json (Kaggle 2x T4)
kaggle-dataset/ / kaggle-dataset-v2/     # dataset-metadata.json (train.parquet ở Kaggle)
final-gomastergi-v2-r16/                 # config + train_args + voices preset (không trọng số)
final-gomastergi-r16/                    # config v1 (không trọng số)
infer_v2_ab.py / infer_v2_ab.log         # test A/B v1r16 vs v2r16 vs v2r32
chay_giong_gomastergi.py                 # chạy thử giọng (CLI)
gomastergi-colab.ipynb                   # chạy giọng trên Colab
V2_RUN_LOG.md                            # nhật ký chạy v2 đầy đủ
```

## Ghi công

- Base model + code train: [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS)
  (pnnbao97), model gốc `pnnbao-ump/VieNeu-TTS-v3-Turbo` — Apache 2.0.
- Giọng: [@gomastergi](https://www.youtube.com/@gomastergi) (video Genshin 100/200 ngày).

## License

Apache License 2.0 — xem [`LICENSE`](LICENSE). Upstream VieNeu-TTS cũng Apache 2.0.
