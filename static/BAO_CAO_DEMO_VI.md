# Báo Cáo Demo: HyperNetworks

## 1. Mục tiêu

Demo này hiện thực ý tưởng **dynamic hypernetwork** trong bài báo
[*HyperNetworks*](https://arxiv.org/abs/1609.09106) của David Ha, Andrew Dai,
Quoc V. Le. Trọng tâm của demo là biến thể **HyperLSTM**, nơi một mạng LSTM nhỏ
sinh ra các hệ số điều biến động cho LSTM chính ở từng bước thời gian.

## 2. Ý tưởng chính của bài báo

- Thay vì dùng đúng một bộ trọng số cố định cho mọi timestep, HyperLSTM cho một
  mạng phụ đọc trạng thái hiện tại và sinh ra các vector điều biến.
- Các vector này không tạo ra toàn bộ ma trận mới, mà **scale động** phần
  input-to-hidden, hidden-to-hidden và bias của từng cổng LSTM.
- Cách làm này giúp mô hình linh hoạt hơn nhưng vẫn giữ số tham số ở mức hợp lý.

## 3. Những gì đã hiện thực trong project này

- `dynamic_hypernetwork/hyperlstm.py`
  Cài đặt `HyperLSTMCell` và `HyperLSTM` theo đúng tinh thần Eq. (10)-(13).
- `run_char_experiment.py`
  CLI để train, lưu checkpoint, đánh giá, và generate text từ checkpoint.
- `compare_models.py`
  Script so sánh trực tiếp `LSTM` thường với `HyperLSTM`.
- `demo_commands.sh`
  Các lệnh chạy nhanh để trình bày demo.

## 4. Kịch bản demo đề xuất khi nộp/thuyết trình

1. Giới thiệu bài báo và nêu vấn đề:
   LSTM truyền thống chia sẻ cùng trọng số cho mọi timestep.
2. Trình bày ý tưởng:
   Hypernetwork sinh điều biến động để LSTM chính phản ứng linh hoạt hơn.
3. Chạy baseline `LSTM`.
4. Chạy `HyperLSTM`.
5. Mở file `artifacts/.../comparison.md` để so sánh:
   `loss`, `perplexity`, số tham số và sample text.
6. Kết luận:
   HyperLSTM tăng độ linh hoạt của mô hình; chi phí là huấn luyện phức tạp hơn.

## 5. Điểm mạnh của bản demo

- Có thể chạy end-to-end trên CPU.
- Có checkpoint `best.pt` và `last.pt`.
- Có log huấn luyện `history.jsonl`.
- Có sample text định kỳ trong thư mục `samples/`.
- Có baseline để đối chiếu, phù hợp với một bài demo học thuật.

## 6. Giới hạn hiện tại

- Đây là bản hiện thực giáo dục bằng PyTorch, không phải bản tái tạo nguyên xi
  toàn bộ pipeline TensorFlow gốc của tác giả.
- Kết quả phụ thuộc mạnh vào dataset, số bước train và hidden size.
- Để có kết quả đẹp hơn khi nộp, nên dùng `Tiny Shakespeare` hoặc corpus riêng
  của môn học với số bước train lớn hơn.

## 7. Cách chạy đề xuất

### Chạy nhanh

```bash
python run_char_experiment.py train \
  --model hyperlstm \
  --device cpu \
  --output-dir artifacts/hyperlstm_quick_demo \
  --steps 120 \
  --eval-every 30 \
  --sample-every 60
```

### So sánh hai mô hình trên Tiny Shakespeare

```bash
python compare_models.py \
  --download-tinyshakespeare \
  --device cpu \
  --output-dir artifacts/shakespeare_comparison \
  --steps 300 \
  --eval-every 50 \
  --sample-every 150 \
  --prompt "ROMEO:"
```

## 8. Tài liệu tham khảo

- Ha, D., Dai, A., Le, Q. V. (2016). *HyperNetworks*. arXiv:1609.09106.
- Link bài báo: https://arxiv.org/abs/1609.09106
- Ghi chú tham khảo triển khai: https://blog.otoro.net/2016/09/28/hyper-networks/
