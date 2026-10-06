# Lab Day 2 — DeepWeeds

## 1. Mục tiêu và thiết lập

Thí nghiệm đánh giá backbone, công thức huấn luyện và phương pháp suy luận trên DeepWeeds. Dữ liệu được chia cố định theo fold 0: 10.501 ảnh train, 3.501 ảnh validation và 3.507 ảnh test; tổng cộng 17.509 ảnh, không có trùng lặp giữa các tập và không thiếu ảnh.

Các thí nghiệm dùng ảnh 224×224, ImageNet normalization, AdamW, batch size 64, 12 epoch, AMP và learning rate `1e-4` cho backbone/`1e-3` cho head. Chỉ validation được dùng để chọn cấu hình; test chỉ được mở ở vòng chung kết với ba seed (0, 1, 2).

## 2. So sánh backbone

| Experiment | Backbone | Params (M) | GMAC | Macro-F1 val | Top-1 val | s/epoch |
|---|---|---:|---:|---:|---:|---:|
| B01 | ResNet-50 | 23.53 | 4.132 | 0.8024 | 0.8558 | 48.10 |
| B02 | ResNeXt-50 32×4d | 23.00 | 4.286 | 0.8560 | 0.8880 | 64.03 |
| B03 | ConvNeXt-Tiny | 27.83 | 4.455 | **0.9657** | **0.9734** | 57.20 |
| B04 | DeiT-Small | 21.67 | 4.241 | 0.9465 | 0.9606 | 38.36 |
| B05 | EfficientNet-B0 | 4.02 | 0.385 | 0.8101 | 0.8575 | **33.27** |

ConvNeXt-Tiny đạt validation macro-F1 cao nhất và được chọn cho các thí nghiệm tiếp theo. EfficientNet-B0 nhẹ và nhanh hơn rõ rệt nhưng chất lượng thấp hơn. Kết quả cũng cho thấy GMAC không trực tiếp quyết định thời gian mỗi epoch: DeiT có GMAC gần các backbone lớn nhưng thời gian thấp hơn, trong khi EfficientNet có chi phí tính toán nhỏ nhất.

Biểu đồ: `curves/B01_resnet50_seed0.png` đến `curves/B05_efficientnet_b0_seed0.png`.

## 3. Ablation công thức huấn luyện

Các thí nghiệm T01–T11 được thực hiện trên ConvNeXt-Tiny; mỗi cấu hình thay đổi một yếu tố so với T00 khi phù hợp.

| Experiment | Thay đổi | Macro-F1 val | Δ so với T00 |
|---|---|---:|---:|
| T00 | Baseline | 0.9657 | 0.0000 |
| T01 | Khởi tạo từ scratch | 0.3215 | -0.6457 |
| T02 | Đóng băng backbone | 0.8545 | -0.1127 |
| T03 | Augmentation màu | 0.9588 | -0.0084 |
| T04 | RandAugment | 0.9638 | -0.0034 |
| T05 | CutMix | **0.9743** | **+0.0071** |
| T06 | Label smoothing | 0.9638 | -0.0035 |
| T07 | Focal loss | 0.9588 | -0.0084 |
| T08 | Class-weighted CE | 0.9671 | -0.0001 |
| T09 | Balanced sampler | 0.9668 | -0.0004 |
| T10 | EMA | 0.9635 | -0.0037 |
| T11 | CutMix + label smoothing + EMA | 0.9695 | +0.0023 |

CutMix là thay đổi đơn lẻ hiệu quả nhất và được chọn làm cấu hình cuối. Khởi tạo scratch và đóng băng backbone đều kém rõ rệt so với fine-tuning toàn bộ mạng. Cấu hình kết hợp T11 không vượt T05, vì vậy không giả định rằng các cải tiến cộng dồn một cách độc lập.

Biểu đồ tương ứng nằm trong `curves/T00_*.png` đến `curves/T11_*.png`.

## 4. Phương pháp suy luận

Trên validation với checkpoint T05:

| ID | Phương pháp | Macro-F1 | Top-1 | ECE | P50 batch 1 (ms) |
|---|---|---:|---:|---:|---:|
| I00 | Một view | 0.9743 | 0.9803 | 0.0057 | 7.95 |
| I01 | Hflip, trung bình xác suất | 0.9768 | 0.9823 | 0.0056 | 15.74 |
| I02 | Hflip, trung bình logit | 0.9768 | 0.9823 | 0.0052 | — |
| I03 | Bốn flip, trung bình xác suất | **0.9788** | **0.9837** | 0.0090 | — |
| I04 | Multi-scale 224/256/288 | 0.9775 | 0.9820 | 0.0173 | — |
| I07 | Temperature scaling | 0.9743 | 0.9803 | **0.0034** | — |
| I08 | Conv-BN fusion | 0.9743 | 0.9803 | 0.0057 | 7.88 |

I03 cho chất lượng validation cao nhất, còn temperature scaling cải thiện hiệu chuẩn ECE. Conv-BN fusion giảm nhẹ p50 từ 7.95 xuống 7.88 ms nhưng không thay đổi chất lượng. Do đó cấu hình chung kết giữ recipe T05; các số test được báo cáo cho F01 và baseline T00.

## 5. Chung kết trên test

F01 (ConvNeXt-Tiny + CutMix) được chạy với ba seed độc lập và so sánh với T00.

| Cấu hình | Macro-F1 test (mean ± std) | Top-1 test (mean ± std) | ECE test (mean ± std) |
|---|---:|---:|---:|
| F01 | **0.9745 ± 0.0022** | **0.9793 ± 0.0014** | **0.0043 ± 0.0011** |
| T00 | 0.9656 ± 0.0019 | 0.9732 ± 0.0017 | 0.0165 ± 0.0018 |

F01 cải thiện khoảng 0.009 macro-F1 và 0.006 top-1 so với baseline, đồng thời có ECE thấp hơn đáng kể. Độ lệch giữa ba seed nhỏ; kết luận này phù hợp với cả ba lần chạy thay vì chỉ dựa trên một seed.

Các dự đoán gốc được lưu trong `predictions/`, trong đó có `F01_seed<k>_test.csv` và `T00_seed<k>_test.csv`. Chỉ số theo từng lớp nằm trong sheet `PerClass` của `results.xlsx`.

## 6. Hạn chế và kết luận

Các backbone và ablation được quét chủ yếu với một seed; chỉ cấu hình cuối và baseline được lặp ba seed theo yêu cầu. Thời gian đo phụ thuộc GPU Tesla T4 và phiên bản PyTorch/timm của Kaggle, vì vậy nên xem đây là số đo trong môi trường thí nghiệm cụ thể.

Trong thiết lập này, ConvNeXt-Tiny là backbone tốt nhất về validation macro-F1; CutMix là thay đổi có lợi nhất trong nhóm ablation; và F01 là cấu hình chung kết ổn định, vượt baseline trên test. Toàn bộ bảng số liệu, biểu đồ và prediction files đi kèm trong thư mục submission.
