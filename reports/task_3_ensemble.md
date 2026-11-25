# Kết hợp ML + DL (3.0–3.4)

Phần này kết hợp nhánh ML (`ML_Hieu`, tasks 1.1–1.5) và nhánh DL (tasks 2.1–2.5) cho Kaggle Playground Series S6E11. Mọi model của cả hai nhánh được đánh giá trên **cùng 5 fold**, và phương án kết hợp được chọn bằng **CV lồng**. Điểm leaderboard không được dùng để chọn.

## 1. Vấn đề ban đầu

| | ML (task 1.5) | DL (task 2.5) |
|---|---|---|
| Model | Shallow tree ensemble: LGBM `num_leaves` 3/4/8, Spline LR, XGB depth 3; rank blend; decimal-fraction features | MLP 128→64, one-hot |
| Đánh giá | `StratifiedKFold(5, shuffle=True, random_state=9)` | 1 holdout 80/20, seed 9 |
| Điểm | OOF 0.94792 · LB 0.94918 | Validation 0.94462 · LB 0.94263 |

Hai nhánh không dùng chung phép chia, và DL không có dự đoán OOF trên toàn bộ train. Vì vậy chưa thể blend hay stack đúng cách. Thử nghiệm co-training trước đó (`1.ML/src/run_co_training.py`) dùng pseudo-label với ngưỡng cố định và cho LB 0.94795, thấp hơn ML gốc.

## 2. Nền tảng chung (3.0)

- [`shared/folds/folds_5_seed9.csv`](../shared/folds/folds_5_seed9.csv): cột `id, fold`, tạo bằng đúng splitter của ML task 1.3–1.5. Notebook 3.0 đã kiểm tra từng dòng trùng với splitter đó. Mỗi fold có khoảng 139.992 dòng, tỷ lệ gian lận khoảng 1,67%.
- [`shared/features.py`](../shared/features.py): load dữ liệu, feature của ML task 1.5 (`n_flags`, 6 decimal fractions, `log_amt_rel_balance`, `velocity_x_risk`), `LinearFE`, folds, `rank01`, `tune_weights`, và hàm ghi/đọc dự đoán.
- [`shared/dl.py`](../shared/dl.py): vòng huấn luyện PyTorch (early stopping theo AUC fold validation), `MLP`, `PeriodicEmbedding` (PLR), `EmbeddingMLP`.
- Mỗi model ghi `oof/<model>.csv` (xác suất OOF trên 699.960 dòng train) và `test/<model>.csv` (trung bình 5 fold), kèm `results/<model>_metrics.json`.

Các notebook nằm trong [`3.ENSEMBLE/src/`](../3.ENSEMBLE/src/) và chạy trên Kaggle. Mỗi notebook dùng output của notebook trước làm input.

## 3. ML trên folds chung (3.1)

Notebook 3.1 chạy lại đúng cấu hình ML task 1.5:

| Model | OOF AUC | AP | Std AUC giữa fold |
|---|---:|---:|---:|
| lgb_nl3 | 0.94774 | 0.5156 | 0.0019 |
| lgb_nl4 | 0.94758 | 0.5139 | 0.0020 |
| xgb | 0.94730 | 0.5140 | 0.0021 |
| lgb_nl8 | 0.94720 | 0.5130 | 0.0018 |
| spline_lr | 0.94469 | 0.5008 | 0.0016 |

Rank blend (tuned, shrink 0.7/0.3) đạt **0.94789**, so với 0.94792 ở task 1.5. Kết quả ML được tái lập trên folds chung.

## 4. DL trên folds chung (3.2–3.3)

| Model | Thay đổi | OOF AUC | TB AUC theo fold | Spearman với lgb_nl3 | Gain khi blend với ML |
|---|---|---:|---:|---:|---:|
| mlp_base | MLP 2.1 chạy 5-fold | 0.94256 | 0.94265 | 0.942 | −0.00001 |
| mlp_fe | + 9 feature của ML | 0.94225 | 0.94232 | 0.943 | 0 |
| mlp_plr | PLR σ=0.1, embedding cho category và cột rời rạc, MLP 256→128, OneCycle | 0.94332 | 0.94465 | 0.958 | 0 |
| mlp_plr_sig1 | σ=1 | 0.94516 | 0.94593 | 0.963 | +0.00011 |
| mlp_plr_sig1_reg | σ=1, dropout 0.3/0.2, wd 1e-2, lr 1e-3 | 0.94578 | 0.94649 | 0.971 | +0.00012 |
| mlp_plr_sig3 | σ=3 | 0.94431 | 0.94588 | 0.940 | +0.00046 |
| mlp_plr_sig3_reg | σ=3 + regularize | 0.94544 | 0.94655 | 0.955 | +0.00049 |
| mlp_plr_sig10_reg | σ=10 + regularize | 0.94355 | 0.94399 | 0.926 | +0.00038 |
| **mlp_plr_sig3_reg_x3** | σ=3 + regularize, trung bình 3 seed | **0.94707** | **0.94739** | 0.967 | **+0.00057** |

Cột "Gain" là mức tăng AUC trung bình theo fold khi blend rank ML với model DL. Trọng số DL được chọn trên 4 fold và chấm trên fold còn lại.

Nhận xét:

- Khi chạy 5-fold, MLP baseline (0.94256) thấp hơn holdout 2.1 (0.94462). Như vậy holdout cũ là một phép chia dễ hơn trung bình.
- MLP one-hot thua cả Spline LR (0.94469). Nút thắt nằm ở **cách biểu diễn cột số**: thêm feature không giúp gì, còn PLR embedding thì giúp.
- σ của PLR kiểm soát mức khác biệt giữa DL và ML. σ lớn hơn cho tương quan với LGBM thấp hơn và gain blend cao hơn, cho tới σ=10, khi AUC riêng giảm quá nhiều. Giá trị tốt nhất là **σ=3**.
- Trung bình 3 seed tăng AUC riêng thêm khoảng 0.0016 và cho gain blend lớn nhất.

## 5. Kết hợp (3.4)

Đánh giá bằng CV lồng: trọng số hoặc meta-model học trên 4 fold, chấm trên fold còn lại.

| Phương án | TB AUC theo fold | Std | AP | Gain so với ML-only | Số fold thắng ML-only |
|---|---:|---:|---:|---:|---:|
| ml_only (5 ML, rank blend) | 0.94784 | 0.00192 | 0.5170 | — | — |
| ml+dl_best (5 ML + sig3_reg_x3) | 0.94839 | 0.00181 | 0.5201 | +0.00055 | 5/5 |
| **ml+dl_all (5 ML + 4 DL PLR)** | **0.94852** | **0.00179** | 0.5203 | **+0.00068** | **5/5** |
| stack_lr (LR trên logit 9 model) | 0.94850 | 0.00179 | 0.5204 | +0.00066 | 5/5 |

Phương án được chọn là **`ml+dl_all`**. Đây là phương án có AUC trung bình cao nhất, thắng ML-only ở cả 5 fold, và có std thấp nhất. Trọng số học lại trên toàn bộ OOF ([final_blend.json](../3.ENSEMBLE/results/final_blend.json)):

| lgb_nl3 | lgb_nl4 | lgb_nl8 | spline_lr | xgb | sig3_reg_x3 | sig1_reg | sig3 | sig10_reg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.191 | 0.181 | 0.076 | 0.033 | 0.100 | 0.186 | 0.033 | 0.098 | 0.102 |

Tổng trọng số của DL là khoảng **0.42**, nên DL đóng góp đáng kể chứ không chỉ là phần phụ.

## 6. Submission

- [`submissions/submission_ensemble.csv`](../submissions/submission_ensemble.csv): phương án `ml+dl_all`.
- [`submissions/submission_ml_only.csv`](../submissions/submission_ml_only.csv): ML-only trên cùng pipeline, dùng để đối chiếu.
- Cả hai file đều có 299.984 dòng, đúng thứ tự `sample_submission.csv`, và giá trị là rank trong [0,1]. ROC-AUC chỉ phụ thuộc thứ tự nên không bị ảnh hưởng.

| File | Public LB | So với ML-only (cùng pipeline) |
|---|---:|---:|
| DL task 2.5 (tham chiếu cũ) | 0.94263 | |
| ML task 1.5 (tham chiếu cũ) | 0.94918 | |
| submission_ml_only.csv | 0.94930 | — |
| **submission_ensemble.csv** | **0.94972** | **+0.00042** |

Trên public LB, ensemble tăng +0.00042 so với ML-only và +0.00054 so với ML task 1.5. Kết quả này cùng chiều với gain trên CV (+0.00068). Điểm LB được ghi lại sau khi đã chọn phương án, không dùng để chọn.

## 7. Giới hạn

- Gain vẫn nhỏ: +0.00068 trên CV và +0.00042 trên public LB. Public LB chỉ là một phần của test, nên điểm private có thể khác.
- Folds chung được dùng cả để early stopping cho từng model lẫn để chọn trọng số blend. CV lồng chỉ loại trừ phần lạc quan của bước blend.
- Các thí nghiệm DL (σ, regularization) được chọn dựa trên cùng OOF đó. Có thể có một phần lạc quan nhỏ do chọn cấu hình.
- Chưa thử: TabNet trên folds chung, nhiều seed hơn cho các model DL khác, dùng GBDT OOF làm input cho MLP, và pseudo-label tính trong từng fold.
