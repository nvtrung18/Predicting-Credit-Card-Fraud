# Kết hợp ML + DL (3.0–3.6)

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

## 7. Bản v2 (3.3 vòng 4, 3.5, 3.6)

Mục tiêu là tăng thêm gain so với v1 (`ml+dl_all`, CV 0.94852, LB 0.94972). Mọi thử nghiệm vẫn chạy trên cùng 5 fold, và v2 được chọn bằng CV lồng.

### 7.1 Model ML mới (3.5)

Cột "Gain" là mức tăng khi blend ensemble v1 với từng model, với alpha chọn trên 4 fold và chấm trên fold còn lại ([ml_new_vs_v1_3_5.csv](../3.ENSEMBLE/results/ml_new_vs_v1_3_5.csv)).

| Model | OOF AUC | Gain so với v1 | Nhận xét |
|---|---:|---:|---|
| `lgb_nl3_cnt` (thêm 13 feature đếm tần suất trên train+test) | **0.94787** | **+0.00009** | Cao hơn `lgb_nl3` 0.00013; alpha 0.25–0.30 ở cả 5 fold |
| `lgb_nl3_x3` (3 seed) | 0.94784 | 0 | Spearman với `lgb_nl3` = 0.9987, đổi seed gần như không đổi gì |
| `lgb_nl4_x3` (3 seed) | 0.94782 | ~0 | |
| `catboost_d4` | 0.94662 | ~0 | Yếu hơn LGBM khoảng 0.001 |
| `catboost_d6` | 0.94613 | ~0 | |

Feature đếm tần suất là thông tin mới với các model ML trước đó. Seed bagging và CatBoost không thêm được thông tin.

### 7.2 Model DL mới (3.3 vòng 4)

Cột "Gain" là mức tăng khi blend 5 model ML (v1) với từng model DL, chấm bằng CV lồng ([dl_vs_ml_3_3.csv](../3.ENSEMBLE/results/dl_vs_ml_3_3.csv)).

| Model | OOF AUC | Spearman với lgb_nl3 | Gain so với ML-only |
|---|---:|---:|---:|
| `mlp_plr_sig3_reg_x5` (5 seed) | **0.94750** | 0.971 | +0.00060 |
| `mlp_plr_sig3_x3` | 0.94711 | 0.965 | +0.00054 |
| `mlp_plr_sig10_reg_x3` | 0.94602 | 0.944 | +0.00061 |
| `mlp_plr_sig2_reg` | 0.94595 | 0.960 | +0.00041 |
| `mlp_plr_sig5_reg` | 0.94548 | 0.944 | **+0.00064** |
| `resnet_plr_sig3` (ResNet head) | 0.94281 | 0.904 | +0.00025 |

- Với 5 seed, `mlp_plr_sig3_reg_x5` gần ngang LGBM (0.94750 so với 0.94774).
- σ=5 có AUC riêng thấp hơn σ=3, nhưng khác LGBM nhiều hơn nên cho gain blend cao nhất.
- ResNet head khác LGBM nhiều nhất (Spearman 0.904), nhưng AUC riêng thấp nên gain nhỏ.

### 7.3 Blend v2 (3.6)

Các ứng viên gồm 15 model: 9 model của v1 và 11 model mới, trong đó bản nhiều seed thay cho bản ít seed cùng cấu hình ([blend_comparison_v2.csv](../3.ENSEMBLE/results/blend_comparison_v2.csv)). Một phương án chỉ thay `v1` khi có AUC trung bình theo fold cao hơn và thắng `v1` ở ít nhất 4/5 fold.

| Phương án | TB AUC theo fold | AP | Gain so với v1 | Số fold thắng v1 |
|---|---:|---:|---:|---:|
| v1 (tái lập 3.4) | 0.94852 | 0.5203 | — | — |
| rw_all (rank toàn cục) | 0.94875 | 0.5217 | +0.00023 | 5/5 |
| **rw_all_foldrank (rank trong từng fold)** | **0.94877** | 0.5217 | **+0.00024** | **5/5** |
| stack_lr_all | 0.94876 | 0.5221 | +0.00023 | 4/5 |
| hill_all_foldrank (Caruana) | 0.94866 | 0.5213 | +0.00014 | 4/5 |

Phương án được chọn là **`rw_all_foldrank`**. Ba phương án đứng đầu chỉ chênh nhau khoảng 0.00002, nên gain này không phụ thuộc vào cách blend.

Trọng số học lại trên toàn bộ OOF ([final_blend_v2.json](../3.ENSEMBLE/results/final_blend_v2.json)), làm tròn:

| Nhóm | Model và trọng số |
|---|---|
| ML (≈ 0.55) | `lgb_nl3_cnt` 0.260, `xgb` 0.062, `lgb_nl3_x3` 0.060, `catboost_d4` 0.057, `lgb_nl4_x3` 0.046, `lgb_nl8` / `spline_lr` / `catboost_d6` 0.020 mỗi model |
| DL (≈ 0.45) | `mlp_plr_sig5_reg` 0.151, `mlp_plr_sig3_reg_x5` 0.147, `mlp_plr_sig10_reg_x3` 0.065, `resnet_plr_sig3` 0.029, `mlp_plr_sig3_x3` 0.023, `sig1_reg` / `sig2_reg` 0.020 mỗi model |

Mức 0.020 là sàn do bước shrink 0.3 tạo ra, nghĩa là các model đó gần như không được chọn.

### 7.4 Public LB

| File | Public LB | So với bản trước |
|---|---:|---:|
| submission_ml_only.csv | 0.94930 | |
| submission_ensemble.csv (v1) | 0.94972 | +0.00042 so với ML-only |
| **[submission_v2.csv](../submissions/submission_v2.csv)** | **0.94988** | **+0.00016 so với v1** |

LB tăng cùng chiều với CV (+0.00024). Tính từ ML task 1.5 (0.94918), toàn bộ pipeline kết hợp tăng +0.00070 trên public LB. Điểm LB vẫn chỉ được ghi lại sau khi đã chọn phương án.

## 8. Giới hạn

- Gain vẫn nhỏ: v1 +0.00068 trên CV và +0.00042 trên public LB; v2 thêm +0.00024 trên CV và +0.00016 trên public LB. Public LB chỉ là một phần của test, nên điểm private có thể khác.
- Folds chung được dùng cả để early stopping cho từng model lẫn để chọn trọng số blend. CV lồng chỉ loại trừ phần lạc quan của bước blend.
- Các thí nghiệm DL (σ, regularization) được chọn dựa trên cùng OOF đó. Có thể có một phần lạc quan nhỏ do chọn cấu hình.
- v2 chọn trong 15 model và nhiều phương án blend hơn v1, nên phần lạc quan do chọn cấu hình có thể lớn hơn một chút.
- Đã thử nhưng không giúp: seed bagging cho LGBM nông, CatBoost (CPU, depth 4/6). Bản CatBoost chạy GPU chưa được chạy.
- Chưa thử: TabNet trên folds chung, nhiều seed hơn cho `mlp_plr_sig5_reg` và `sig10_reg`, thêm feature đếm tần suất cho DL và các GBDT khác, dùng GBDT OOF làm input cho MLP, và pseudo-label tính trong từng fold.
