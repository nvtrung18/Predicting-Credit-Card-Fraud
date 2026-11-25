# Phát hiện gian lận thẻ tín dụng bằng Deep Learning

Tài liệu của nhánh **DL**, trình bày quy trình đã thực hiện từ **2.1 đến 2.5** cho bài toán Credit Card Fraud trong Kaggle Playground Series S6E11: xây dựng mô hình, thử nghiệm cách xử lý dữ liệu và mất cân bằng lớp, đánh giá trên validation, chọn mô hình rồi tạo submission.

Các cấu hình và số liệu bên dưới được đối chiếu với mã notebook và các file kết quả đã lưu trong repository. Kết quả validation và thời gian chạy là kết quả của những lần chạy được lưu, không phải kết quả huấn luyện lại khi viết tài liệu này.

## 1. Bài toán và dữ liệu

Đây là bài toán phân loại nhị phân trên dữ liệu dạng bảng. Với mỗi giao dịch, mô hình tạo một giá trị dự đoán `is_fraud` trong khoảng `[0, 1]`, biểu diễn xác suất gian lận:

- Nhãn `0`: giao dịch không gian lận.
- Nhãn `1`: giao dịch gian lận.
- `id`: định danh để đối chiếu dữ liệu và xuất kết quả; không được đưa vào đầu vào mô hình.

### Dữ liệu sử dụng

| Dữ liệu | Số dòng | Vai trò |
|---|---:|---|
| [train.csv](data/raw/train.csv) | 699.960 | Dữ liệu có nhãn, dùng để huấn luyện và tạo tập validation |
| [test.csv](data/raw/test.csv) | 299.984 | Dữ liệu cần dự đoán, không có cột nhãn `is_fraud` |
| [sample_submission.csv](data/raw/sample_submission.csv) | 299.984 | Mẫu quy định hai cột `id,is_fraud` và thứ tự ID khi xuất kết quả |

Sau khi bỏ `id` và nhãn, dữ liệu có **24 đặc trưng gốc**:

| Nhóm | Số cột | Các cột |
|---|---:|---|
| Giá trị số | 13 | `amount_usd`, `hours_since_last_txn`, `txn_count_last_24h`, `distance_from_home_km`, `card_age_months`, `customer_age`, `account_balance_usd`, `cvv_retry_count`, `velocity_score`, `time_of_day_hour`, `day_of_week`, `merchant_risk_score`, `prior_disputes` |
| Boolean | 6 | `is_foreign_transaction`, `is_new_merchant`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch`, `is_ai_generated_scam_attempt` |
| Phân loại | 5 | `merchant_category`, `card_type`, `auth_method`, `channel`, `device_type` |

Các cột boolean được xử lý như giá trị số; nhánh tiền xử lý số vì vậy gồm **19 cột số/boolean**. Các notebook dùng đặc trưng có sẵn trong dữ liệu để xây dựng những mô hình dưới đây.

### Mất cân bằng lớp

Theo [metadata của lần huấn luyện cuối](2.DL/results/task_2_5_final/model_metadata.json), toàn bộ train có **11.689 giao dịch gian lận**, tương đương khoảng **1,67%**. Phần còn lại thuộc lớp không gian lận. Vì vậy, chỉ nhìn accuracy có thể che khuất việc bỏ sót lớp gian lận; quy trình sử dụng ROC-AUC và Average Precision để đánh giá dự đoán xác suất.

## 2. Thiết kế đánh giá chung

### Chia train/validation một lần và dùng lại

Ở bước 2.1, dữ liệu có nhãn được chia bằng `train_test_split` với:

- `test_size=0.20`: dành 20% để validation.
- `random_state=9`: cố định seed cho phép chia.
- `stratify=y`: giữ tỷ lệ hai lớp gần nhau giữa train và validation.

| Tập | Số dòng | Số giao dịch gian lận | Số giao dịch không gian lận |
|---|---:|---:|---:|
| Train của thí nghiệm | 559.968 | 9.351 | 550.617 |
| Validation | 139.992 | 2.338 | 137.654 |

Membership của từng ID được lưu tại [split_ids_seed9.csv](shared/splits/split_ids_seed9.csv). Bước 2.2 và 2.3 đọc lại file này rồi ghép theo `id`, thay vì tự chia lại. Bước 2.4 căn chỉnh dự đoán theo ID và kiểm tra nhãn trước khi so sánh. Như vậy, bốn ứng viên được đánh giá trên cùng các giao dịch validation.

### Ngăn rò rỉ dữ liệu

- Trong các thí nghiệm 2.1–2.3, scaler, encoder, bảng ánh xạ category và median nếu có chỉ được fit/tính trên phần train. Validation chỉ dùng phép biến đổi đã học từ train.
- Các cột `id`, `is_fraud` và cột membership `split` nếu có đều bị loại khỏi đặc trưng.
- Quyết định chọn mô hình ở 2.4 dùng dự đoán và nhãn validation; không dùng nhãn test hoặc điểm leaderboard.
- Sau khi chốt mô hình và số epoch, bước 2.5 mới fit lại tiền xử lý trên toàn bộ dữ liệu có nhãn. Test chỉ được transform và dự đoán.

### Các chỉ số sử dụng

| Chỉ số | Cách dùng trong dự án |
|---|---|
| **ROC-AUC** | Chỉ số chính để chọn checkpoint và xếp hạng ứng viên; tính trực tiếp từ nhãn và xác suất bằng `roc_auc_score` |
| **Average Precision (AP)** | Chỉ số bổ sung để xem chất lượng precision/recall của lớp gian lận; tính bằng `average_precision_score` và dùng để phân xử khi ROC-AUC bằng nhau |
| **Training loss** | Theo dõi quá trình tối ưu theo epoch; loss có trọng số và loss không trọng số không được xem là cùng một thang đo để chọn mô hình |
| **Precision, recall và số lỗi tại ngưỡng 0,5** | Dùng để diễn giải lỗi của mô hình được chọn trong bước 2.4 |

AP trong tài liệu là kết quả của `average_precision_score`, không phải một giá trị diện tích đường precision-recall được tính riêng bằng quy tắc hình thang. Ngưỡng `0,5` chỉ phục vụ phân tích lỗi; ROC-AUC và submission sử dụng xác suất liên tục.

## 3. Bước 2.1 — Xây dựng MLP baseline

Nguồn: [notebook 2.1](2.DL/src/credit-card-fraud-2-1-mlp-baseline.ipynb), [metrics](2.DL/results/task_2_1_mlp/metrics.json), [báo cáo 2.1](reports/task_2_1_mlp.md).

### Mục tiêu

Xây dựng mô hình Deep Learning ban đầu làm mốc so sánh. MLP nhận vector đặc trưng sau tiền xử lý và học quan hệ phi tuyến để dự đoán xác suất gian lận.

### Tiền xử lý đã áp dụng

1. Loại `id` và `is_fraud` khỏi bảng đặc trưng.
2. Đổi các cột boolean thành `int8`, biểu diễn bằng `0/1`.
3. Dùng `StandardScaler` cho 19 cột số/boolean để đưa các đặc trưng về thang đo chuẩn hóa theo thống kê train.
4. Dùng `OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)` cho 5 cột phân loại. Category chưa xuất hiện khi fit được xử lý bằng cơ chế bỏ qua của encoder.
5. Ghép hai nhánh bằng `ColumnTransformer`, thu được **54 đặc trưng đầu vào**, chuyển ma trận sang `float32` để đưa vào PyTorch.

Notebook baseline không thêm bước điền missing riêng; bước điền median không được áp dụng đồng loạt cho mọi mô hình trong phạm vi này.

### Kiến trúc mạng

```text
54 đặc trưng
  → Linear(54, 128) → ReLU → Dropout(0.20)
  → Linear(128, 64) → ReLU → Dropout(0.10)
  → Linear(64, 1)
  → logit → sigmoid khi dự đoán xác suất
```

ReLU tạo tính phi tuyến; dropout ngẫu nhiên bỏ một phần activation trong lúc train để hạn chế phụ thuộc quá mức vào các neuron. Đầu ra của mạng là **logit**. Khi huấn luyện, logit được truyền trực tiếp vào `BCEWithLogitsLoss`; khi dự đoán mới áp dụng sigmoid.

### Cấu hình huấn luyện

| Thành phần | Cấu hình |
|---|---|
| Framework | PyTorch |
| Loss | `BCEWithLogitsLoss`, không đặt trọng số lớp |
| Optimizer | `AdamW` |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Batch size | `4096`, shuffle dữ liệu train |
| Giới hạn epoch | `12` |
| Early stopping | Patience `3`, theo validation ROC-AUC |
| Điều kiện cập nhật checkpoint | ROC-AUC tăng hơn giá trị tốt nhất trước đó trên `1e-6` |
| Gradient clipping | `clip_grad_norm_`, `max_norm=1.0` |
| Seed | `9` cho Python, NumPy, PyTorch và generator của DataLoader |
| GPU ghi trong artifact | Tesla T4 |

Notebook đặt cấu hình cuDNN deterministic và tắt benchmark khi có CUDA. Những thiết lập này hỗ trợ khả năng tái lập; không có nghĩa mọi môi trường hoặc phiên bản thư viện đều cho kết quả giống tuyệt đối.

### Kết quả đã lưu

- Chạy **12 epoch**, checkpoint được chọn ở **epoch 9**.
- Validation ROC-AUC: **0.944622**.
- Validation AP: **0.484340**.
- Sau khi train, mô hình được nạp lại trọng số tốt nhất trước khi lưu dự đoán validation.

Đầu ra gồm [checkpoint](2.DL/checkpoints/task_2_1_mlp/best_model.pt), [preprocessor đã fit](2.DL/checkpoints/task_2_1_mlp/preprocessor.joblib), [lịch sử train](2.DL/results/task_2_1_mlp/training_history.csv), [dự đoán validation](2.DL/results/task_2_1_mlp/validation_predictions_seed9.csv) và [learning curves](2.DL/results/task_2_1_mlp/learning_curves.png). Bước này tạo mốc đánh giá, chưa tạo submission cho test.

## 4. Bước 2.2 — Thử nghiệm TabNet baseline

Nguồn: [notebook 2.2](2.DL/src/credit-card-fraud-2-2-tabnet-baseline.ipynb), [metrics](2.DL/results/task_2_2_tabnet/metrics.json).

### Mục tiêu và cách biểu diễn dữ liệu

Thử một kiến trúc dành cho dữ liệu bảng với cơ chế attention chọn đặc trưng qua các decision step, để so sánh với MLP trên cùng holdout. Notebook dùng `TabNetClassifier` từ `pytorch-tabnet`; khi chưa có package, code cài phiên bản `4.1.0`.

Tiền xử lý của TabNet khác nhánh one-hot:

1. Đọc lại split ID của 2.1 và kiểm tra membership.
2. Chuyển cột số sang dạng numeric, đưa giá trị vô hạn về missing và điền bằng median tính trên train.
3. Chuẩn hóa nhóm số/boolean bằng `StandardScaler` fit trên train.
4. Với category, thay missing bằng `__MISSING__`, xây mapping từ train với các mã bắt đầu ở `1`; mã `0` dành cho category lạ ở validation.
5. Ghép 19 cột số/boolean với 5 cột category đã mã hóa, rồi truyền `cat_idxs`, `cat_dims` cho TabNet để xử lý category bằng embedding nội bộ.

### Cấu hình đã áp dụng

| Thành phần | Giá trị |
|---|---|
| `n_d`, `n_a` | `16`, `16` |
| `n_steps` | `4` |
| `gamma` | `1.3` |
| `lambda_sparse` | `1e-4` |
| `mask_type` | `entmax` |
| `cat_emb_dim` | `1` |
| Kích thước các bảng category, gồm mã unknown | `[13, 6, 6, 6, 9]` |
| Optimizer | `Adam`, learning rate `0.02` |
| Batch size / virtual batch size | `4096` / `512` |
| Giới hạn epoch / patience | `40` / `5` |
| Metric validation khi fit | `auc` |
| Seed | `9` |

`n_d` và `n_a` quy định kích thước biểu diễn cho nhánh quyết định và attention; `n_steps` là số bước xử lý. `lambda_sparse` đặt mức phạt sparsity cho attention mask. Virtual batch size là kích thước nhóm dùng cho Ghost Batch Normalization bên trong TabNet.

### Kết quả đã lưu

- Huấn luyện **19 epoch**, epoch tốt nhất là **epoch thứ 14**.
- Trong metrics gốc, `best_epoch=13` là chỉ số bắt đầu từ `0`; bước 2.4 chuyển sang cách đếm bắt đầu từ `1`.
- Validation ROC-AUC: **0.943122**; AP: **0.479909**.
- Thời gian huấn luyện ghi trong artifact: khoảng **228,53 giây** trên Tesla T4.

TabNet không vượt MLP baseline trong lần so sánh này. Kết luận chỉ áp dụng cho cấu hình và holdout đã chạy.

Các file được lưu: [mô hình TabNet](2.DL/checkpoints/task_2_2_tabnet/tabnet_seed9.zip), [preprocessor](2.DL/checkpoints/task_2_2_tabnet/preprocessor.joblib), [lịch sử train](2.DL/results/task_2_2_tabnet/training_history.csv) và [dự đoán validation](2.DL/results/task_2_2_tabnet/validation_predictions_tabnet_seed9.csv).

## 5. Bước 2.3 — Thử nghiệm trọng số lớp và categorical embedding cho MLP

Nguồn: [notebook 2.3](2.DL/src/credit-card-fraud-2-3-mlp-imbalance-experiments.ipynb), [metrics tổng hợp](2.DL/results/task_2_3_mlp/metrics.json), [bảng thí nghiệm](2.DL/results/task_2_3_mlp/experiment_table.csv).

Bước này kiểm tra hai thay đổi riêng biệt so với MLP baseline. Cả hai đều dùng lại membership seed 9, kiến trúc phần dense `128 → 64 → 1`, ReLU, dropout `0.20/0.10`, AdamW `lr=1e-3`, `weight_decay=1e-4`, batch `4096`, gradient clipping `1.0`, tối đa 12 epoch và patience 3 theo validation ROC-AUC.

### Thí nghiệm A — MLP với trọng số cho lớp gian lận

Giữ cách mã hóa one-hot và kiến trúc của baseline, thay loss bằng `BCEWithLogitsLoss(pos_weight=...)`. Trọng số được tính từ **train của thí nghiệm**, không lấy thống kê validation/test:

```text
pos_weight = số mẫu không gian lận / số mẫu gian lận
           = 550617 / 9351
           ≈ 58.883221
```

Trọng số này tăng đóng góp của mẫu có nhãn gian lận vào hàm loss. Mục đích là thử cách giúp mô hình chú ý hơn đến lớp hiếm; dữ liệu không được nhân bản hoặc lấy mẫu lại bằng SMOTE trong thí nghiệm này.

Kết quả: **ROC-AUC 0.944596**, **AP 0.480716**, checkpoint tốt nhất tại **epoch 5**, chạy tổng cộng **8 epoch**. Trọng số lớp chưa cải thiện hai chỉ số so với baseline trong lần chạy này. Do đã đổi loss, điểm dự đoán của mô hình weighted cũng không được mặc định là đã hiệu chuẩn xác suất.

### Thí nghiệm B — MLP với categorical embedding

Thay one-hot bằng các vector embedding học được cho 5 cột phân loại:

- Nhóm số/boolean: chuyển numeric, xử lý missing bằng median train và chuẩn hóa bằng scaler train.
- Nhóm category: dùng `__MISSING__`, mapping từ train với mã `1..K`, và mã `0` cho category lạ ở validation.
- Mỗi cột category có một `nn.Embedding` riêng. Chiều embedding được tính theo `min(16, max(2, ceil(sqrt(K + 1))))`, với `K` là số category quan sát trong train.
- Ghép vector embedding của các cột với vector số đã chuẩn hóa, rồi đưa vào mạng dense.
- Dùng `BCEWithLogitsLoss` **không có trọng số lớp**, nhằm đánh giá thay đổi về cách biểu diễn category.

```text
19 giá trị số/boolean đã chuẩn hóa ──────────────┐
5 cột category → 5 bảng embedding → ghép vector ├→ Dense 128 → Dense 64 → 1 logit
                                               ┘
```

Kết quả: **ROC-AUC 0.944339**, **AP 0.482583**, checkpoint tốt nhất tại **epoch 10**, chạy tổng cộng **12 epoch**. Thí nghiệm chạm giới hạn epoch; chưa vượt baseline. Đây là một thử nghiệm biểu diễn category, không phải mô hình kết hợp embedding và class weighting.

Checkpoint và preprocessing được lưu tại [thư mục checkpoint 2.3](2.DL/checkpoints/task_2_3_mlp/); metric, dự đoán, lịch sử và biểu đồ của từng ứng viên nằm tại [kết quả 2.3](2.DL/results/task_2_3_mlp/).

## 6. Bước 2.4 — So sánh validation, phân tích lỗi và chọn mô hình

Nguồn: [notebook 2.4](2.DL/src/credit-card-fraud-2-4-validation-analysis.ipynb), [bảng so sánh](2.DL/results/task_2_4_validation_analysis/comparison.csv), [quyết định chọn mô hình](2.DL/results/task_2_4_validation_analysis/selected_model.json).

### Kiểm tra trước khi so sánh

Notebook nạp bốn bộ dự đoán đã lưu, kiểm tra các cột `id`, `is_fraud`, `fraud_probability`, ID không trùng, cùng tập ID, cùng nhãn sau khi căn chỉnh và xác suất hữu hạn trong `[0,1]`. Sau đó mới tính lại ROC-AUC và AP từ dự đoán validation.

### Kết quả trên cùng holdout seed 9

| Mô hình | Bước | Validation ROC-AUC | Validation AP | Epoch tốt nhất |
|---|---|---:|---:|---:|
| **MLP baseline** | **2.1** | **0.944622** | **0.484340** | **9** |
| MLP với trọng số lớp | 2.3 | 0.944596 | 0.480716 | 5 |
| MLP với categorical embedding | 2.3 | 0.944339 | 0.482583 | 10 |
| TabNet baseline | 2.2 | 0.943122 | 0.479909 | 14 |

Quy tắc chọn là **ROC-AUC lớn nhất, AP phân xử nếu ROC-AUC bằng nhau**. MLP baseline được chọn vì có cả ROC-AUC và AP cao nhất trong bốn ứng viên. Chênh lệch ROC-AUC với MLP weighted chỉ khoảng `0.000026`; kết quả này chưa chứng minh khác biệt có ý nghĩa thống kê hoặc thứ hạng ổn định trên mọi phép chia dữ liệu.

### Phân tích lỗi ở ngưỡng 0,5

Với MLP baseline, gán dự đoán gian lận khi xác suất `>= 0.5`:

| Nhãn thực tế | Dự đoán không gian lận | Dự đoán gian lận |
|---|---:|---:|
| Không gian lận | TN = 137.433 | FP = 221 |
| Gian lận | FN = 1.677 | TP = 661 |

- Precision: khoảng **74,94%**.
- Recall: khoảng **28,27%**.
- Diễn giải: trong 2.338 giao dịch gian lận ở validation, ngưỡng này phát hiện 661 và bỏ sót 1.677. Vì vậy ROC-AUC cao không đồng nghĩa đã có recall cao ở ngưỡng mặc định.

Ngưỡng này chưa được tối ưu theo chi phí điều tra hoặc tổn thất gian lận. Nó được dùng để đọc lỗi, không dùng để biến dự đoán submission thành nhãn `0/1`.

### Đầu ra phân tích

- [Learning curves so sánh](2.DL/results/task_2_4_validation_analysis/learning_curves_comparison.png): loss và ROC-AUC theo epoch.
- [ROC curves và phân phối xác suất](2.DL/results/task_2_4_validation_analysis/roc_and_probability_distributions.png).
- [Tóm tắt phân tích](2.DL/results/task_2_4_validation_analysis/analysis_summary.md).
- [selected_model.json](2.DL/results/task_2_4_validation_analysis/selected_model.json): mô hình được chọn, metric validation, epoch tốt nhất, số lỗi và giới hạn của đánh giá; đây là đầu vào cho 2.5.

Thời gian train của baseline không có trong bảng metrics được dùng để so sánh, nên được để trống. Quy trình chọn mô hình không suy diễn thời gian còn thiếu và không dựa vào leaderboard.

## 7. Bước 2.5 — Huấn luyện lại mô hình được chọn và tạo submission

Nguồn: [notebook 2.5](2.DL/src/credit-card-fraud-2-5-final-dl-submission.ipynb), [metadata lần chạy cuối](2.DL/results/task_2_5_final/model_metadata.json), [báo cáo được tạo khi chạy](2.DL/results/task_2_5_final/task_2_5_report.md).

### Huấn luyện lại trên toàn bộ train

1. Đọc mô hình và số epoch từ quyết định của 2.4: **MLP baseline, 9 epoch**.
2. Nạp toàn bộ **699.960 dòng có nhãn** và loại `id`, `is_fraud` khỏi đặc trưng.
3. Đổi boolean sang `0/1`; xử lý missing category bằng `__MISSING__`; chuyển nhóm số sang numeric, thay giá trị vô hạn và điền bằng median của toàn bộ train.
4. Fit mới `StandardScaler` và `OneHotEncoder` trên toàn bộ train; dùng cùng preprocessor để transform test.
5. Khởi tạo MLP mới với hai lớp ẩn `128`, `64`, ReLU và dropout `0.20/0.10`; train với AdamW `lr=1e-3`, `weight_decay=1e-4`, batch `4096`, gradient clipping `1.0` và loss BCE không trọng số.
6. Chạy đúng **9 epoch** đã chốt, lưu mô hình và preprocessor phục vụ inference.

Đây là **refit từ đầu**, không phải tiếp tục train checkpoint của 2.1. Vì toàn bộ train được dùng để học, lần refit không giữ validation để early stopping; số epoch lấy từ thí nghiệm trước đó. Các metric validation trong metadata là metric dùng để chọn mô hình, không phải metric được đo mới sau refit.

Metadata có ghi tỷ lệ âm/dương của toàn bộ train trong trường `full_train_pos_weight`, nhưng mô hình cuối là baseline nên **không sử dụng trọng số này trong loss**.

Lần chạy được lưu dùng **Tesla T4**, thời gian train khoảng **53,24 giây**.

### Dự đoán và kiểm tra file đầu ra

Mô hình chuyển sang chế độ evaluation, dùng `torch.inference_mode()`, dự đoán test theo batch tối đa **8.192 dòng**, rồi áp dụng sigmoid để lấy xác suất. Kết quả được ánh xạ theo `id` vào mẫu submission để giữ đúng thứ tự.

Các kiểm tra trong notebook gồm:

- ID train/test không trùng trong từng tập; mẫu submission có đúng schema và không trùng ID.
- Tập ID của mẫu submission khớp với test.
- Số xác suất dự đoán bằng số dòng test; số dòng đầu ra khớp mẫu.
- Đầu ra có đúng hai cột theo thứ tự `id,is_fraud`.
- Không thiếu ID hoặc xác suất; xác suất hữu hạn và nằm trong `[0,1]`.

### Kết quả cuối đã lưu

| Nội dung | Kết quả |
|---|---|
| Mô hình cuối | MLP baseline |
| Số epoch refit | 9 |
| Dữ liệu refit | 699.960 dòng |
| Số dự đoán test | 299.984 |
| Schema submission | `id,is_fraud` |
| Khoảng xác suất ghi trong metadata | Khoảng `4.8269e-08` đến `0.98970747` |
| Validation dùng để chọn mô hình | ROC-AUC `0.944622`, AP `0.484340` |

File bàn giao: [submission.csv](2.DL/results/task_2_5_final/submission.csv), [model metadata](2.DL/results/task_2_5_final/model_metadata.json), [lịch sử refit](2.DL/results/task_2_5_final/training_history.csv), [checkpoint cuối](2.DL/checkpoints/task_2_5_final/best_model.pt) và [preprocessor cuối](2.DL/checkpoints/task_2_5_final/preprocessor.joblib).

**Điểm leaderboard được người thực hiện báo lại: `0.94263`.** Đây là thông tin đã được ghi nhận trong dự án, tách biệt với metric validation. Notebook không tự gửi submission; metadata tại thời điểm tạo file ghi `competition_submission_uploaded=false`. Các artifact này xác nhận việc tạo file, không tự xác nhận điểm leaderboard sau đó.

## 8. Chạy lại quy trình 2.1–2.5 trên Kaggle

Các notebook sử dụng đường dẫn dữ liệu `/kaggle/input/competitions/playground-series-s6e11` và ghi kết quả vào `/kaggle/working`. Có dữ liệu trong repository chưa đủ để chạy nguyên notebook trên máy cá nhân mà không điều chỉnh đường dẫn và môi trường.

1. Import notebook cần chạy vào Kaggle, gắn dữ liệu cuộc thi, bật GPU và kiểm tra các thư viện mà notebook import: NumPy, pandas, scikit-learn, PyTorch, matplotlib, joblib; TabNet cần thêm `pytorch-tabnet`.
2. Chạy theo bảng phụ thuộc dưới đây; lưu output của mỗi bước để gắn vào bước tiếp theo.
3. Bảo đảm file mà notebook tìm bằng tên chỉ xuất hiện một lần trong input. Tránh gắn nhiều bản output chứa trùng split, metrics hoặc prediction cần tìm.
4. Sau 2.5, tải file submission, kiểm tra lại định dạng và thực hiện thao tác nộp trên Kaggle khi cần. Notebook chỉ tạo file.

| Bước | Notebook | Output bước trước cần gắn làm input |
|---|---|---|
| 2.1 | [MLP baseline](2.DL/src/credit-card-fraud-2-1-mlp-baseline.ipynb) | Không có; dùng dữ liệu cuộc thi |
| 2.2 | [TabNet baseline](2.DL/src/credit-card-fraud-2-2-tabnet-baseline.ipynb) | 2.1, để lấy split ID |
| 2.3 | [MLP imbalance experiments](2.DL/src/credit-card-fraud-2-3-mlp-imbalance-experiments.ipynb) | 2.1, để lấy split, metrics và dự đoán baseline |
| 2.4 | [Validation analysis](2.DL/src/credit-card-fraud-2-4-validation-analysis.ipynb) | 2.1, 2.2 và 2.3 |
| 2.5 | [Final DL submission](2.DL/src/credit-card-fraud-2-5-final-dl-submission.ipynb) | 2.4, để lấy quyết định chọn mô hình |

## 9. Giới hạn của kết quả

- So sánh dựa trên **một holdout cố định seed 9**; chưa chạy cross-validation trong quy trình 2.1–2.5.
- Kết quả cho biết cấu hình nào tốt hơn trong bốn lần thử đã lưu; không chứng minh MLP luôn tốt hơn TabNet, hoặc class weighting/embedding luôn kém hơn one-hot.
- Việc chọn mô hình và checkpoint cùng dựa vào holdout này; chưa có thêm một tập đánh giá độc lập có nhãn để kiểm chứng sau chọn.
- Chưa có bước hiệu chuẩn xác suất hoặc chọn ngưỡng theo chi phí nghiệp vụ; chưa có bằng chứng triển khai vận hành thực tế.
- Thời gian và metric phụ thuộc dữ liệu, phần cứng, seed và môi trường thư viện. Những số liệu đã lưu không phải cam kết kết quả của mọi lần chạy lại.

Các nguồn kết quả chính để kiểm tra lại: [metrics MLP](2.DL/results/task_2_1_mlp/metrics.json), [metrics TabNet](2.DL/results/task_2_2_tabnet/metrics.json), [thí nghiệm MLP](2.DL/results/task_2_3_mlp/experiment_table.csv), [bảng so sánh validation](2.DL/results/task_2_4_validation_analysis/comparison.csv), [quyết định chọn mô hình](2.DL/results/task_2_4_validation_analysis/selected_model.json) và [metadata lần refit](2.DL/results/task_2_5_final/model_metadata.json).

## 10. Kết hợp ML + DL (3.0–3.4)

Nhánh `ML_DL_ensemble` gộp code ML của nhánh `ML_Hieu` (trong [1.ML/src/](1.ML/src/)) với nhánh DL. Mọi model được đánh giá trên cùng 5 fold ([folds_5_seed9.csv](shared/folds/folds_5_seed9.csv)), và phương án kết hợp được chọn bằng CV lồng. Chi tiết: [báo cáo kết hợp ML + DL](reports/task_3_ensemble.md).

| Notebook (Kaggle) | Nội dung | Kết quả chính |
|---|---|---|
| [3.0](3.ENSEMBLE/src/3-0-shared-folds.ipynb) | Tạo folds chung | Trùng từng dòng với splitter ML 1.3–1.5 |
| [3.1](3.ENSEMBLE/src/3-1-ml-models-shared-folds.ipynb) | 5 model ML task 1.5 | Blend OOF 0.94789 (task 1.5: 0.94792) |
| [3.2](3.ENSEMBLE/src/3-2-mlp-baseline-5fold.ipynb) | MLP 2.1 chạy 5-fold | OOF 0.94256 |
| [3.3](3.ENSEMBLE/src/3-3-dl-upgrades.ipynb) | MLP + PLR numeric embedding | `mlp_plr_sig3_reg_x3` OOF 0.94707 |
| [3.4](3.ENSEMBLE/src/3-4-blend-and-submission.ipynb) | Blend / stacking + submission | `ml+dl_all`: 0.94852 so với ML-only 0.94784 (thắng 5/5 fold) |

**Public LB:** ensemble ML + DL **0.94972**, ML-only trên cùng pipeline 0.94930, ML task 1.5 0.94918, DL task 2.5 0.94263.

Code dùng chung: [shared/features.py](shared/features.py), [shared/dl.py](shared/dl.py). Metrics: [3.ENSEMBLE/results/](3.ENSEMBLE/results/). Submission: [submissions/submission_ensemble.csv](submissions/submission_ensemble.csv).
