# Kiến thức nền tảng ML/DL cho bài toán Credit Card Fraud

Tài liệu này giải thích các khái niệm cần thiết để hiểu nhánh Deep Learning của dự án: dữ liệu, cách chia tập, tiền xử lý, quá trình học của mạng, các thuật toán đã thử, metric Kaggle và cách tiếp tục cải thiện kết quả.

## 1. Bài toán đang giải quyết

Mỗi dòng dữ liệu mô tả một giao dịch thẻ. Mô hình nhận các đặc trưng của giao dịch và dự đoán xác suất giao dịch đó là gian lận.

- **Đặc trưng (features, X):** thông tin đầu vào của giao dịch, ví dụ giá trị giao dịch, thời gian, thiết bị, kênh thanh toán.
- **Nhãn (target, y):** cột `is_fraud`, với 1 là gian lận và 0 là bình thường.
- **Đầu ra của mô hình:** một điểm số hoặc xác suất trong khoảng 0 đến 1 cho mỗi giao dịch.
- **Loại bài toán:** phân loại nhị phân, vì mỗi giao dịch thuộc một trong hai lớp.

Theo các file kết quả của dự án, train có 699.960 dòng và test có 299.984 dòng. Tỷ lệ fraud trong train chỉ khoảng 1,67%, nên đây là dữ liệu mất cân bằng lớp. Kaggle yêu cầu nộp xác suất `is_fraud` và chấm bằng ROC-AUC. [Trang cuộc thi](https://www.kaggle.com/competitions/playground-series-s6e11/overview)

## 2. ML và DL khác nhau thế nào?

**Machine Learning (ML)** là nhóm phương pháp cho phép máy học quy luật từ dữ liệu. Logistic Regression, Decision Tree, Random Forest, Gradient Boosting và mạng nơ-ron đều là các phương pháp ML.

**Deep Learning (DL)** là một nhánh của ML dùng mạng nơ-ron có nhiều lớp để học biểu diễn và quan hệ phi tuyến phức tạp. Vì vậy DL không phải hướng đối lập với ML; DL nằm bên trong phạm vi ML.

Trong nhánh DL của dự án:

- **MLP** là mạng nơ-ron nhiều lớp fully connected, nhận các đặc trưng đã được biến đổi.
- **TabNet** là mạng dành cho dữ liệu dạng bảng, dùng attention để chọn và kết hợp đặc trưng qua nhiều bước.
- **Embedding MLP** vẫn là MLP, nhưng biểu diễn biến phân loại bằng vector học được thay vì tạo một cột 0/1 cho mỗi giá trị như one-hot.

LLM không được dùng để dự đoán trong pipeline này. Mô hình dự đoán là MLP hoặc TabNet, được huấn luyện trực tiếp trên các dòng giao dịch.

## 3. Train, validation và test

Ba phần dữ liệu có nhiệm vụ khác nhau:

| Tập | Dùng để làm gì? | Có nhãn khi huấn luyện? |
|---|---|---|
| Train | Tính loss và cập nhật trọng số mô hình | Có |
| Validation | So sánh mô hình, chọn cấu hình và số epoch | Có |
| Test | Tạo dự đoán cuối để nộp Kaggle | Không có nhãn công khai |

Task 2.1 chia [train.csv](../../data/raw/train.csv) theo tỷ lệ 80/20: 559.968 dòng train và 139.992 dòng validation. Đây không phải là chia [test.csv](../../data/raw/test.csv). Validation được tách bằng stratified split để tỷ lệ fraud gần như bằng nhau ở hai phần.

**Seed 9** cố định các phép lấy mẫu ngẫu nhiên để các lần chạy có thể tái lập và các mô hình so sánh trên cùng dòng validation. Seed không có giá trị “tốt” mặc định; 9 không tự làm mô hình chính xác hơn 42 hay một số khác.

### Rò rỉ dữ liệu (data leakage)

Các phép tiền xử lý như StandardScaler và OneHotEncoder phải được fit trên train split rồi mới áp dụng lên validation. Nếu fit trên cả validation, thông tin của tập đang dùng để đánh giá đã lọt vào bước chuẩn bị dữ liệu, khiến điểm validation có thể lạc quan hơn thực tế.

Cột `id` chỉ dùng để ghép lại đúng giao dịch, không nên làm đầu vào mô hình. Cột `is_fraud` là đáp án cần dự đoán nên cũng không được đưa vào X.

## 4. Mạng nơ-ron học như thế nào?

Trong MLP của task 2.1, mỗi lớp biến đổi đầu vào bằng các trọng số và bias mà mô hình phải học. Kích thước `54 → 128 → 64 → 1` nghĩa là có 54 giá trị đầu vào, hai hidden layer có lần lượt 128 và 64 neuron, rồi một đầu ra.

Quá trình học lặp lại các bước sau:

1. **Forward pass:** đưa một batch giao dịch qua các lớp để tính đầu ra.
2. **Logit và xác suất:** lớp cuối tạo logit z. Hàm sigmoid chuyển z thành xác suất p = 1 / (1 + e⁻ᶻ).
3. **Tính loss:** so sánh xác suất với nhãn thật. Với binary cross entropy, dự đoán sai và quá tự tin bị phạt nhiều hơn.
4. **Backpropagation:** tính xem mỗi trọng số góp phần gây ra lỗi bao nhiêu.
5. **Optimizer update:** AdamW cập nhật trọng số để loss giảm.
6. Lặp qua nhiều batch và epoch, đồng thời theo dõi metric validation.

Trong code, `BCEWithLogitsLoss` nhận trực tiếp logits và kết hợp phép sigmoid với binary cross entropy theo cách ổn định về số học. Sigmoid vẫn được dùng để tạo xác suất khi tính metric hoặc ghi submission.

### Các tham số thường gặp

| Tham số | Ý nghĩa trong thực tế |
|---|---|
| Epoch | Một lượt mô hình đi qua toàn bộ phần train |
| Batch size | Số dòng được xử lý trước mỗi lần cập nhật trọng số |
| Learning rate | Độ lớn bước cập nhật; quá cao có thể làm học chao đảo, quá thấp có thể học chậm |
| Dropout | Tắt ngẫu nhiên một phần neuron trong lúc train để hạn chế học thuộc |
| Weight decay | Hình thức regularization, hạn chế trọng số tăng quá lớn |
| Patience | Số lần validation không cải thiện trước khi dừng sớm |
| Gradient clipping | Giới hạn độ lớn gradient để tránh cập nhật quá mạnh |
| Seed | Cố định các nguồn ngẫu nhiên; không phải một tham số làm tăng chất lượng mô hình |

## 5. Tiền xử lý đặc trưng

Task 2.1 có 19 cột số/Boolean và 5 cột phân loại. Boolean được đổi thành số 0/1.

- **StandardScaler:** đưa đặc trưng số về thang đo tương đương hơn, thường bằng cách trừ trung bình và chia độ lệch chuẩn. Điều này giúp tối ưu mạng ổn định hơn khi các cột có đơn vị khác nhau.
- **One-hot encoding:** biến một cột phân loại thành nhiều cột 0/1. Ví dụ `channel=web` sẽ được biểu diễn bằng một vị trí bật trong nhóm cột channel.
- Sau one-hot, MLP nhận **54 đặc trưng**. Có thể hiểu đây là 19 cột số/Boolean cộng với 35 cột one-hot được tạo từ 5 cột phân loại.
- `handle_unknown="ignore"` cho phép encoder xử lý giá trị phân loại mới xuất hiện ở validation hoặc test mà không làm notebook lỗi.

Một lựa chọn khác là **embedding**: mỗi giá trị phân loại được ánh xạ sang một vector nhỏ có thể học. Những giá trị có quan hệ hữu ích có thể được mạng biểu diễn gần nhau. Embedding tiết kiệm số chiều khi category có nhiều giá trị, nhưng không đảm bảo sẽ tốt hơn one-hot.

## 6. Mất cân bằng lớp và các metric

Vì fraud chiếm khoảng 1,67%, một mô hình đoán tất cả giao dịch là bình thường vẫn có thể đạt accuracy khoảng 98,33% trong một ví dụ có cùng tỷ lệ, nhưng sẽ bỏ sót toàn bộ fraud. Do đó accuracy không phải metric phù hợp để chọn mô hình ở đây.

- **ROC-AUC — metric chính của cuộc thi:** đo khả năng xếp hạng một giao dịch fraud cao hơn một giao dịch bình thường trên mọi ngưỡng. Có thể hình dung chọn ngẫu nhiên một mẫu fraud và một mẫu bình thường; AUC là xác suất điểm của mẫu fraud cao hơn, có tính cả cách xử lý điểm bằng nhau. AUC 0,5 tương ứng mức xếp hạng ngẫu nhiên; càng gần 1 càng tốt.
- **Average Precision (AP):** tóm tắt đường precision-recall và tập trung nhiều hơn vào việc tìm đúng lớp fraud hiếm. Trong dự án này AP là metric phụ để cùng xem chất lượng phát hiện lớp dương.
- **Precision:** trong các giao dịch mô hình báo là fraud, tỷ lệ thực sự là fraud.
- **Recall:** trong tất cả giao dịch fraud thật, tỷ lệ mô hình phát hiện được.
- **Threshold:** ngưỡng để đổi xác suất thành nhãn 0/1. Threshold 0,5 chỉ phục vụ phân tích false positive/false negative trong task 2.4. Vì Kaggle chấm ROC-AUC trên điểm số liên tục, đổi threshold không làm tăng ROC-AUC.

## 7. Các task DL trong dự án

### Task 2.1 — MLP baseline

[Notebook 2.1](../src/credit-card-fraud-2-1-mlp-baseline.ipynb) tạo mô hình chuẩn đầu tiên. Cấu hình: 54 đầu vào, hidden layers 128 và 64, ReLU, dropout 0,20/0,10, batch size 4096, AdamW với learning rate 0,001 và weight decay 0,0001. Loss là BCEWithLogitsLoss không class weight. Tối đa 12 epoch, patience 3; gradient clipping ở mức 1,0.

Kết quả tốt nhất trên validation ở epoch 9: ROC-AUC **0,944622**, AP **0,484340**. Đây là kết quả validation, chưa phải điểm Kaggle. [Metrics task 2.1](../results/task_2_1_mlp/metrics.json)

#### Đọc notebook task 2.1 theo từng khối

1. **Nạp và khảo sát:** notebook đọc [train.csv](../../data/raw/train.csv) từ input Kaggle, xem số dòng/cột, kiểu dữ liệu, số giá trị category và số mẫu mỗi nhãn. Mục đích là biết cột nào là số, cột nào cần mã hóa và tỷ lệ mất cân bằng trước khi chọn mô hình.
2. **Tách nhãn và chia dữ liệu:** y lấy từ is_fraud; X bỏ id và is_fraud. train_test_split chia các chỉ số dòng theo 80/20, random_state=9 và stratify=y. Chia theo chỉ số giúp lưu lại membership train/validation bằng id để notebook khác dùng lại chính xác.
3. **Tiền xử lý:** Boolean đổi thành 0/1; numeric đi qua StandardScaler; 5 cột category đi qua OneHotEncoder. ColumnTransformer gom các bước theo loại cột. fit_transform chỉ chạy trên train split; validation chỉ transform bằng các thống kê và category đã học từ train.
4. **Dựng mô hình:** DataLoader xáo thứ tự train và chia batch 4096. MLP lần lượt tính Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear. Hidden layers giúp học quan hệ phi tuyến; output cuối là một logit cho mỗi giao dịch.
5. **Cập nhật trọng số:** mỗi batch gọi zero_grad, tính logits và loss, backward để tính gradient, clip gradient tối đa 1, rồi AdamW cập nhật. Learning rate 0,001 đặt bước cập nhật; weight decay 0,0001 regularize trọng số. Đây là baseline không class weight để có mốc so sánh sạch.
6. **Chọn checkpoint:** sau mỗi epoch, notebook chuyển sang eval mode, tắt gradient và chấm ROC-AUC/AP trên validation. Checkpoint được lưu khi ROC-AUC tốt hơn; dừng khi 3 epoch liên tiếp không cải thiện hoặc đã chạy tối đa 12 epoch. Vì vậy best_epoch=9 là epoch có AUC validation cao nhất, còn epochs_run=12 là tổng số epoch đã chạy.
7. **Lưu kết quả:** [validation_predictions_seed9.csv](../results/task_2_1_mlp/validation_predictions_seed9.csv) có id, nhãn thật và xác suất dự đoán; [metrics.json](../results/task_2_1_mlp/metrics.json) lưu cấu hình/metric; [training_history.csv](../results/task_2_1_mlp/training_history.csv) và [learning_curves.png](../results/task_2_1_mlp/learning_curves.png) giúp xem loss và metric thay đổi theo epoch. [Checkpoint best_model.pt](../checkpoints/task_2_1_mlp/best_model.pt) lưu trọng số tốt nhất; [preprocessor.joblib](../checkpoints/task_2_1_mlp/preprocessor.joblib) lưu biến đổi đã fit. File validation predictions không phải submission test.

#### Code cốt lõi 2.1: chia dữ liệu và tiền xử lý

Trích từ [notebook 2.1](../src/credit-card-fraud-2-1-mlp-baseline.ipynb):

```python
y = train[TARGET].astype(np.float32).to_numpy()
row_indices = np.arange(len(train))
train_idx, valid_idx = train_test_split(
    row_indices, test_size=0.20, random_state=SEED, stratify=y
)

X = train.drop(columns=[ID_COL, TARGET]).copy()
bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
X[bool_cols] = X[bool_cols].astype("int8")
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [col for col in X.columns if col not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), num_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), cat_cols),
    ],
    remainder="drop",
)
X_train = preprocessor.fit_transform(X.iloc[train_idx]).astype(np.float32)
X_valid = preprocessor.transform(X.iloc[valid_idx]).astype(np.float32)
y_train = y[train_idx]
y_valid = y[valid_idx]
```

**Giải thích để trả lời:** y lấy nhãn cần dự đoán; row_indices là vị trí các giao dịch. test_size=0.20 dành 20% cho validation, random_state=SEED cố định phép chia, còn stratify=y giữ tỷ lệ fraud gần nhau ở hai phần. X loại id và nhãn vì chúng không phải đầu vào hợp lệ của mô hình. Boolean đổi sang số; cột chữ/category được tách khỏi cột số. ColumnTransformer áp dụng scaler cho numeric và one-hot cho category. fit_transform học thống kê/nhóm category chỉ từ train; validation chỉ transform bằng các gì train đã học, tránh rò rỉ.

#### Code cốt lõi 2.1: MLP, loss và cập nhật trọng số

```python
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 128),
    nn.ReLU(),
    nn.Dropout(0.20),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.10),
    nn.Linear(64, 1),
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

**Giải thích để trả lời:** nn.Linear tạo lớp fully connected; ReLU thêm tính phi tuyến; Dropout(0.20) và Dropout(0.10) tắt ngẫu nhiên phần trăm neuron tương ứng trong lúc train để giảm overfit. Lớp cuối chỉ trả một logit. BCEWithLogitsLoss nhận logit và nhãn nhị phân; AdamW cập nhật model.parameters(). lr=1e-3 là learning rate, weight_decay=1e-4 là regularization. Không có pos_weight nên đây là baseline không weighting.

Phần cập nhật cho một batch trong notebook:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(batch_x)
loss = criterion(logits, batch_y)
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Đọc theo thứ tự:** xóa gradient cũ → forward pass để ra logits → tính loss → backward tính gradient → giới hạn gradient tối đa 1 → optimizer cập nhật trọng số. Validation chuyển mô hình sang eval mode, dùng sigmoid để lấy xác suất rồi tính ROC-AUC/AP. Checkpoint lưu khi ROC-AUC tốt hơn; patience=3 dừng khi ba epoch liền không cải thiện.

**Câu trả lời mẫu:** “Task 2.1 là baseline Deep Learning. Em chia train thành train/validation stratified 80/20 với seed 9, chuẩn hóa cột số, one-hot cột phân loại rồi huấn luyện MLP 54–128–64–1. Mô hình dùng BCEWithLogitsLoss và AdamW; chọn checkpoint theo ROC-AUC validation. Kết quả tốt nhất 0,944622 ở epoch 9.”

### Task 2.2 — TabNet baseline

[Notebook 2.2](../src/credit-card-fraud-2-2-tabnet-baseline.ipynb) giữ nguyên các dòng train/validation seed 9 để so sánh công bằng. TabNet dùng `n_d=16` và `n_a=16` cho kích thước decision/attention, `n_steps=4` cho số bước, `gamma=1,3` để điều chỉnh tái sử dụng đặc trưng và `lambda_sparse=0,0001` cho regularization của mask. Cấu hình train là Adam, learning rate 0,02, batch size 4096, virtual batch 512, tối đa 40 epoch và patience 5.

Kết quả validation: ROC-AUC **0,943122**, AP **0,479909**. Với cấu hình đã thử, TabNet thấp hơn MLP baseline. [Metrics task 2.2](../results/task_2_2_tabnet/metrics.json)

#### Đọc notebook task 2.2 theo từng khối

1. **Thiết lập:** notebook kiểm tra/cài pytorch-tabnet phiên bản 4.1.0 nếu Kaggle chưa có, khai báo seed 9 và thư mục input/output. Cố định phiên bản giúp lần chạy có môi trường dễ tái lập hơn; đây là phần chuẩn bị môi trường, chưa phải học mô hình.
2. **Dùng đúng split:** notebook lấy [split_ids_seed9.csv](../../shared/splits/split_ids_seed9.csv) từ output task 2.1 và ghép vào train bằng id. Các bước kiểm tra id trùng, id thiếu và membership giúp bảo đảm TabNet được chấm trên đúng các giao dịch MLP đã dùng.
3. **Tiền xử lý phù hợp TabNet:** category thiếu được gán nhãn thiếu; numeric chuyển về số, xử lý vô cực/thiếu bằng median tính trên train, sau đó chuẩn hóa. Category được ánh xạ thành mã nguyên; mã 0 dành cho category chưa thấy. cat_idxs chỉ vị trí cột category và cat_dims cho TabNet biết số mã hợp lệ. Không dùng one-hot như ở MLP; TabNet tự xử lý category qua categorical embedding với cat_emb_dim=1.
4. **Hiểu cấu hình:** n_d=16 là kích thước decision representation; n_a=16 là kích thước attention representation; n_steps=4 cho 4 bước lựa chọn/kết hợp đặc trưng. gamma=1,3 điều chỉnh mức tái sử dụng đặc trưng giữa các bước. lambda_sparse=0,0001 phạt mask quá dày; mask_type=entmax tạo lựa chọn đặc trưng thưa hơn. Đây là cách TabNet tạo attention trên dữ liệu bảng.
5. **Huấn luyện:** Adam dùng learning rate 0,02; batch size 4096; virtual_batch_size 512 chia batch lớn thành nhóm nhỏ hơn cho một số phép chuẩn hóa bên trong TabNet. Tối đa 40 epoch, patience 5, theo dõi AUC validation. Notebook lưu model và xác suất validation để task 2.4 so sánh.
6. **Diễn giải kết quả:** AUC 0,943122 thấp hơn MLP baseline 0,944622 khoảng 0,00150. Metrics task 2.2 ghi best_epoch=13 theo chỉ số epoch nội bộ của thư viện; task 2.4 cộng 1 khi hiển thị theo cách đếm từ epoch 1, nên bảng so sánh thể hiện epoch 14. Kết luận đúng là cấu hình TabNet đã chạy chưa thắng trên split này; không đủ căn cứ để kết luận TabNet nói chung kém MLP trên mọi dữ liệu/cấu hình.

#### Code cốt lõi 2.2: category codes và TabNet

```python
for col_idx, col in enumerate(cat_cols):
    categories = sorted(X_train_raw[col].astype(str).unique().tolist())
    mapping = {value: index + 1 for index, value in enumerate(categories)}
    category_mappings[col] = mapping
    X_train_cat[:, col_idx] = X_train_raw[col].astype(str).map(mapping).to_numpy(dtype=np.float32)
    X_valid_cat[:, col_idx] = (
        X_valid_raw[col].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.float32)
    )
    cat_dims.append(len(mapping) + 1)  # 0 dành cho category lạ ở validation.
```

**Giải thích để trả lời:** mapping tạo mã nguyên cho từng giá trị category dựa trên train. Mã bắt đầu từ 1 để giữ 0 làm mã dự phòng cho category validation chưa thấy ở train. cat_dims ghi số mã cần dành chỗ, bao gồm mã 0. Các cột category không one-hot ở đây; TabNet nhận chỉ số cột và cardinality để tự xử lý chúng.

```python
model = TabNetClassifier(
    n_d=16, n_a=16, n_steps=4, gamma=1.3, lambda_sparse=1e-4,
    cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr": 2e-2},
    mask_type="entmax", seed=SEED, verbose=10,
)
model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_valid, y_valid)], eval_name=["valid"], eval_metric=["auc"],
    max_epochs=40, patience=5, batch_size=4096, virtual_batch_size=512,
    num_workers=0, drop_last=False,
)
```

**Giải thích để trả lời:** n_d/n_a là chiều biểu diễn decision/attention; n_steps=4 cho số bước attention; gamma điều chỉnh mức tái sử dụng feature giữa các bước; lambda_sparse regularize mask để lựa chọn đặc trưng thưa hơn. Adam dùng lr=0,02. eval_set và eval_metric yêu cầu theo dõi AUC trên validation; max_epochs=40 là giới hạn, patience=5 là dừng sớm. batch_size=4096 là batch huấn luyện; virtual_batch_size=512 chia batch lớn thành nhóm nhỏ hơn cho các phép chuẩn hóa nội bộ.

**Câu trả lời mẫu:** “Em thử TabNet vì nó được thiết kế cho dữ liệu dạng bảng và có cơ chế attention chọn đặc trưng. Em giữ nguyên membership validation để so sánh công bằng. Với cấu hình n_d=n_a=16, 4 bước attention và early stopping theo AUC, điểm là 0,943122, thấp hơn MLP baseline; vì vậy task 2.4 không chọn TabNet.”

### Task 2.3 — Hai thí nghiệm MLP

[Notebook 2.3](../src/credit-card-fraud-2-3-mlp-imbalance-experiments.ipynb) thử từng thay đổi trên cùng split:

1. **MLP weighted:** dùng `pos_weight = số mẫu bình thường / số mẫu fraud`, bằng khoảng 58,88 trong train split. Mô hình bị phạt nặng hơn khi gán xác suất thấp cho fraud. ROC-AUC là **0,944596**, AP **0,480716**.
2. **MLP embedding:** thay one-hot bằng embedding cho 5 cột phân loại và giữ MLP phía sau. ROC-AUC là **0,944339**, AP **0,482583**.

Hai thay đổi này chưa vượt ROC-AUC của baseline. Điều đó cho thấy mất cân bằng cần được đánh giá bằng validation; thêm class weight không tự động làm AUC cao hơn. [Bảng task 2.3](../results/task_2_3_mlp/experiment_table.csv)

#### Đọc notebook task 2.3 theo từng khối

1. **Khôi phục thí nghiệm chuẩn:** notebook đọc [split_ids_seed9.csv](../../shared/splits/split_ids_seed9.csv) và [predictions](../results/task_2_1_mlp/validation_predictions_seed9.csv)/[metrics](../results/task_2_1_mlp/metrics.json) của task 2.1, ghép theo id rồi kiểm tra membership. Đây là mốc chuẩn để thay từng ý tưởng mà không đổi validation.
2. **MLP weighted:** pos_weight = số mẫu non-fraud / số mẫu fraud trong train split = khoảng 58,883. BCEWithLogitsLoss dùng trọng số này cho lỗi ở lớp dương, khiến mô hình chú ý hơn tới fraud. Dữ liệu không được nhân bản hay xóa mẫu; validation giữ nguyên phân phối. Kiến trúc và các tham số huấn luyện giữ gần baseline để tập trung xem ảnh hưởng của loss.
3. **MLP embedding:** mỗi category được đổi thành mã số, rồi nn.Embedding biến mã đó thành vector học được. Kích thước embedding dùng quy tắc dựa trên căn bậc hai số category, tối thiểu 2 và tối đa 16. Vector category được nối với numeric đã chuẩn hóa rồi đưa vào hidden layers 128/64. Category lạ ở validation được mã hóa 0.
4. **So sánh có kiểm soát:** hai thí nghiệm dùng cùng seed/split, batch size 4096, AdamW learning rate 0,001, weight decay 0,0001, tối đa 12 epoch và early stopping patience 3. Đo ROC-AUC chính, AP phụ. Đây giúp trả lời thay đổi nào có tác dụng trên validation, nhưng một seed vẫn có thể chưa đại diện cho biến động dữ liệu.
5. **Kết quả:** weighted AUC 0,944596 ở epoch 5 gần baseline 0,944622 nhưng thấp hơn một lượng nhỏ; embedding AUC 0,944339 ở epoch 10 cũng thấp hơn baseline. AP của embedding 0,482583 thấp hơn AP baseline 0,484340. Trong lần chạy này, tăng trọng số lớp fraud và embedding category chưa cải thiện metric chọn mô hình.

#### Code cốt lõi 2.3: class weighting và embedding

```python
positive_count = int(y_train.sum())
negative_count = int(len(y_train) - positive_count)
pos_weight = negative_count / positive_count

criterion = nn.BCEWithLogitsLoss() if weighting is None else nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([weighting], dtype=torch.float32, device=DEVICE)
)
```

**Giải thích để trả lời:** với nhãn 0/1, tổng y_train đếm số mẫu fraud; số dòng còn lại là non-fraud. Tỷ lệ non-fraud/fraud cho pos_weight≈58,883. Nếu weighting=None thì dùng loss chuẩn; nếu truyền trọng số thì loss phạt lỗi trên lớp fraud nặng hơn. Cách này đổi loss, không thay số lượng dòng train và không resample validation.

```python
cat_emb_dims.append(min(16, max(2, int(np.ceil(np.sqrt(len(mapping) + 1))))))

class EmbeddingMLP(nn.Module):
    def __init__(self, n_numeric, dimensions, embedding_dimensions):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality, embedding_dim in zip(dimensions, embedding_dimensions)
        ])
        self.network = nn.Sequential(
            nn.Linear(n_numeric + sum(embedding_dimensions), 128),
            nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 1),
        )

    def forward(self, numeric, categorical):
        embedded = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.network(torch.cat([numeric, *embedded], dim=1))
```

**Giải thích để trả lời:** công thức đặt kích thước vector embedding theo căn bậc hai cardinality, nhưng giữ trong khoảng 2–16 để không quá nhỏ/quá lớn. Mỗi nn.Embedding học vector cho một category. forward tra vector từng cột category, nối các vector với numeric, rồi đưa tổ hợp đó vào MLP. Khác one-hot, embedding có các tham số học được; nó có thể học quan hệ hữu ích giữa category nhưng không đảm bảo luôn tăng AUC.

**Câu trả lời mẫu:** “Em thử class weighting vì fraud chỉ chiếm khoảng 1,67%. pos_weight làm tăng phạt cho lỗi ở lớp fraud nhưng không thay đổi số dòng. Em cũng thử embedding để biểu diễn category gọn hơn one-hot. Cả hai dùng cùng split với baseline; kết quả AUC đều thấp hơn nhẹ, nên em giữ baseline.”

### Task 2.4 — So sánh và chọn

[Notebook 2.4](../src/credit-card-fraud-2-4-validation-analysis.ipynb) kiểm tra bốn mô hình trên cùng validation và chọn MLP baseline theo ROC-AUC cao nhất. AP được dùng làm metric phụ. Ở threshold 0,5, baseline có 221 false positive và 1.677 false negative; các con số này diễn giải quyết định tại threshold đó chứ không phải cách tính leaderboard.

| Mô hình | ROC-AUC validation | Average Precision | Epoch được ghi nhận |
|---|---:|---:|---:|
| MLP baseline | **0,944622** | **0,484340** | 9 |
| MLP weighted | 0,944596 | 0,480716 | 5 |
| MLP embedding | 0,944339 | 0,482583 | 10 |
| TabNet baseline | 0,943122 | 0,479909 | 14 theo cách đếm từ 1 |

ROC-AUC được dùng để chọn vì đó là metric cuộc thi; AP giúp theo dõi riêng lớp fraud hiếm. Hai metric đo góc nhìn khác nhau nên AP không thay thế tiêu chí chọn đã chốt.

Cross-validation chưa chạy, nên so sánh và chọn mô hình hiện dựa trên một holdout validation duy nhất. Đây là hạn chế cần nhớ nếu tiếp tục thử nhiều cấu hình trên cùng validation. [Bảng so sánh](../results/task_2_4_validation_analysis/comparison.csv) · [Quyết định chọn mô hình](../results/task_2_4_validation_analysis/selected_model.json)

#### Đọc notebook task 2.4 theo từng khối

1. **Nạp artifact:** task 2.4 nhận metrics và validation predictions từ 2.1, 2.2, 2.3. Notebook yêu cầu đúng một file phù hợp cho mỗi mô hình, rồi so sánh id và nhãn thật giữa các file. Nếu khác membership, metric không còn là so sánh công bằng.
2. **Tính lại và xếp hạng:** ROC-AUC/AP được tính lại từ probability và y_validation đã đối chiếu. Bảng còn lưu best_epoch và thời gian nếu artifact có thông tin. Quy tắc chọn: AUC cao nhất; AP dùng khi AUC hòa.
3. **Đồ thị:** learning curves cho biết mô hình còn cải thiện hay đã bắt đầu overfit; ROC curve và phân phối xác suất giúp nhìn tổng quan khả năng tách hai lớp. Một đồ thị không thay thế metric và kiểm tra dữ liệu.
4. **Phân tích threshold 0,5:** xác suất >=0,5 chuyển thành nhãn fraud để đếm TP=661, TN=137.433, FP=221, FN=1.677. Precision khoảng 0,749 và recall khoảng 0,283. Recall thấp ở ngưỡng này nghĩa là nhiều fraud validation chưa đạt 0,5; nó không phủ định AUC cao vì AUC đánh giá khả năng xếp hạng trên mọi ngưỡng.
5. **Quyết định và giới hạn:** notebook chọn MLP baseline với AUC 0,944622 và không dùng test label hoặc leaderboard để chọn. Không chạy cross-validation, nên độ ổn định của thứ hạng giữa mô hình chưa được đo qua nhiều fold.

#### Code cốt lõi 2.4: xếp hạng theo validation

Trước khi tính metric, notebook căn chỉnh các dự đoán theo ID và kiểm tra rằng mỗi mô hình có cùng nhãn validation:

```python
reference = predictions["MLP baseline"]
reference_ids = reference[ID_COL].to_numpy()
reference_y = reference[TARGET].to_numpy(dtype=np.int64)
for name, frame in list(predictions.items()):
    if frame[ID_COL].duplicated().any() or set(frame[ID_COL]) != set(reference_ids):
        raise ValueError(f"{name}: validation membership differs by id.")
    frame = frame.set_index(ID_COL).loc[reference_ids].reset_index()
    if not np.array_equal(frame[TARGET].to_numpy(dtype=np.int64), reference_y):
        raise ValueError(f"{name}: validation labels differ after aligning by id.")
    predictions[name] = frame
```

**Giải thích để trả lời:** notebook chọn predictions của MLP baseline làm thứ tự tham chiếu. Nó kiểm tra ID không trùng và đúng cùng tập; sau đó reorder từng bảng về đúng thứ tự ID tham chiếu và so nhãn thật. Nếu sai membership hoặc nhãn, notebook dừng thay vì cho ra bảng so sánh sai.

```python
rows = []
for name, frame in predictions.items():
    probability = frame["fraud_probability"].to_numpy(dtype=np.float64)
    rows.append({
        "model": name,
        "validation_roc_auc": float(roc_auc_score(reference_y, probability)),
        "validation_average_precision": float(average_precision_score(reference_y, probability)),
    })
comparison = pd.DataFrame(rows).sort_values(
    ["validation_roc_auc", "validation_average_precision"],
    ascending=False,
).reset_index(drop=True)
```

**Giải thích để trả lời:** vòng lặp lấy xác suất của từng mô hình. Cùng reference_y được dùng để tính hai metric trên cùng validation. DataFrame sắp giảm dần theo ROC-AUC; AP là tiêu chí phụ nếu AUC bằng nhau. Mô hình đầu bảng là ứng viên được chọn. So sánh chỉ công bằng khi từng file prediction ứng với cùng id và nhãn.

#### Code cốt lõi 2.4: ngưỡng 0,5 chỉ để xem lỗi

```python
threshold = 0.5
predicted_label = (selected_probability >= threshold).astype(np.int64)
false_positive = int(((predicted_label == 1) & (reference_y == 0)).sum())
false_negative = int(((predicted_label == 0) & (reference_y == 1)).sum())
true_positive = int(((predicted_label == 1) & (reference_y == 1)).sum())
true_negative = int(((predicted_label == 0) & (reference_y == 0)).sum())
```

**Giải thích để trả lời:** threshold đổi xác suất thành nhãn nhị phân; các biểu thức đếm FP/FN/TP/TN. Đoạn này chỉ tạo confusion counts để diễn giải mô hình tại một ngưỡng. Nó không thay điểm xác suất gốc nên không tính lại hoặc cải thiện ROC-AUC.

**Câu trả lời mẫu:** “Task 2.4 là bước đánh giá, không huấn luyện model mới. Em xác nhận các dự đoán dùng cùng id và nhãn validation, tính lại ROC-AUC/AP rồi xếp hạng. MLP baseline đứng đầu với AUC 0,944622. Threshold 0,5 chỉ dùng xem lỗi; giới hạn là em mới dùng một holdout, chưa có cross-validation.”

### Task 2.5 — Refit và tạo submission

[Notebook 2.5](../src/credit-card-fraud-2-5-final-dl-submission.ipynb) lấy MLP baseline và epoch 9 đã chọn, fit preprocessing trên toàn bộ train rồi train lại trên 699.960 dòng. Sau đó notebook dự đoán xác suất cho 299.984 dòng test và sắp theo ID của sample submission.

Kết quả bạn báo sau khi tự submit là Kaggle score **0,94263**. Con số đó thấp hơn validation AUC 0,944622 khoảng 0,00199. Có thể do validation split chưa đại diện hoàn toàn cho test hoặc do biến động lấy mẫu; đây là diễn giải hợp lý từ chênh lệch, chưa phải nguyên nhân đã được chứng minh. Metadata notebook được tạo trước khi bạn báo đã submit nên chưa ghi lại điểm này. [Metadata task 2.5](../results/task_2_5_final/model_metadata.json) · [submission.csv](../results/task_2_5_final/submission.csv)

#### Đọc notebook task 2.5 theo từng khối

1. **Đọc quyết định:** notebook lấy tên mô hình và best_epoch từ [selected_model.json](../results/task_2_4_validation_analysis/selected_model.json) của task 2.4. Với lựa chọn hiện tại, đó là MLP baseline và 9 epoch. Số epoch đã được chọn bằng validation trước đó.
2. **Đọc train/test/sample:** [train.csv](../../data/raw/train.csv) được dùng để học; [test.csv](../../data/raw/test.csv) chỉ dùng tạo xác suất; [sample_submission.csv](../../data/raw/sample_submission.csv) cung cấp schema và thứ tự ID cần giữ. Notebook không cần test label và không được dùng test label để chọn mô hình.
3. **Fit lại tiền xử lý:** scaler và one-hot encoder fit trên toàn bộ [train.csv](../../data/raw/train.csv), sau đó transform [test.csv](../../data/raw/test.csv) bằng đúng biến đổi đó. Việc dùng thêm phần validation cũ để train cuối giúp model tận dụng toàn bộ nhãn train sau khi cấu hình đã được chốt.
4. **Huấn luyện lại:** model được khởi tạo và huấn luyện từ đầu trên toàn bộ 699.960 dòng trong đúng 9 epoch; đây là refit, không phải tiếp tục từ checkpoint 80% train. Vì không còn holdout trong bước cuối, notebook cố định epoch thay vì early stopping trên test. Code tính pos_weight để metadata có thể lưu, nhưng MLP baseline được chọn nên nhánh fit dùng loss không weighting.
5. **Dự đoán và dựng CSV:** logits của test được chuyển thành xác suất bằng sigmoid. Dự đoán được ghép theo id vào sample_submission, rồi kiểm tra cột id/is_fraud, số dòng, ID trùng/thiếu, giá trị null, số hữu hạn và khoảng [0,1]. Kết quả là 299.984 xác suất.
6. **Hiểu score:** bạn báo Kaggle score 0,94263 sau khi tự submit. Điểm này gần validation AUC 0,944622 nhưng thấp hơn khoảng 0,00199. Đây là khác biệt giữa hai bộ đánh giá; chỉ từ hai điểm chưa thể xác định chính xác nguyên nhân.

#### Code cốt lõi 2.5: lấy quyết định và refit mô hình

```python
selected_name = decision["selected_model"]
epochs = int(decision["best_epoch"])
supported = {"MLP baseline", "MLP weighted", "MLP embedding", "TabNet baseline"}
if selected_name not in supported or epochs < 1:
    raise ValueError(f"Invalid selected model or epoch: {selected_name}, {epochs}.")
```

**Giải thích để trả lời:** tên mô hình và số epoch đến từ quyết định task 2.4. supported là allowlist để notebook chỉ chạy những nhánh đã cài đặt; kiểm tra epochs ngăn cấu hình lỗi. Với kết quả hiện tại, selected_name là MLP baseline và epochs=9.

Trước khi huấn luyện, nhánh MLP baseline fit encoder trên full train rồi chỉ transform test:

```python
X_train_raw = train.drop(columns=[ID_COL, TARGET]).copy()
X_test_raw = test.drop(columns=[ID_COL]).copy()
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), num_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), cat_cols),
    ],
    remainder="drop",
)
X_train = preprocessor.fit_transform(X_train_raw).astype(np.float32)
X_test = preprocessor.transform(X_test_raw).astype(np.float32)
```

**Giải thích để trả lời:** train giữ lại nhãn riêng trong y_train, không đưa id/target vào X. Preprocessor được fit trên toàn bộ train có nhãn vì giờ đã chốt mô hình; test chỉ nhận transform. Không được fit lại encoder trên test vì cần cùng cột và cùng phép biến đổi.

Phần khởi tạo nhánh MLP trong notebook:

```python
if selected_name in {"MLP baseline", "MLP weighted"}:
    final_model = fit_mlp(
        nn.Sequential(
            nn.Linear(X_train.shape[1], 128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 1),
        ),
        [X_train],
        weighted=(selected_name == "MLP weighted"),
    )
```

**Giải thích để trả lời:** đây là khởi tạo lại cùng kiến trúc MLP rồi gọi fit_mlp trên X_train đã xử lý từ toàn bộ train. Vì selected_name là MLP baseline, biểu thức weighted trả về False nên BCE loss không dùng pos_weight. Hàm chạy đúng 9 epoch đã chốt từ validation; đây là train lại từ đầu, không phải nạp checkpoint epoch 9 của lần train 80%.

#### Code cốt lõi 2.5: ghép dự đoán vào submission theo ID

```python
probability_by_id = pd.Series(test_probability, index=test[ID_COL].to_numpy())
submission = sample_submission[[ID_COL]].copy()
submission[TARGET] = submission[ID_COL].map(probability_by_id)
submission = submission[[ID_COL, TARGET]]
if submission.isna().any().any() or len(submission) != len(sample_submission):
    raise ValueError("Submission IDs/predictions are missing or row count is wrong.")
if list(submission.columns) != [ID_COL, TARGET]:
    raise ValueError(f"Invalid submission schema: {submission.columns.tolist()}")
```

**Giải thích để trả lời:** Series ánh xạ xác suất theo ID test. Notebook bắt đầu từ cột ID của sample_submission rồi map xác suất theo ID, nên không phụ thuộc thứ tự hiện tại của test. Kiểm tra null/số dòng phát hiện ID thiếu; kiểm tra columns đảm bảo đúng schema Kaggle. Giá trị TARGET là xác suất, không phải nhãn 0/1.

**Câu trả lời mẫu:** “Sau khi chọn MLP baseline và 9 epoch trên validation, em fit lại preprocessing và train model từ đầu trên toàn bộ [train.csv](../../data/raw/train.csv). Sau đó em dự đoán xác suất cho [test.csv](../../data/raw/test.csv) và ghép theo ID của [sample_submission.csv](../../data/raw/sample_submission.csv). Em không dùng nhãn test. Kaggle trả điểm 0,94263; đây là điểm leaderboard em nhận được sau khi nộp.”

## 8. Cách tiếp tục cải thiện điểm số

1. **Đánh giá ổn định trước:** vì các lựa chọn task 2.4 dùng một split, hãy dùng Stratified K-Fold hoặc vài split cố định. So sánh trung bình ROC-AUC và độ dao động giữa các fold, không chỉ một điểm.
2. **Tạo thí nghiệm có kiểm soát:** ghi tên thí nghiệm, tham số thay đổi, seed/fold, ROC-AUC, AP và epoch tốt nhất. Ban đầu chỉ thay một nhóm tham số mỗi lần để biết thay đổi nào có tác dụng.
3. **Ưu tiên thử MLP baseline:** đây là mô hình tốt nhất hiện tại. Có thể lần lượt khảo sát learning rate, kích thước hidden layer, dropout, weight decay và batch size; không chọn cấu hình chỉ vì một lần Kaggle score tăng nhẹ.
4. **Thử ensemble sau cùng:** trung bình dự đoán của vài mô hình hoặc seed chỉ đáng giữ nếu nó cải thiện điểm validation/cross-validation. Các mô hình giống nhau có thể mắc cùng loại lỗi, nên ensemble không được đảm bảo tốt hơn.
5. **Không tối ưu threshold để tăng ROC-AUC:** threshold phục vụ quyết định fraud/normal; leaderboard hiện đánh giá thứ tự của xác suất.
6. **Hạn chế dò public leaderboard:** dùng validation/CV làm căn cứ chính. Nếu thử quá nhiều cấu hình rồi chỉ chọn theo public score, mô hình có thể khớp vào phần public mà không tổng quát tốt.

## 9. Từ điển ngắn

| Thuật ngữ | Cách hiểu ngắn |
|---|---|
| Feature | Một thông tin đầu vào của giao dịch |
| Label/target | Đáp án thật mà mô hình học để dự đoán |
| Parameter/weight | Giá trị bên trong mô hình được học từ dữ liệu |
| Hyperparameter | Cấu hình đặt trước như learning rate, batch size, dropout |
| Logit | Đầu ra thô trước khi đổi thành xác suất |
| Probability | Điểm dự đoán trong khoảng 0–1 |
| Loss | Đại lượng mô hình cố giảm trong khi train |
| Overfitting | Học tốt trên dữ liệu train nhưng tổng quát kém trên dữ liệu mới |
| Checkpoint | Bản lưu trọng số của mô hình tại một thời điểm, thường là epoch validation tốt nhất |
| Submission | File xác suất dự đoán test theo schema Kaggle yêu cầu |

## 10. File nên đọc cùng tài liệu

- [Notebook task 2.1](../src/credit-card-fraud-2-1-mlp-baseline.ipynb) và [metrics task 2.1](../results/task_2_1_mlp/metrics.json)
- [Bảng so sánh task 2.4](../results/task_2_4_validation_analysis/comparison.csv)
- [Notebook task 2.5](../src/credit-card-fraud-2-5-final-dl-submission.ipynb) và [submission cuối](../results/task_2_5_final/submission.csv)

## 11. Câu hỏi vấn đáp thường gặp

### Vì sao chọn bài toán phân loại nhị phân?

Vì mỗi giao dịch có một trong hai nhãn: fraud hoặc non-fraud. Mô hình xuất xác suất thuộc lớp fraud để Kaggle đánh giá khả năng xếp hạng.

### Vì sao chia train/validation 80/20 và stratify?

Train dùng để cập nhật trọng số; validation dùng để chọn checkpoint và so sánh mô hình. Fraud hiếm nên stratify giữ tỷ lệ lớp gần nhau giữa hai phần, giúp phép so sánh ổn định hơn.

### Seed 9 có ý nghĩa gì? Tại sao không chọn seed cho AUC cao nhất?

Seed cố định phép ngẫu nhiên để lần chạy có thể tái lập và hai nhánh so sánh cùng dòng dữ liệu. Seed không có ý nghĩa tốt/xấu tự thân. Chọn seed chỉ vì nó cho điểm cao sẽ làm validation trở thành một mục tiêu bị dò và có thể khiến ước lượng quá lạc quan.

### Vì sao bỏ cột id khỏi feature?

ID dùng để định danh/ghép dự đoán về đúng dòng, không phải tín hiệu hành vi giao dịch có ý nghĩa đã được xác nhận. Đưa ID vào có thể làm mô hình học quy luật giả theo thứ tự hoặc vùng ID.

### Vì sao phải chuẩn hóa numeric và mã hóa category?

Numeric có thang đo khác nhau; chuẩn hóa giúp quá trình tối ưu của mạng dễ ổn định hơn. Mạng không hiểu tên category dạng chữ nên phải đổi sang dạng số; one-hot và embedding là hai cách biểu diễn khác nhau.

### Tại sao không dùng accuracy?

Fraud chỉ khoảng 1,67%. Một mô hình đoán tất cả là non-fraud có thể đạt accuracy cao nhưng phát hiện fraud bằng không. Vì cuộc thi chấm ROC-AUC, nhóm dùng AUC làm metric chọn mô hình và xem AP như metric phụ.

### ROC-AUC 0,944622 nghĩa là gì? Có phải đúng 94,46% giao dịch không?

Không. AUC không phải accuracy. Có thể diễn giải là khi lấy một fraud và một non-fraud ngẫu nhiên, điểm của fraud có cơ hội khoảng 94,46% cao hơn điểm non-fraud, tính theo quy ước xử lý ties.

### Loss và metric khác nhau như thế nào?

Loss như BCEWithLogitsLoss tạo tín hiệu gradient để cập nhật trọng số. ROC-AUC/AP đánh giá kết quả trên validation. Loss giảm không bảo đảm AUC tăng, nên notebook vừa tối ưu loss trên train vừa chọn checkpoint theo ROC-AUC validation.

### Vì sao MLP baseline thắng TabNet và MLP weighted trong lần chạy này?

Trên đúng split và cấu hình đã thử, MLP baseline có ROC-AUC cao nhất. Điều đó chỉ chứng minh nó là lựa chọn tốt nhất trong bộ thí nghiệm hiện có, không chứng minh mọi MLP luôn tốt hơn mọi TabNet hoặc mô hình có weighting.

### pos_weight khoảng 58,88 có nghĩa là gì?

Đó là tỷ lệ số mẫu non-fraud trên số mẫu fraud trong train split. Nó tăng trọng số loss của lớp fraud để lỗi trên lớp hiếm có ảnh hưởng lớn hơn. Nó không nhân dữ liệu fraud lên và cũng không bảo đảm ROC-AUC tăng.

### One-hot và embedding khác nhau ra sao?

One-hot tạo một cột riêng cho từng category; cách này đơn giản và không áp đặt quan hệ thứ tự giữa category. Embedding ánh xạ category sang vector có thể học, gọn hơn nhưng cần mạng học được biểu diễn hữu ích. Trong thí nghiệm hiện tại embedding chưa vượt baseline.

### Vì sao threshold 0,5 không làm thay đổi Kaggle AUC?

Threshold biến xác suất thành nhãn 0/1 và làm thay đổi confusion matrix, precision, recall. ROC-AUC đánh giá thứ tự score ở mọi threshold, nên việc đổi một ngưỡng phân loại không làm thay đổi các score gốc.

### Vì sao task 2.5 train lại trên toàn bộ train?

Khi đã chọn mô hình và epoch bằng validation, có thể dùng cả dữ liệu validation cũ làm dữ liệu train cuối để tận dụng toàn bộ nhãn có sẵn. Số epoch được cố định bằng best_epoch=9 vì bước refit không giữ validation riêng.

### Vì sao điểm Kaggle thấp hơn validation?

Validation và tập được Kaggle chấm là hai phần dữ liệu khác nhau. Chênh lệch có thể do biến động lấy mẫu hoặc validation chưa đại diện hoàn toàn; đây là giả thuyết hợp lý, chưa thể khẳng định chỉ từ hai điểm. Cross-validation sẽ giúp ước lượng độ ổn định tốt hơn.

### Hạn chế lớn nhất hiện tại là gì? Em sẽ làm gì tiếp?

So sánh mô hình dựa trên một holdout validation duy nhất và mới thử một cấu hình cơ bản cho mỗi họ mô hình. Bước tiếp theo là stratified cross-validation, tuning có kiểm soát và chỉ giữ thay đổi/ensemble nếu điểm trung bình trên validation folds được cải thiện.

## 12. Bài trình bày mẫu khoảng một phút

“Bài toán của em là dự đoán xác suất gian lận cho mỗi giao dịch thẻ. Dữ liệu fraud chỉ khoảng 1,67%, nên accuracy không phản ánh tốt khả năng phát hiện lớp hiếm; Kaggle đánh giá bằng ROC-AUC. Em dùng stratified split 80/20 với seed 9 và giữ cùng membership validation cho các mô hình. Task 2.1 tạo MLP baseline với 54 đặc trưng sau tiền xử lý, hidden layers 128 và 64, BCEWithLogitsLoss và AdamW; AUC validation cao nhất là 0,944622 ở epoch 9. Sau đó em so sánh TabNet, MLP có class weight và MLP dùng embedding; cả ba chưa vượt baseline trên split này. Task 2.4 chọn MLP baseline, còn task 2.5 train lại 9 epoch trên toàn bộ train và tạo xác suất cho test theo đúng ID. Điểm Kaggle em nhận được là 0,94263. Hạn chế hiện tại là chưa chạy cross-validation, nên hướng tiếp theo là đo độ ổn định qua nhiều folds trước khi tuning sâu hoặc ensemble.”
