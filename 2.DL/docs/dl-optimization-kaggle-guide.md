# Hướng dẫn chạy pipeline tối ưu Deep Learning trên Kaggle

> **Tài liệu lịch sử:** Source notebook 2.6–2.11 đã được gỡ khỏi nhánh DL; quy trình hiện tại giữ các bước 2.1–2.5. Hướng dẫn bên dưới mô tả pipeline cũ. Các liên kết source trỏ về commit lưu trước khi gỡ, không phải source đang có trong nhánh.

## 1. Phạm vi và trạng thái

Đây là phần triển khai tiếp theo của nhánh 2, gồm MLP, embedding MLP, TabNet và ensemble DL. Các notebook lịch sử 2.1–2.5 được giữ nguyên. Source mới là notebook; không yêu cầu chạy train trên laptop.

**Verified — đã có trong source:** sàng lọc cấu hình theo holdout seed 9, preprocessing theo split/fold, CV 5-fold, lựa chọn OOF và inference bằng checkpoint cùng cấu hình thực tế. Mỗi task có Markdown giải thích các khối code để ôn vấn đáp.

**Unknown — chưa có kết quả thực nghiệm mới:** chưa chạy các notebook tối ưu trên Kaggle, nên chưa có AUC mới hoặc bằng chứng vượt score 0.94263 bạn báo trước đây. Các con số mới chỉ xuất hiện sau khi notebook chạy thành công.

## 2. Thứ tự và input bắt buộc

Mọi notebook cần gắn competition data. Mặc định DATA_DIR=None: code tìm dữ liệu bên dưới `/kaggle/input`, không giả định cấu trúc mount cố định. Task 2.7–2.11 chọn thư mục có checksum train trùng foundation 2.6; task 2.11 còn yêu cầu đủ cả ba CSV. Nếu có nhiều thư mục hợp lệ, đặt DATA_DIR thành đường dẫn cụ thể được liệt kê trong thông báo lỗi.

| Task/source | Output notebook cần Add Input | GPU | Đầu ra chính |
|---|---|---|---|
| [2.6 — Foundation](https://github.com/nvtrung18/Predicting-Credit-Card-Fraud/blob/abba0da4c0c0eff06db78c15156ec2d3a76c350e/2.DL/src/credit-card-fraud-2-6-experiment-foundation.ipynb) | Output task 2.1, có file split theo ID | Có | Audit, holdout seed 9, CV folds, module chung, baseline chạy lại |
| [2.7 — MLP tuning](https://github.com/nvtrung18/Predicting-Credit-Card-Fraud/blob/abba0da4c0c0eff06db78c15156ec2d3a76c350e/2.DL/src/credit-card-fraud-2-7-mlp-tuning.ipynb) | Output 2.6 | Có | Bảng thí nghiệm và toàn bộ config/checkpoint/history |
| [2.8 — Feature/embedding/TabNet](https://github.com/nvtrung18/Predicting-Credit-Card-Fraud/blob/abba0da4c0c0eff06db78c15156ec2d3a76c350e/2.DL/src/credit-card-fraud-2-8-dl-feature-experiments.ipynb) | Outputs 2.6 + 2.7 | Có | Bảng screening gồm cả ứng viên cũ và biến thể mới |
| [2.9 — Cross-validation](https://github.com/nvtrung18/Predicting-Credit-Card-Fraud/blob/abba0da4c0c0eff06db78c15156ec2d3a76c350e/2.DL/src/credit-card-fraud-2-9-dl-cross-validation.ipynb) | Outputs 2.6 + 2.8 | Có | Baseline + tối đa 2 ứng viên, 5 checkpoints mỗi config, OOF và điểm từng fold |
| [2.10 — OOF ensemble](https://github.com/nvtrung18/Predicting-Credit-Card-Fraud/blob/abba0da4c0c0eff06db78c15156ec2d3a76c350e/2.DL/src/credit-card-fraud-2-10-dl-oof-ensemble.ipynb) | Outputs 2.6 + 2.9 | Không bắt buộc | Quyết định chọn mô hình đơn/ensemble và trọng số cố định |
| [2.11 — Submission](https://github.com/nvtrung18/Predicting-Credit-Card-Fraud/blob/abba0da4c0c0eff06db78c15156ec2d3a76c350e/2.DL/src/credit-card-fraud-2-11-optimized-dl-submission.ipynb) | Outputs 2.6 + 2.9 + 2.10 | Khuyến nghị | CSV dự đoán, metadata và báo cáo; không train lại |

1. Import source notebook vào Kaggle, Add Input theo bảng, bật GPU cho bước train.
2. Dùng Save Version / Run All; chờ trạng thái thành công và output có đủ artifact.
3. Gắn **output đã lưu** vào task tiếp theo. Các biến Python trong phiên interactive không tự đi sang notebook khác.
4. Giữ đúng version của 2.6 xuyên suốt pipeline. Nếu thay engine/config dữ liệu, chạy lại các task phụ thuộc; checksum sẽ chặn việc trộn các version không khớp.
5. Task 2.11 có thể chạy riêng khi đã có outputs 2.6, 2.9 và 2.10. Tải CSV từ output sau khi hoàn tất; notebook không tự submit leaderboard.

Nếu gắn nhiều version 2.6, điền FOUNDATION_DIR chính xác. Với các manifest khác, chỉ gắn một output tương ứng, tránh tìm thấy nhiều file cùng tên.

## 3. Code dùng chung nằm ở đâu?

[Notebook 2.6](https://github.com/nvtrung18/Predicting-Credit-Card-Fraud/blob/abba0da4c0c0eff06db78c15156ec2d3a76c350e/2.DL/src/credit-card-fraud-2-6-experiment-foundation.ipynb) có các cell `%%writefile` chứa đầy đủ source có thể đọc/sửa. Khi chạy trên Kaggle, chúng xuất module `credit_fraud_dl_engine.py` vào output của 2.6. Các notebook sau import module này từ input đã gắn; laptop không cần một source Python riêng.

Module có các phần:

| Phần code | Vai trò và lý do |
|---|---|
| BASELINE, config, config_id | Mỗi cấu hình có ID theo nội dung; tránh nhầm LR/kiến trúc/seed giữa các lượt |
| Preprocessor | Fit median/scaler/category mapping trên train của đúng split/fold; lưu lại để test chỉ transform |
| mlp_layers, EmbeddingMLP | Dựng kiến trúc từ config; embedding ghép vector category với numeric |
| train_run | Train, chọn best checkpoint bằng validation AUC, lưu lịch sử và artifact |
| predict_run | Dựng lại model từ spec/checkpoint, nạp đúng preprocessor; không hardcode model cuối |
| scores, prediction_frame | Kiểm tra xác suất và căn chỉnh OOF bằng ID/nhãn |
| export_submission | Ghép theo ID của sample, kiểm tra schema/số dòng/khoảng xác suất |

Huấn luyện có guard bắt buộc thư mục Kaggle và GPU. MLP dùng cuda:0; GPU thứ hai nếu được cấp chưa được dùng bằng multi-GPU. Không phải cứ được cấp T4 x2 là code tự tận dụng cả hai.

## 4. Cấu hình và ngân sách thí nghiệm

### 2.6 — Baseline

Baseline: hidden 128/64, dropout 0.20/0.10, AdamW LR 0.001, weight decay 0.0001, batch 4096, 12 epoch, patience 3, training seed 9, không scheduler và không class weighting. MLP có kiểm tra/điền thiếu nhất quán; nếu dữ liệu/môi trường khác lượt lịch sử, baseline chạy lại có thể lệch. Mọi thí nghiệm mới so với baseline chạy lại trong cùng foundation.

Audit feature trùng không dùng nhãn để tạo nhóm. CV dùng StratifiedGroupKFold khi feature vector bị trùng, ngược lại StratifiedKFold; cả hai có 5 fold, shuffle seed 9. Holdout lịch sử không đổi, và audit ghi riêng số nhóm trùng xuyên holdout.

### 2.7 — Tuning MLP

1. 5 lựa chọn LR/batch: LR 0.0003, 0.001, 0.003; batch 1024, 2048 hoặc 4096 theo danh sách đã định.
2. 4 biến thể hidden/dropout trên cấu hình thắng bước 1.
3. Weight decay 0, 0.00001, 0.0001; scheduler off; class weight 3 hoặc balanced, giữ các biến khác cố định.
4. Các cấu hình trùng nội dung không chạy lại. Ngân sách tối đa 16 config mới; không chạy toàn bộ tích các giá trị.

Các cấu hình tuning đặt max_epochs=50, patience=8, scheduler ReduceLROnPlateau theo AUC, factor=0.5, scheduler patience=2. Learning rate từng epoch được ghi lại. Giới hạn epoch cao không đồng nghĩa epoch cuối là checkpoint được chọn.

### 2.8 — Preprocessing và mô hình bổ sung

- Log1p chỉ chọn trong các cột số tiền/khoảng cách/thời gian đã có, không âm và skewness > 1 ở train của holdout. Không có cột phù hợp thì bỏ biến thể log và ghi danh sách rỗng.
- Sin/cos bổ sung cho giờ/ngày, giữ feature gốc; thử riêng trước khi kết hợp log + cyclical.
- Thử Boolean giữ 0/1 thay vì scale.
- Embedding MLP: kích thước theo cardinality hoặc 4 chiều/category, giữ tham số MLP đã chọn.
- TabNet: tối đa 4 cấu hình. RUN_TABNET=True mặc định; cần package pytorch-tabnet 4.1.0 hoặc Internet để pip. Có thể đặt False nếu thiếu compute; manifest ghi rõ đã bỏ qua. Các biến thể TabNet dùng Adam, weight decay 0, không class weight, tối đa 60 epoch/patience 10.

### 2.9–2.10 — Xác nhận và lựa chọn

Tối đa 15 lượt train CV: baseline + 2 config mới, mỗi config 5 fold. Có thể đặt SHORTLIST_IDS để kiểm chứng các config đã chốt thay vì chọn tự động; tối đa 2 ID và phải tồn tại trong screening.

Mặc định chỉ thay baseline nếu mean fold AUC tăng ít nhất 0.0001 và thắng tối thiểu 3/5 fold. Ensemble chỉ thử 25/75, 50/50, 75/25 giữa hai config có mean fold AUC cao nhất, và phải cải thiện so với mô hình đơn đã chọn theo cùng chính sách. Đây là tiêu chí thực hành, không phải kiểm định thống kê hay cam kết tăng leaderboard score.

## 5. Chống nhầm cấu hình và checkpoint

1. Foundation lưu checksum dữ liệu, module và membership. Các task sau kiểm tra trước khi chạy.
2. Mỗi run lưu đầy đủ config, model_spec, best_epoch, môi trường, thời gian, metric và checksum model/preprocessor/history/predictions.
3. Sau khi train, code nạp lại checkpoint + preprocessor và so dự đoán validation với dự đoán vừa có. Run chỉ đánh dấu reload_verified sau khi khớp trong sai số số học cho phép.
4. Chỉ reuse run đã hoàn tất, đã qua bước reload và có cùng signature/checksum. Lượt chưa hoàn tất bắt đầu lại từ đầu; không có cơ chế tiếp tục giữa chừng từ optimizer state.
5. Mở một Kaggle session mới không tự khôi phục file working của session cũ. Muốn reuse cần tự gắn/copy artifact đúng version; pipeline không tự tải remote.
6. Inference yêu cầu cùng sklearn version để nạp preprocessor. Nên giữ cùng Kaggle image/environment cho các task; metadata có phiên bản thực tế. Không cài phiên bản thư viện mới tùy ý giữa training và inference.

## 6. Vì sao không refit full train trong 2.11?

Lựa chọn triển khai là **fold ensemble** đã có trong kế hoạch: dùng trực tiếp 5 checkpoint tốt nhất của CV cho mỗi config. Mỗi model thấy khoảng 80% train, và mỗi dòng từng nằm trong train của các fold còn lại. Dự đoán test là trung bình 5 model, rồi mới blend config theo trọng số OOF đã khóa.

Ưu điểm là dùng đúng model/preprocessor đã được kiểm chứng, không đoán lại số epoch cho một lượt refit full train và không huấn luyện thêm chỉ để tạo CSV. Đây là lựa chọn cần so sánh thực nghiệm, không đảm bảo luôn tốt hơn refit.

## 7. Cách đọc kết quả và trình bày vấn đáp

1. **Holdout:** so baseline và tuning trên đúng membership 80/20 seed 9. Dùng để screening, không coi điểm tốt nhất sau nhiều lượt là ước lượng test độc lập.
2. **Mean fold AUC:** trung bình AUC của 5 validation fold; so ứng viên và baseline trên đúng các fold này.
3. **Pooled OOF AUC:** ghép dự đoán OOF của tất cả các dòng rồi tính một AUC chung. Có thể khác mean fold AUC do cách gộp và thang score giữa model fold; không đánh đồng.
4. **Std fold AUC:** mô tả mức dao động giữa các fold; không phải confidence interval, và các fold dùng train chồng lắp.
5. **Giới hạn:** shortlist đã được screening trên holdout có nhãn; mỗi fold dùng validation để early stop; trọng số ensemble cũng được chọn trên OOF. Vì vậy đây là CV chọn mô hình, chưa phải nested CV hoặc test độc lập. Không kết luận chắc chắn mô hình mới tốt hơn từ một chênh lệch nhỏ.
6. **Leaderboard:** score 0.94263 là baseline do bạn báo. Metadata submission mới để leaderboard score là null cho đến khi có kết quả thật; không giả lập hoặc điền trước.

Khi thầy hỏi “Em tối ưu những gì?”, có thể trả lời: “Em giữ baseline và membership dữ liệu, đưa cấu hình vào từng thí nghiệm, tuning quá trình học và kiến trúc MLP, thử tiền xử lý/embedding/TabNet, rồi kiểm chứng shortlist bằng cùng 5 fold. Em dùng OOF để xét ensemble và nạp đúng checkpoint/preprocessor khi tạo submission. Em báo riêng điểm holdout, CV và Kaggle vì chúng khác nhau.”

## 8. Nguồn kỹ thuật

- [PyTorch — ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html): scheduler giảm LR khi metric chững lại.
- [scikit-learn — StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html): fold giữ gần tỷ lệ lớp.
- [pytorch-tabnet v4.1.0](https://github.com/dreamquark-ai/tabnet/tree/v4.1.0): implementation và định dạng lưu/nạp TabNet.

## 9. Lỗi không tìm thấy dữ liệu trong cell đầu

Nếu log báo FileNotFoundError khi đọc train ở đường dẫn cố định, notebook chưa tìm thấy competition data tại đường dẫn đó. Việc nạp được module của 2.6 chỉ xác nhận output notebook đã được gắn; competition data là một input riêng.

1. Add Input phải có competition playground-series-s6e11 và các notebook outputs trong bảng mục 2.
2. Dùng bản source mới có DATA_DIR=None và hàm resolve_data_dir. Nó hỗ trợ cả mount trực tiếp hoặc nằm trong thư mục lồng nhau, đồng thời giữ kiểm tra checksum.
3. Nếu dữ liệu chưa được gắn, thông báo mới yêu cầu Add Input và liệt kê các entry đang có. Nếu CSV khác checksum 2.6, code dừng để tránh dùng nhầm dữ liệu. Nếu có nhiều bản hợp lệ, điền chính xác DATA_DIR.
4. Bản sửa chỉ thay loader, không thay module/checkpoint 2.6. Có thể dùng output 2.6 hiện có với các notebook 2.7–2.11 đã sửa, không phải train lại 2.6 chỉ vì lỗi đường dẫn.
5. Các dòng cảnh báo debugger về frozen modules không phải nguyên nhân FileNotFoundError này.
