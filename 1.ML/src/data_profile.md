# BÁO CÁO HỒ SƠ DỮ LIỆU & CHIẾN LƯỢC ĐÁNH GIÁ (DATA PROFILE & EVALUATION)
**Dự án:** Predicting Credit Card Fraud (Kaggle Playground Series s6e11)  
**Nhóm:** Lê Trung Hiếu (Nhánh Pure ML) & Nguyễn Văn Trung (Nhánh Pure DL)  
**Ngày cập nhật:** 14/11/2026  

---

## 1. Tổng quan Kích thước Dữ liệu
* **Tập huấn luyện (`train.csv`):** 699,960 dòng × 26 cột (kích thước file ~98.5 MB).
* **Tập kiểm tra (`test.csv`):** 299,984 dòng × 25 cột (kích thước file ~41.1 MB).
* **Tập nộp mẫu (`sample_submission.csv`):** 299,984 dòng × 2 cột (`id`, `is_fraud`).

---

## 2. Kiểm tra Chất lượng Dữ liệu
* **Giá trị khuyết (Missing values):** `0` (Không có ô nào bị thiếu dữ liệu ở cả tập train và test).
* **Dòng trùng lặp (Duplicate rows):** `0` (Dữ liệu không có dòng nào trùng lặp).

---

## 3. Phân phối Nhãn Mục tiêu (`is_fraud`)
* **Lớp 0 (Giao dịch hợp lệ):** 688,271 mẫu (**98.3300%**)
* **Lớp 1 (Gian lận - Fraud):** 11,689 mẫu (**1.6700%**)
* 👉 **Nhận định:** Dữ liệu có mức độ mất cân bằng lớp cực cao (**Imbalanced Ratio ~ 59:1**). Điều này khẳng định việc sử dụng **ROC-AUC** làm độ đo chính và **PR-AUC** làm độ đo phụ là hoàn toàn bắt buộc và chính xác.

---

## 4. Danh mục 25 Đặc trưng (Features)
1. **Mã định danh:**
   * `id`: ID duy nhất của từng giao dịch.
2. **Đặc trưng Số (Numerical Features - 12 cột):**
   * `amount_usd`: Số tiền giao dịch (USD).
   * `hours_since_last_txn`: Số giờ kể từ giao dịch trước đó.
   * `txn_count_last_24h`: Số lượng giao dịch trong 24 giờ qua.
   * `distance_from_home_km`: Khoảng cách từ nhà đến địa điểm phát sinh giao dịch (km).
   * `card_age_months`: Tuổi thọ của thẻ (số tháng).
   * `customer_age`: Tuổi của khách hàng.
   * `account_balance_usd`: Số dư tài khoản hiện tại (USD).
   * `cvv_retry_count`: Số lần nhập lại sai mã CVV.
   * `velocity_score`: Điểm tốc độ giao dịch (tần suất/bất thường).
   * `time_of_day_hour`: Giờ trong ngày (0 - 23).
   * `day_of_week`: Ngày trong tuần (0 - 6).
   * `merchant_risk_score`: Điểm rủi ro của người bán/cửa hàng.
   * `prior_disputes`: Số lần tranh chấp/khiếu nại trước đây.
3. **Đặc trưng Phân loại (Categorical Features - 5 cột):**
   * `merchant_category`: Danh mục kinh doanh (Groceries, Online Retail, Travel, v.v.).
   * `card_type`: Loại thẻ (Visa, Mastercard, Discover, Amex, v.v.).
   * `auth_method`: Phương thức xác thực (PIN, OTP, Biometric, Signature, v.v.).
   * `channel`: Kênh giao dịch (POS, In-App, Web, ATM, v.v.).
   * `device_type`: Loại thiết bị (iPhone, Android Phone, Windows PC, Mac, v.v.).
4. **Đặc trưng Nhị phân / Cảnh báo gian lận (Boolean Features - 6 cột):**
   * `is_foreign_transaction`: Giao dịch ở nước ngoài (True/False).
   * `is_new_merchant`: Giao dịch tại cửa hàng mới lần đầu (True/False).
   * `used_vpn`: Có sử dụng mạng riêng ảo VPN không (True/False).
   * `ip_country_mismatch`: Quốc gia của địa chỉ IP không khớp với thẻ (True/False).
   * `billing_shipping_mismatch`: Địa chỉ thanh toán không khớp địa chỉ giao hàng (True/False).
   * `is_ai_generated_scam_attempt`: Dấu hiệu tấn công lừa đảo bằng AI (True/False).

---

## 5. Chiến lược Chia tập Dùng chung (Train / Validation Split)
* **Phương pháp:** **Stratified Split 80/20** cố định ngẫu nhiên với **`seed = 9`**.
* **File kết quả dùng chung:** `data/train_split.csv`
  * Cột bổ sung: `is_val` (Boolean: `False` là tập train 80%, `True` là tập val 20%).
* **Thống kê sau khi chia:**
  * **Tập Train (80%):** 559,968 mẫu (Tỷ lệ fraud: **1.6699%** - 9,351 ca)
  * **Tập Validation (20%):** 139,992 mẫu (Tỷ lệ fraud: **1.6701%** - 2,338 ca)
  * Cả hai tập đều giữ nguyên tỷ lệ phân phối gian lận ban đầu.

---

## 6. Tiêu chí Đánh giá Thống nhất (Metrics)
* **Metric chính:** **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve).
* **Metric phụ:** **PR-AUC** (Area Under the Precision-Recall Curve).
* **Module tính điểm chung:** `metrics.py` (cung cấp hàm `evaluate_predictions(y_true, y_pred_proba)`).
