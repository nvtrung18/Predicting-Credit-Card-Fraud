# Predicting Credit Card Fraud — Machine Learning Branch

Giải pháp nhánh **Machine Learning** cho bài toán phát hiện gian lận thẻ tín dụng ([Kaggle Playground Series S6E11](https://www.kaggle.com/competitions/playground-series-s6e11)).  
Thước đo đánh giá chính: **ROC-AUC** (phụ: **PR-AUC**).

---

## Cấu trúc mã nguồn

| File | Mô tả chi tiết |
| :--- | :--- |
| `create_split.py` | Chia tập `train.csv` (699,960 dòng) theo tỷ lệ Stratified 80/20 (`seed=9`) để làm chuẩn đối sánh chung. |
| `metrics.py` | Module tự cài đặt hàm tính **ROC-AUC** và **PR-AUC**. |
| `data_profile.md` | Báo cáo phân tích dữ liệu (phân phối lớp, kiểu dữ liệu, tỷ lệ gian lận `1.67%`). |
| `baseline.py` | **Task 1.1:** Mô hình đối chuẩn ban đầu (Logistic Regression & LightGBM mặc định). |
| `task_1_2_fe.py` | **Task 1.2:** Kỹ nghệ đặc trưng (7 biến hành vi tài chính) & xử lý mất cân bằng (`scale_pos_weight`). |
| `task_1_3_kfold.py` | **Task 1.3:** Tối ưu siêu tham số LightGBM và đánh giá chéo **Stratified 5-Fold CV** (`seed=9`). |
| `task_1_4_shallow_ensemble.py` | **Task 1.4:** Kiến trúc **Shallow Tree Ensemble** (3 LightGBM cây nông + Spline LR + XGBoost) kết hợp Rank Blending. |
| `task_1_5_decimal_fe.py` | **Task 1.5:** Mở rộng `task_1_4_shallow_ensemble.py` với đặc trưng phần dư thập phân (Decimal Artifacts) và tương tác phi tuyến. |
| `run_co_training.py` | **Học chéo (Co-Training):** Kết hợp nhãn đồng thuận độ tin cậy cao giữa ML và Deep Learning trên tập Test. |

---

## Kết quả thực nghiệm

| Giai đoạn / File | Phương pháp chính | OOF ROC-AUC | Kaggle Public Score |
| :--- | :--- | :---: | :---: |
| `baseline.py` | Logistic Regression + LightGBM Baseline | — | `0.94592` |
| `task_1_2_fe.py` | Feature Engineering (7 features) + Class Weight | — | `0.94596` |
| `task_1_3_kfold.py` | LightGBM Tuned (`num_leaves=45`, 5-Fold) | `0.94509` | `0.94697` |
| `task_1_4_shallow_ensemble.py` | Shallow Tree Ensemble (5 models + Rank Blend) | `0.94762` | `0.94812` |
| **`task_1_5_decimal_fe.py`** | **Shallow Tree Ensemble + Decimal Features** | **`0.94792`** | **`0.94918`** 🏆 |
| `run_co_training.py` | Co-Training (ML + DL Consensus Labels) | `0.94785` | `0.94795` |

---

## Hướng dẫn chạy lại

1. Cài đặt thư viện:
```bash
pip install numpy pandas scikit-learn lightgbm xgboost scipy
```

2. Tải dữ liệu từ Kaggle và đặt `train.csv`, `test.csv` vào thư mục `data/`.

3. Chạy các script theo thứ tự hoặc chạy trực tiếp mô hình tốt nhất:
```bash
python create_split.py
python baseline.py
python task_1_2_fe.py
python task_1_3_kfold.py
python task_1_4_shallow_ensemble.py
python task_1_5_decimal_fe.py
```
