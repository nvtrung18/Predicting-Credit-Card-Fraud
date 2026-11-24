# Nhánh DL — báo cáo cuối

- Mô hình cuối: MLP baseline
- Validation seed 9: ROC-AUC 0.944622; Average Precision 0.484340.
- Huấn luyện lại trên 699,960 dòng train trong 9 epoch.
- Test: 299,984 dòng; submission: 299,984 dòng, cột id/is_fraud.
- Xác suất submission nằm trong [0.000000, 0.989707].
- Không dùng test labels; không dùng leaderboard để chọn mô hình.
- Giới hạn: so sánh dựa trên một holdout validation chung; không chạy cross-validation.