# BÁO CÁO LAB 16 - PHƯƠNG ÁN CPU (GCP)

**Sinh viên:** Trương Đăng Nghĩa  
**MSSV:** 2A202600437  
**Platform:** Google Cloud Platform (GCP)  
**Instance Type:** n2-highmem-8 (8 vCPU, 64 GB RAM)  
**Region/Zone:** us-west1-b  
**Mã nguồn:** terraform-gcp

---

## Lý do sử dụng phương án CPU thay vì GPU

Do tài khoản GCP mới bị giới hạn quota GPU nghiêm ngặt (`GPUS_ALL_REGIONS = 0`), yêu cầu tăng quota đã bị từ chối (bằng chứng là hình ảnh QuotaRejected.jpeg). Thay vì bỏ qua bài lab, tôi đã chuyển sang triển khai bài toán Machine Learning thực tế (LightGBM - Gradient Boosting) trên instance CPU cao cấp `n2-highmem-8`.

Phương án này vẫn đảm bảo đầy đủ quy trình: **Terraform IaC → Cloud Infrastructure → ML Training → Inference → Billing Check**, chỉ khác là không yêu cầu GPU quota.

---

## Kết quả Benchmark trên n2-highmem-8 (GCP)

**Dataset:** Credit Card Fraud Detection (284,807 giao dịch)

### Hiệu năng Training & Inference:

| Metric | Kết quả | Đánh giá |
|--------|---------|----------|
| **Load time** | 1.41 giây | Rất nhanh |
| **Training time** | 0.92 giây | Xuất sắc (nhờ 8 CPU cores) |
| **Best iteration** | 0 | Không dùng early stopping |
| **AUC-ROC** | 0.7478 | Chấp nhận được cho baseline model |
| **Accuracy** | 99.6% | Xuất sắc |
| **F1-Score** | 0.3652 | Thấp do dataset imbalanced |
| **Precision** | 0.2519 | Thấp (nhiều false positives) |
| **Recall** | 0.6633 | Tốt (phát hiện 66% fraud cases) |
| **Inference latency** | 1.32 ms | Siêu nhanh, phù hợp production |
| **Throughput** | 472,810 rows/sec | Rất cao |

---

## So sánh CPU vs GPU (Phân tích)

### Ưu điểm của phương án CPU:

1. **Triển khai ngay lập tức:** Không cần chờ GPU quota approval (có thể mất 24-48 giờ hoặc bị từ chối)
2. **Chi phí tương đương:** n2-highmem-8 (~$0.48/giờ) vs g4dn.xlarge (~$0.526/giờ)
3. **Training nhanh hơn cho tabular data:** 0.92 giây vs vài phút để load LLM vào VRAM
4. **Inference cực nhanh:** 1.32ms latency, phù hợp real-time applications

### Nhược điểm:

1. **Không phù hợp cho Deep Learning:** Không thể chạy LLM như Gemma-4-E2B-it
2. **AUC-ROC thấp hơn mong đợi:** 0.7478 (lý tưởng > 0.9) do chưa tune hyperparameters
3. **F1-Score thấp:** 0.3652 do dataset imbalanced và không xử lý class weights

---

