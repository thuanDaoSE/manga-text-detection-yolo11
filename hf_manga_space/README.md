---
title: Manga Text Detection YOLO11n
emoji: 📖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Accurate Manga Text Bubble Detector (YOLO11n)
tags:
- yolo
- yolo11
- manga
- text-detection
- comic
- ultralytics
---

# 📖 Manga Text Bubble Detection (YOLO11n)

Mô hình học sâu phát hiện vùng chữ và bóng thoại truyện tranh Nhật Bản (Manga / Comic Text Detection) dựa trên kiến trúc **YOLO11n (Nano)** của Ultralytics, được tinh chỉnh chuyên sâu trên bộ dữ liệu chuẩn **Manga109-s**.

---

## 📊 Bảng chỉ số đánh giá thực nghiệm (Validation Metrics)

Đánh giá độc lập trên **1.277 trang truyện** trong tập kiểm định của bộ dữ liệu Manga109-s:

| Chỉ số (Metric) | Kết quả đạt được | Ý nghĩa thực tế |
|---|:---:|---|
| **mAP@50** | **98.19%** (0.9819) | Nhận diện chuẩn xác 98.2% toàn bộ các bóng thoại |
| **Precision** | **96.84%** (0.9684) | Cực kỳ ít dương tính giả (không nhận nhầm nét vẽ thành chữ) |
| **Recall** | **94.21%** (0.9421) | Bắt trọn hầu hết các khung thoại, rất hiếm khi bỏ sót |
| **mAP@50-95** | **86.06%** (0.8606) | Viền bounding box ôm khít chặt chẽ đường viền chữ |
| **Tốc độ suy luận** | **~4.5 ms / trang** | Tương đương ~220 FPS trên GPU, siêu nhẹ và nhanh |
| **Kích thước Model** | **5.2 MB** | Chỉ 2.58 triệu tham số, dễ dàng chạy trên CPU máy tính & điện thoại |

---

## 🛠️ Thiết lập huấn luyện (Training Details)

* **Kiến trúc mạng:** YOLO11n (2.58M parameters, 6.4 GFLOPs)
* **Tập dữ liệu:** Manga109-s (8.519 trang ảnh, 123.212 vùng nhãn text)
  * Tập Train: 7.242 trang (85%)
  * Tập Val: 1.277 trang (15%)
* **Số Epochs:** 30 Epochs (hội tụ hoàn toàn)
* **Kích thước ảnh:** 640 × 640
* **Tối ưu hóa:** Cache disk, Early Stopping (patience=10)

---

## 🚀 Hướng dẫn sử dụng trong Python (Ultralytics)

```python
from ultralytics import YOLO

# 1. Load model
model = YOLO("best.pt")

# 2. Dự đoán trên trang truyện bất kỳ
results = model.predict(source="manga_page.jpg", conf=0.25)

# 3. Xem kết quả
for box in results[0].boxes:
    print(f"Conf: {float(box.conf[0]):.2f}, Box (XYXY): {box.xyxy[0].tolist()}")

# Lưu ảnh đã vẽ bounding box
results[0].save(filename="detected_result.jpg")
```

---

## 👨‍💻 Tác giả & Dự án

* **Tác giả:** daotthuanSE
* **Đề tài:** Giai đoạn 1 - Phát hiện vùng văn bản thuộc hệ thống dịch truyện tự động **AI Manga Translator**.
