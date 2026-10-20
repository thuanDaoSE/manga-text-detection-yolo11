# 📖 AI Manga & Comic Text Bubble Detection (YOLO11n vs YOLOv8n)

Hệ thống thị giác máy tính nhận diện vùng bóng thoại và chữ truyện tranh Nhật Bản (Manga / Comic Text Detection) dựa trên kiến trúc **YOLO11n (Nano)** của Ultralytics, được tinh chỉnh chuyên sâu trên bộ dữ liệu chuẩn học thuật **Manga109-s**.

Dự án bao gồm toàn bộ quy trình: thu thập & tiền xử lý dữ liệu, huấn luyện mô hình, thực nghiệm so sánh đối chứng với **YOLOv8n**, và triển khai Web Demo trực tuyến trên **Hugging Face Spaces**.

---

## 🌐 1. Live Web Demo (Hugging Face Spaces)

Trải nghiệm trực tiếp mô hình mà không cần cài đặt tại:
👉 **[Hugging Face Space: daotthuanSE/manga-text-detection-yolo11n](https://huggingface.co/spaces/daotthuanSE/manga-text-detection-yolo11n)**

* **Phần cứng:** ZeroGPU (NVIDIA A10G) / CPU.
* **Thời gian phản hồi:** ~3 - 5 ms / trang ảnh.
* **Tính năng:** Tải ảnh tùy ý, thử ảnh mẫu 1-click, trích xuất bảng tọa độ Bounding Box `(X, Y, W, H)`, và xem trực tiếp các biểu đồ đánh giá thực nghiệm.

---

## 🏆 2. Bảng So Sánh Đối Chứng Thực Nghiệm (Benchmark Results)

Cả hai mô hình đều được huấn luyện và kiểm thử độc lập trong cùng điều kiện thực nghiệm trên **1.277 trang truyện** thuộc tập kiểm định (Manga109-s) với GPU **NVIDIA Tesla T4 (Kaggle)** qua 30 Epochs:

| Tiêu chí so sánh | YOLOv8n (Mô hình Baseline) | YOLO11n (Mô hình Đề xuất) 🏆 | Đánh giá & Ý nghĩa kỹ thuật |
|---|:---:|:---:|---|
| **mAP@50 (%)** | **98.42%** | **98.22%** | Cả hai đều đạt độ chính xác cực đại (~98.2% - 98.4%) |
| **mAP@50-95 (%)** | **86.80%** | **85.91%** | Tương đương nhau, viền bounding box ôm khít đường chữ |
| **Precision (%)** | **97.03%** | **96.95%** | Tỷ lệ bắt nhầm nét vẽ cực thấp (< 3.1%) |
| **Recall (%)** | **94.57%** | **94.16%** | Độ bao quát bắt trọn bóng thoại đạt trên 94% |
| **Số tham số (Parameters)** | 3.13 Triệu (3.13M) | **2.59 Triệu (2.59M)** | **YOLO11n nhẹ hơn 17.2%** (tiết kiệm ~535.000 tham số) |
| **Khối lượng tính toán (FLOPs)** | 8.6 GFLOPs | **6.4 GFLOPs** | **YOLO11n giảm tới 25.6% chi phí tính toán** |
| **Tốc độ suy luận (Inference)** | ~2.79 ms | **~2.98 ms** | Siêu nhanh (~330 FPS trên GPU T4, ~40 ms trên CPU) |
| **Kích thước file trọng số** | 6.5 MB | **5.2 MB** | Siêu nhẹ, dễ dàng nhúng vào thiết bị biên |

![Biểu đồ so sánh đối chứng](comparison_chart.png)

> 💡 **Luận điểm khoa học cốt lõi:**  
> Trên tập dữ liệu Manga109-s, mAP@50 > 98.2% đã tiệm cận mức bão hòa (Ceiling Accuracy) của bài toán. Điểm cải tiến mang tính đột phá của **YOLO11n** nằm ở việc **giảm tới 25.6% chi phí tính toán (FLOPs)** và **tiết kiệm 17.2% bộ nhớ tham số**. Nhờ kiến trúc khối C3k2 và Attention C2PSA, YOLO11n tối ưu vượt trội khi triển khai thực tế trên CPU văn phòng, Mobile App hoặc Web Extension dịch truyện mà không tốn chi phí thuê server GPU đắt đỏ.

---

## 📁 3. Cấu Trúc Thư Mục Dự Án

```
├── hf_manga_space/           # Mã nguồn triển khai Web Demo trên Hugging Face Spaces
│   ├── assets/               # Chứa các biểu đồ đánh giá thực nghiệm
│   ├── examples/             # Ảnh mẫu kiểm thử
│   ├── app.py                # Ứng dụng Gradio Web UI đa chức năng (3 Tabs)
│   ├── best.pt               # Trọng số tối ưu YOLO11n (5.2 MB)
│   └── requirements.txt      # Thư viện cho Space (khóa pydantic<=2.10.6)
├── training_results/         # Toàn bộ biểu đồ & log quá trình huấn luyện
│   ├── results.png           # Biểu đồ diễn biến Loss & Metrics qua 30 Epochs
│   ├── BoxPR_curve.png       # Đường cong Precision-Recall (AUC = 0.982)
│   ├── confusion_matrix_normalized.png  # Ma trận nhầm lẫn chuẩn hóa
│   └── args.yaml             # Bộ siêu tham số huấn luyện (Hyperparameters)
├── notebooks/
│   └── train_yolov11.ipynb   # Jupyter Notebook huấn luyện trên Cloud GPU
├── scripts/
│   ├── full_pipeline.py      # Tự động hóa tải, parse nhãn và train
│   ├── download_data.py      # Script tải dataset Manga109 từ Hugging Face
│   ├── convert_manga109_to_yolo.py  # Chuyển đổi nhãn XML sang YOLO format
│   └── prepare_data.py       # Phân chia tập Train/Val (85/15)
├── best.pt                   # File trọng số tốt nhất YOLO11n (5.2 MB)
├── comparison_chart.png      # Biểu đồ cột so sánh đối chứng YOLO11n vs YOLOv8n
├── model_comparison_results.csv # Bảng dữ liệu số đo so sánh thực nghiệm
├── notebookcdfb972e6e.ipynb  # Notebook thực nghiệm đối chứng hoàn chỉnh trên Kaggle
├── manga_text.yaml           # Cấu hình Ultralytics dataset
├── requirements.txt          # Danh sách thư viện Python
└── README.md                 # Tài liệu hướng dẫn & báo cáo dự án
```

---

## 🛠️ 4. Hướng Dẫn Cài Đặt & Sử Dụng Nhanh (Quickstart)

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
git clone https://github.com/thuanDaoSE/manga-text-detection-yolo11.git
cd manga-text-detection-yolo11
pip install -r requirements.txt
```

### 2. Chạy suy luận (Inference) trên ảnh manga bất kỳ
File trọng số đã huấn luyện `best.pt` (5.2 MB) đã được đính kèm sẵn trong repository:

```python
from ultralytics import YOLO

# 1. Nạp mô hình đã huấn luyện
model = YOLO("best.pt")

# 2. Dự đoán trên trang truyện manga
results = model.predict(
    source="path/to/manga_page.jpg",
    conf=0.25,
    iou=0.45,
    save=True
)

# 3. Trích xuất thông tin các ô thoại
for box in results[0].boxes:
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].tolist()
    print(f"Độ tin cậy: {conf:.2%}, Tọa độ: {xyxy}")
```

### 3. Chạy giao diện Web Demo cục bộ (Local Gradio Web App)
```bash
python hf_manga_space/app.py
```
Sau đó truy cập vào `http://127.0.0.1:7860` trên trình duyệt.

---

## 📊 5. Thông Tin Huấn Luyện (Training Setup)

* **Tập dữ liệu:** Manga109-s (Đại học Tokyo) — 8.519 trang ảnh, 123.212 vùng nhãn bóng thoại.
  * Tập Train: 7.242 trang (85%).
  * Tập Val: 1.277 trang (15%).
* **Số Epochs:** 30 Epochs (hội tụ hoàn toàn).
* **Kích thước ảnh đầu vào:** 640 × 640.
* **Batch size:** 16 (tối ưu hóa AMP FP16).
* **Phần cứng:** NVIDIA Tesla T4 GPU (Kaggle) / Cloud GPU.

---

## 👨‍💻 Tác Giả & Bản Quyền

* **Tác giả:** Đào Đặng Thuận ([thuanDaoSE](https://github.com/thuanDaoSE))
* **Mã nguồn:** Giấy phép mã nguồn mở Apache-2.0.
