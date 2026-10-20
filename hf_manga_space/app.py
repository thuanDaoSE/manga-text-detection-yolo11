import os
import time
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Hỗ trợ ZeroGPU trên Hugging Face Spaces
try:
    import spaces
    gpu_decorator = spaces.GPU
    print("ZeroGPU environment detected. Enabling @spaces.GPU acceleration!")
except Exception:
    def gpu_decorator(fn):
        return fn

# 1. Khởi tạo mô hình
MODEL_PATH = "best.pt"
print(f"Loading YOLO11n weights from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

@gpu_decorator
def detect_text(image, conf_thresh, iou_thresh):
    if image is None:
        return None, "⚠️ Vui lòng tải lên một trang truyện để kiểm tra!", []
    
    start_time = time.time()
    
    # Dự đoán bằng YOLO11n
    results = model.predict(
        source=image,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False
    )
    
    elapsed_ms = (time.time() - start_time) * 1000
    res = results[0]
    
    # Lấy ảnh đã vẽ bounding box
    res_bgr = res.plot(line_width=2, font_size=1)
    res_rgb = Image.fromarray(res_bgr[..., ::-1])
    
    # Trích xuất thông tin bounding box
    boxes_data = []
    for idx, box in enumerate(res.boxes):
        conf = float(box.conf[0])
        xyxy = [int(x) for x in box.xyxy[0]]
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        boxes_data.append([idx + 1, f"{conf:.1%}", f"{xyxy[0]}, {xyxy[1]}", f"{w} × {h}"])
        
    # Tốc độ suy luận thuần túy từ YOLO model
    inf_ms = getattr(res, "speed", {}).get("inference", 4.5)
    
    num_boxes = len(res.boxes)
    summary_md = f"""
### 📊 Kết quả phân tích:
* **Số vùng chữ phát hiện được:** <span style="color:#10b981;font-weight:bold;font-size:18px;">{num_boxes} vùng</span>
* **Tốc độ suy luận mạng (Model Inference):** `{inf_ms:.1f} ms` *(~{int(1000/max(inf_ms, 0.1))} FPS)*
* **Kích thước ảnh:** `{image.width} × {image.height}` px
* **Độ tin cậy trung bình:** `{np.mean([float(b.conf[0]) for b in res.boxes]):.1%}` *(nếu có)*
""" if num_boxes > 0 else f"""
### 📊 Kết quả phân tích:
* **Không tìm thấy vùng chữ nào** với ngưỡng tin cậy `{conf_thresh:.2f}`.
* *(Gợi ý: Thử hạ thanh trượt Ngưỡng tin cậy xuống 0.10 - 0.15 nếu chữ mờ hoặc nghệ thuật).*
"""

    return res_rgb, summary_md, boxes_data


# Danh sách ảnh ví dụ (hỗ trợ cả ở thư mục gốc lẫn trong examples/)
candidate_examples = [
    "test.jpg", "test2.jpg", "test9.jpg",
    "examples/test.jpg", "examples/test2.jpg", "examples/test9.jpg"
]
example_files = []
seen_names = set()
for p in candidate_examples:
    if os.path.exists(p):
        fname = os.path.basename(p)
        if fname not in seen_names:
            seen_names.add(fname)
            example_files.append(p)

def get_chart(name):
    for path in [f"assets/{name}", name]:
        if os.path.exists(path):
            return path
    return None

# Giao diện Gradio hiện đại
custom_css = """
#header { text-align: center; margin-bottom: 1.5rem; }
#summary_box { padding: 12px; background: rgba(99, 102, 241, 0.05); border-radius: 8px; border: 1px solid rgba(99, 102, 241, 0.2); }
"""


with gr.Blocks(title="AI Manga Text Bubble Detector (YOLO11n)", css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown("""
# 📖 AI Manga & Comic Text Bubble Detection
### Mô hình YOLO11n nhận diện bóng thoại & chữ truyện tranh đạt độ chính xác **mAP@50 = 98.19%**
Huấn luyện trên tập dữ liệu chuẩn **Manga109-s** (8.519 trang ảnh, 123.000+ nhãn chữ).
""")

    with gr.Tabs():
        with gr.TabItem("🚀 Nhận diện trực tuyến (Live Demo)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📥 Trang truyện đầu vào")
                    input_image = gr.Image(type="pil", show_label=False)
                    
                    with gr.Accordion("⚙️ Cấu hình nâng cao", open=False):
                        conf_slider = gr.Slider(
                            minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                            label="Ngưỡng tin cậy (Confidence Threshold)"
                        )
                        iou_slider = gr.Slider(
                            minimum=0.10, maximum=0.90, value=0.45, step=0.05,
                            label="Ngưỡng triệt tiêu lặp (IoU NMS Threshold)"
                        )
                        
                    with gr.Row():
                        clear_btn = gr.ClearButton([input_image], value="Làm mới")
                        submit_btn = gr.Button("🚀 Nhận diện ngay", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("#### 🎯 Kết quả nhận diện (Bounding Boxes)")
                    output_image = gr.Image(type="pil", show_label=False)
                    summary_output = gr.Markdown(elem_id="summary_box")
                    
                    with gr.Accordion("📋 Danh sách tọa độ các ô thoại", open=False):
                        box_table = gr.Dataframe(
                            headers=["#", "Độ tin cậy", "Tọa độ (X, Y)", "Kích thước (W × H)"],
                            datatype=["number", "str", "str", "str"],
                            label="Bảng chi tiết ô thoại"
                        )

            if example_files:
                gr.Markdown("### 💡 Bấm vào ảnh mẫu dưới đây để thử nhanh 1-click:")
                gr.Examples(
                    examples=example_files,
                    inputs=input_image,
                    outputs=[output_image, summary_output, box_table],
                    fn=detect_text,
                    cache_examples=False
                )

            submit_btn.click(
                fn=detect_text,
                inputs=[input_image, conf_slider, iou_slider],
                outputs=[output_image, summary_output, box_table]
            )
            
            input_image.change(
                fn=detect_text,
                inputs=[input_image, conf_slider, iou_slider],
                outputs=[output_image, summary_output, box_table]
            )

        with gr.TabItem("📊 Báo cáo đánh giá & Biểu đồ thực nghiệm (Metrics & Charts)"):
            gr.Markdown("""
### 📌 Bảng chỉ số đánh giá thực nghiệm (Validation Metrics)
Đánh giá độc lập trên **1.277 trang truyện** thuộc tập kiểm định (Manga109-s):

| Chỉ số (Metric) | Kết quả thực tế | Đánh giá & Ý nghĩa kỹ thuật |
|---|:---:|---|
| **mAP@50** | **98.19%** (0.9819) | Nhận diện chuẩn xác 98.2% toàn bộ bóng thoại manga |
| **Precision** | **96.84%** (0.9684) | Cực kỳ ít dương tính giả (không nhận nhầm nét vẽ manga thành chữ) |
| **Recall** | **94.21%** (0.9421) | Tỷ lệ bao quát rất cao, hiếm khi bỏ sót các bóng thoại nhỏ hoặc chữ dọc |
| **mAP@50-95** | **86.06%** (0.8606) | Viền bounding box ôm khít chặt chẽ đường bao chữ, tối ưu cho bước Inpainting |
| **Tốc độ suy luận** | **~4.5 ms / trang** | Tương đương ~220 FPS trên GPU, ~40 ms trên CPU thông thường |
| **Kích thước mô hình** | **5.2 MB** | Siêu nhẹ (2.58M params), sẵn sàng đóng gói mobile app / web service |

---
### 📈 Trực quan hóa tiến trình huấn luyện & Kiểm định (30 Epochs)
""")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 1. Quá trình hội tụ (Training Loss & Metrics)")
                    gr.Image(value=get_chart("results.png"), show_label=False)
                with gr.Column():
                    gr.Markdown("#### 2. Đường cong Precision - Recall (mAP@50 = 0.982)")
                    gr.Image(value=get_chart("BoxPR_curve.png"), show_label=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 3. Ma trận nhầm lẫn chuẩn hóa (Confusion Matrix)")
                    gr.Image(value=get_chart("confusion_matrix_normalized.png"), show_label=False)
                with gr.Column():
                    gr.Markdown("#### 4. Kết quả nhận diện thực tế trên tập Val (Manga109-s)")
                    gr.Image(value=get_chart("val_batch0_pred.jpg"), show_label=False)

    gr.Markdown("""
---
<center>
<small>
Developed with ❤️ by <b>daotthuanSE</b> • Powered by Ultralytics YOLO11 & Hugging Face Spaces<br>
Mô hình thuộc giai đoạn 1 của dự án <b>AI Manga Translator</b> (Dịch thuật truyện tranh tự động khép kín).
</small>
</center>
""")

if __name__ == "__main__":
    demo.launch()

