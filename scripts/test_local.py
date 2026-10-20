import os
import glob
from ultralytics import YOLO

# 1. Đường dẫn file weights tải về máy local
# Đặt file best.pt cùng thư mục gốc hoặc chỉ định đường dẫn tuyệt đối
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    # Tìm kiếm tự động nếu để ở thư mục cha hoặc thư mục hiện tại
    if os.path.exists(os.path.join("..", "best.pt")):
        MODEL_PATH = os.path.join("..", "best.pt")

print(f"--> Đang load model từ: {os.path.abspath(MODEL_PATH)}")
model = YOLO(MODEL_PATH)

# 2. Tìm ảnh test (bạn có thể thay bằng đường dẫn file ảnh bất kỳ)
# Ví dụ: image_path = "path/to/your/manga_page.jpg"
image_candidates = glob.glob("*.jpg") + glob.glob("*.png") + glob.glob("../data/*.jpg")

if not image_candidates:
    print("\n[CHÚ Ý] Chưa có ảnh test trong thư mục.")
    print("Bạn hãy copy 1 file ảnh truyện tranh (ví dụ: test.jpg) vào thư mục này để chạy nhé!")
else:
    test_img = image_candidates[0]
    print(f"--> Đang chạy test trên ảnh: {test_img}")

    # 3. Chạy dự đoán phát hiện chữ
    results = model.predict(
        source=test_img,
        conf=0.25,        # Ngưỡng tin cậy nhận diện text
        save=True,        # Tự động vẽ bounding box và lưu ảnh
        project="runs",
        name="test_result",
        exist_ok=True
    )

    boxes = results[0].boxes
    print(f"\n✅ THÀNH CÔNG: Đã phát hiện {len(boxes)} vùng chữ (text bubble)!")
    print(f"📁 Ảnh kết quả đã được vẽ khung lưu tại: runs/test_result/")
