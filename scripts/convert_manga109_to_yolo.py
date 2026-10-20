#!/usr/bin/env python3
"""
scripts/convert_manga109_to_yolo.py
Script chuyển đổi định dạng annotation XML của tập dữ liệu Manga109 sang định dạng nhãn chuẩn của YOLOv11.

Quy tắc chuyển đổi:
- Chỉ lọc lấy thẻ <text>, gán class_id = 0.
- Bỏ qua các thẻ khác (<face>, <body>, <frame>).
- Tọa độ bbox [xmin, ymin, xmax, ymax] được chuẩn hóa thành [x_center, y_center, width, height]
  trong khoảng [0, 1].
- Phân chia tập dữ liệu thành train / val theo tỷ lệ chỉ định.
- Tổ chức thư mục output chuẩn theo Ultralytics YOLO:
    data/processed/
        images/train/
        images/val/
        labels/train/
        labels/val/
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Sử dụng lxml để phân tích cú pháp XML nhanh và chuẩn xác
try:
    from lxml import etree
except ImportError:
    import xml.etree.ElementTree as etree

from tqdm import tqdm


def convert_to_yolo_format(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_width: float,
    img_height: float,
    class_id: int = 0
) -> Tuple[int, float, float, float, float]:
    """
    Chuyển đổi tọa độ bounding box tuyệt đối (xmin, ymin, xmax, ymax)
    sang tọa độ chuẩn hóa YOLO (class_id, x_center, y_center, width, height).

    Args:
        xmin: Tọa độ x nhỏ nhất của bounding box
        ymin: Tọa độ y nhỏ nhất của bounding box
        xmax: Tọa độ x lớn nhất của bounding box
        ymax: Tọa độ y lớn nhất của bounding box
        img_width: Chiều rộng của trang truyện (pixel)
        img_height: Chiều cao của trang truyện (pixel)
        class_id: ID nhãn lớp (mặc định 0 cho text)

    Returns:
        Tuple (class_id, x_center, y_center, width, height) đã được chuẩn hóa trong đoạn [0, 1].
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Kích thước ảnh không hợp lệ: width={img_width}, height={img_height}")

    # Đảm bảo xmin <= xmax và ymin <= ymax
    x_min_val = min(xmin, xmax)
    x_max_val = max(xmin, xmax)
    y_min_val = min(ymin, ymax)
    y_max_val = max(ymin, ymax)

    # Giới hạn tọa độ trong biên ảnh
    x_min_val = max(0.0, min(float(img_width), float(x_min_val)))
    x_max_val = max(0.0, min(float(img_width), float(x_max_val)))
    y_min_val = max(0.0, min(float(img_height), float(y_min_val)))
    y_max_val = max(0.0, min(float(img_height), float(y_max_val)))

    # Tính toán tọa độ tâm và kích thước
    box_w = x_max_val - x_min_val
    box_h = y_max_val - y_min_val

    x_center = (x_min_val + (box_w / 2.0)) / img_width
    y_center = (y_min_val + (box_h / 2.0)) / img_height
    norm_w = box_w / img_width
    norm_h = box_h / img_height

    # Đảm bảo không vượt quá biên [0.0, 1.0]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return (class_id, x_center, y_center, norm_w, norm_h)


def parse_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """
    Đọc và phân tích file XML annotation của Manga109 cho một tựa truyện (book).
    Chỉ trích xuất các thẻ <text>, bỏ qua thẻ <face>, <body>, <frame>.

    Args:
        xml_path: Đường dẫn đến file XML (ví dụ: annotations/ARMS.xml)

    Returns:
        Danh sách các dictionary cho mỗi trang:
        [
            {
                "page_index": 0,
                "width": 1654,
                "height": 1170,
                "texts": [
                    (class_id, x_center, y_center, width, height),
                    ...
                ]
            },
            ...
        ]
    """
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    pages_data = []
    pages_node = root.find("pages")
    if pages_node is None:
        # Trường hợp root chính là thẻ pages hoặc chứa trực tiếp thẻ page
        pages_elements = root.findall("page")
    else:
        pages_elements = pages_node.findall("page")

    for page in pages_elements:
        page_index = int(page.get("index", 0))
        width = float(page.get("width", 0))
        height = float(page.get("height", 0))

        if width <= 0 or height <= 0:
            continue

        text_boxes: List[Tuple[int, float, float, float, float]] = []

        # CHỈ lọc và lấy thẻ <text>, bỏ qua các thẻ <face>, <body>, <frame>
        for text in page.findall("text"):
            try:
                xmin = float(text.get("xmin"))
                ymin = float(text.get("ymin"))
                xmax = float(text.get("xmax"))
                ymax = float(text.get("ymax"))

                # Bỏ qua các box rỗng hoặc có kích thước = 0
                if xmax <= xmin or ymax <= ymin:
                    continue

                yolo_bbox = convert_to_yolo_format(
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    img_width=width,
                    img_height=height,
                    class_id=0
                )
                text_boxes.append(yolo_bbox)
            except (ValueError, TypeError):
                continue

        pages_data.append({
            "page_index": page_index,
            "width": width,
            "height": height,
            "texts": text_boxes
        })

    return pages_data


def find_manga109_directories(raw_dir: Path) -> Tuple[Path, Path]:
    """
    Tự động tìm kiếm thư mục annotations và thư mục images bên trong raw_dir.
    Hỗ trợ cấu trúc giải nén trực tiếp hoặc lồng thư mục (Manga109s/...).
    """
    # Tìm kiếm thư mục annotations
    candidates_annotations = list(raw_dir.glob("**/annotations"))
    candidates_images = list(raw_dir.glob("**/images"))

    if not candidates_annotations:
        raise FileNotFoundError(f"Không tìm thấy thư mục 'annotations' trong {raw_dir}")
    if not candidates_images:
        raise FileNotFoundError(f"Không tìm thấy thư mục 'images' trong {raw_dir}")

    annotations_dir = candidates_annotations[0]
    images_dir = candidates_images[0]

    return annotations_dir, images_dir


def convert_dataset(
    raw_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    copy_images: bool = True
):
    """
    Toàn bộ quy trình xử lý chuyển đổi dataset Manga109 sang định dạng chuẩn YOLO:
    1. Quét toàn bộ file XML.
    2. Phân chia danh sách truyện thành tập train và tập val (tránh rò rỉ dữ liệu).
    3. Trích xuất text annotations và ghi file .txt.
    4. Sao chép ảnh trang truyện vào data/processed/images/{train,val}.
    """
    random.seed(seed)
    annotations_dir, images_dir = find_manga109_directories(raw_dir)

    print(f"[*] Thư mục annotations: {annotations_dir}")
    print(f"[*] Thư mục images: {images_dir}")
    print(f"[*] Thư mục output: {output_dir}")

    # Tạo cấu trúc thư mục YOLO
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách toàn bộ file XML (mỗi file đại diện cho 1 tựa truyện)
    xml_files = sorted(list(annotations_dir.glob("*.xml")))
    if not xml_files:
        raise FileNotFoundError(f"Không tìm thấy file .xml nào trong {annotations_dir}")

    print(f"[*] Tìm thấy tổng cộng {len(xml_files)} tựa truyện trong Manga109.")

    # Phân chia train/val theo tựa truyện để đảm bảo mô hình không học vẹt phong cách tác giả
    shuffled_xmls = xml_files.copy()
    random.shuffle(shuffled_xmls)

    val_count = max(1, int(len(shuffled_xmls) * val_ratio))
    val_xmls = set(shuffled_xmls[:val_count])
    train_xmls = set(shuffled_xmls[val_count:])

    print(f"[*] Phân chia tập dữ liệu:")
    print(f"    - Tập Huấn luyện (Train): {len(train_xmls)} tựa truyện")
    print(f"    - Tập Kiểm định (Val):   {len(val_xmls)} tựa truyện (Tỷ lệ: {val_ratio*100:.1f}%)")

    total_images_processed = 0
    total_labels_written = 0
    total_text_boxes = 0

    for xml_file in tqdm(xml_files, desc="Chuyển đổi dữ liệu Manga109 sang YOLO", unit="book"):
        book_title = xml_file.stem
        is_val = xml_file in val_xmls
        split_name = "val" if is_val else "train"

        dest_img_dir = val_img_dir if is_val else train_img_dir
        dest_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        # Đọc dữ liệu từ file XML
        pages = parse_xml(xml_file)

        # Thư mục chứa ảnh gốc của truyện này
        book_img_dir = images_dir / book_title

        for page in pages:
            idx = page["page_index"]
            texts = page["texts"]

            # Tên file ảnh tương ứng (thường dạng 000.jpg, 001.jpg, ...)
            candidate_img_names = [f"{idx:03d}.jpg", f"{idx:03d}.png", f"{idx:04d}.jpg", f"{idx}.jpg"]
            src_img_path = None

            for name in candidate_img_names:
                p = book_img_dir / name
                if p.exists():
                    src_img_path = p
                    break

            if src_img_path is None:
                # Nếu không tìm thấy file ảnh tương ứng, tiếp tục trang tiếp theo
                continue

            # Đặt tên thống nhất có tiền tố book_title để không bị trùng tên giữa các truyện
            unified_stem = f"{book_title}_{idx:03d}"
            dest_img_path = dest_img_dir / f"{unified_stem}{src_img_path.suffix}"
            dest_lbl_path = dest_lbl_dir / f"{unified_stem}.txt"

            # 1. Sao chép file ảnh (thử hardlink trước để tiết kiệm dung lượng đĩa và siêu nhanh)
            if copy_images and not dest_img_path.exists():
                try:
                    os.link(src_img_path, dest_img_path)
                except Exception:
                    shutil.copy2(src_img_path, dest_img_path)

            # 2. Ghi file nhãn .txt chuẩn YOLO
            # Ghi cả trang không có text (file txt rỗng) để YOLO học ảnh âm bản (negative samples)
            with open(dest_lbl_path, "w", encoding="utf-8") as f:
                for bbox in texts:
                    cls_id, xc, yc, w, h = bbox
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            total_images_processed += 1
            total_labels_written += 1
            total_text_boxes += len(texts)

    print("\n" + "=" * 50)
    print("[✓] CHUYỂN ĐỔI HOÀN TẤT THÀNH CÔNG!")
    print(f"    - Tổng số trang ảnh đã xử lý: {total_images_processed}")
    print(f"    - Tổng số file nhãn .txt:      {total_labels_written}")
    print(f"    - Tổng số vùng chữ (text):     {total_text_boxes}")
    print(f"    - Dữ liệu lưu tại:             {output_dir.resolve()}")
    print("=" * 50 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chuyển đổi dataset Manga109 sang định dạng nhãn YOLOv11 Text Detection"
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(repo_root / "data" / "raw"),
        help="Đường dẫn đến thư mục chứa dữ liệu Manga109 gốc (mặc định: data/raw)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "data" / "processed"),
        help="Đường dẫn đến thư mục chứa dữ liệu YOLO đầu ra (mặc định: data/processed)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Tỷ lệ tập validation (mặc định: 0.2 tương đương 20%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed để chia tập train/val nhất quán"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Chỉ tạo file nhãn .txt mà không sao chép ảnh"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    try:
        convert_dataset(
            raw_dir=raw_dir,
            output_dir=output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            copy_images=not args.no_copy
        )
    except Exception as e:
        print(f"[!] Lỗi chuyển đổi dữ liệu: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
