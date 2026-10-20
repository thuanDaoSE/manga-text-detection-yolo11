#!/usr/bin/env python3
"""
scripts/prepare_data.py
=======================
Script xử lý dữ liệu Manga109 sang định dạng nhãn chuẩn YOLO.

Chức năng chính:
  1. Đọc và phân tích file XML annotation của Manga109.
  2. Chuyển đổi tọa độ bbox tuyệt đối [xmin, ymin, xmax, ymax]
     sang tọa độ YOLO chuẩn hóa [x_center, y_center, w, h] (khoảng [0, 1]).
  3. Tự động tạo cấu trúc thư mục chuẩn YOLO:
       dataset/
           images/train/
           images/val/
           labels/train/
           labels/val/
  4. Chia dữ liệu ngẫu nhiên theo tỷ lệ 85% Train - 15% Validation.
  5. Sao chép ảnh và ghi file nhãn .txt tương ứng.

Thư viện sử dụng: xml.etree.ElementTree, shutil, pathlib, random (thư viện chuẩn Python).

Cách dùng:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --raw-dir data/raw --output-dir dataset --val-ratio 0.15
"""

import os
import sys
import shutil
import random
import argparse
import xml.etree.ElementTree as ET   # Thư viện chuẩn Python để đọc XML
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# ===========================================================================
# PHẦN 1: HÀM CHUYỂN ĐỔI TỌA ĐỘ SANG ĐỊNH DẠNG YOLO
# ===========================================================================

def convert_bbox_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    page_width: float,
    page_height: float,
    class_id: int = 0
) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Chuyển đổi tọa độ bounding box tuyệt đối sang định dạng YOLO chuẩn hóa.

    Công thức toán học:
      x_center = (xmin + xmax) / 2  --> chuẩn hóa: chia cho page_width
      y_center = (ymin + ymax) / 2  --> chuẩn hóa: chia cho page_height
      w        = xmax - xmin         --> chuẩn hóa: chia cho page_width
      h        = ymax - ymin         --> chuẩn hóa: chia cho page_height

    Tất cả giá trị đầu ra nằm trong khoảng [0.0, 1.0] và được làm tròn 6 chữ số thập phân.

    Tham số:
        xmin, ymin    : Tọa độ góc trên-trái của bounding box (pixel).
        xmax, ymax    : Tọa độ góc dưới-phải của bounding box (pixel).
        page_width    : Chiều rộng của trang ảnh (pixel), lấy từ thuộc tính XML.
        page_height   : Chiều cao của trang ảnh (pixel), lấy từ thuộc tính XML.
        class_id      : ID nhãn lớp đối tượng (mặc định = 0, tương ứng class 'text').

    Trả về:
        Tuple (class_id, x_center, y_center, w, h) đã chuẩn hóa.
        Trả về None nếu bbox không hợp lệ (kích thước = 0 hoặc ảnh có kích thước không hợp lệ).
    """
    # Kiểm tra kích thước ảnh hợp lệ để tránh lỗi chia cho 0
    if page_width <= 0 or page_height <= 0:
        return None

    # Đảm bảo thứ tự min/max đúng (phòng trường hợp XML bị ghi ngược)
    x1 = min(float(xmin), float(xmax))
    x2 = max(float(xmin), float(xmax))
    y1 = min(float(ymin), float(ymax))
    y2 = max(float(ymin), float(ymax))

    # Kẹp tọa độ trong biên ảnh (clip to image boundary)
    x1 = max(0.0, min(x1, page_width))
    x2 = max(0.0, min(x2, page_width))
    y1 = max(0.0, min(y1, page_height))
    y2 = max(0.0, min(y2, page_height))

    # Bỏ qua bbox có chiều rộng hoặc chiều cao bằng 0 (hộp rỗng)
    if x2 <= x1 or y2 <= y1:
        return None

    # Tính toán tọa độ tâm và kích thước theo công thức YOLO
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    # Chuẩn hóa về khoảng [0, 1] bằng cách chia cho kích thước ảnh
    x_center_norm = round(x_center / page_width,  6)
    y_center_norm = round(y_center / page_height, 6)
    w_norm        = round(w        / page_width,  6)
    h_norm        = round(h        / page_height, 6)

    # Đảm bảo kết quả không vượt quá biên [0.0, 1.0]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    w_norm        = max(0.0, min(1.0, w_norm))
    h_norm        = max(0.0, min(1.0, h_norm))

    return (class_id, x_center_norm, y_center_norm, w_norm, h_norm)


# ===========================================================================
# PHẦN 2: HÀM ĐỌC VÀ PHÂN TÍCH FILE XML CỦA MANGA109
# ===========================================================================

def parse_manga109_xml(xml_path: Path) -> List[Dict]:
    """
    Đọc file XML annotation của Manga109 cho một tựa truyện (book).

    Cấu trúc XML Manga109:
        <book title="TenTruyen">
            <pages>
                <page index="0" width="1654" height="1170">
                    <text xmin="100" ymin="200" xmax="300" ymax="400">...</text>
                    <face xmin="..." .../>   <- Bỏ qua thẻ này
                    <body xmin="..." .../>   <- Bỏ qua thẻ này
                    <frame xmin="..." .../>  <- Bỏ qua thẻ này
                </page>
                ...
            </pages>
        </book>

    Tham số:
        xml_path : Đường dẫn (Path object) tới file XML annotation.

    Trả về:
        Danh sách các dict, mỗi dict đại diện cho 1 trang truyện.
    """
    pages_data = []

    try:
        # Dùng xml.etree.ElementTree (thư viện chuẩn, không cần cài thêm)
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  [CANH BAO] Khong the doc XML '{xml_path.name}': {e}")
        return []

    # Tìm tất cả thẻ <page> - hỗ trợ cả hai cấu trúc XML phổ biến:
    #   <book> -> <pages> -> <page>  (cấu trúc chuẩn Manga109)
    #   <book> -> <page>             (cấu trúc đơn giản hóa)
    pages_node = root.find("pages")
    if pages_node is not None:
        page_elements = pages_node.findall("page")
    else:
        page_elements = root.findall("page")

    if not page_elements:
        print(f"  [CANH BAO] Khong tim thay the <page> nao trong '{xml_path.name}'.")
        return []

    for page_elem in page_elements:
        # Đọc thuộc tính của trang: index, width, height
        try:
            page_index  = int(page_elem.get("index", 0))
            page_width  = float(page_elem.get("width", 0))
            page_height = float(page_elem.get("height", 0))
        except (ValueError, TypeError):
            continue  # Bỏ qua trang bị lỗi dữ liệu

        # Bỏ qua trang có kích thước không hợp lệ
        if page_width <= 0 or page_height <= 0:
            continue

        yolo_labels = []

        # Chỉ xử lý thẻ <text>, bỏ qua <face>, <body>, <frame>
        for text_elem in page_elem.findall("text"):
            try:
                xmin = float(text_elem.get("xmin"))
                ymin = float(text_elem.get("ymin"))
                xmax = float(text_elem.get("xmax"))
                ymax = float(text_elem.get("ymax"))
            except (ValueError, TypeError, AttributeError):
                # Bỏ qua thẻ bị thiếu thuộc tính tọa độ
                continue

            # Chuyển đổi sang định dạng YOLO (class_id mặc định = 0)
            yolo_bbox = convert_bbox_to_yolo(
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                page_width=page_width, page_height=page_height,
                class_id=0
            )

            if yolo_bbox is not None:
                yolo_labels.append(yolo_bbox)

        # Thu thập dữ liệu của trang (kể cả trang không có text - negative sample)
        pages_data.append({
            "page_index": page_index,
            "width":      page_width,
            "height":     page_height,
            "yolo_labels": yolo_labels
        })

    return pages_data


# ===========================================================================
# PHẦN 3: HÀM TIỆN ÍCH - TÌM FILE ẢNH TƯƠNG ỨNG
# ===========================================================================

def _find_image_file(images_dir: Path, book_title: str, page_index: int) -> Optional[Path]:
    """
    Tìm file ảnh tương ứng với một trang truyện trong thư mục images.

    Manga109 lưu ảnh theo nhiều quy ước đặt tên khác nhau:
      - images/TenTruyen/000.jpg  (phổ biến nhất)
      - images/TenTruyen/0000.jpg (4 chữ số)
      - images/TenTruyen/0.jpg    (không padding)
      - Ảnh PNG thay vì JPG

    Tham số:
        images_dir  : Thư mục gốc chứa ảnh (có subfolder theo tên truyện).
        book_title  : Tên tựa truyện (tên thư mục con trong images_dir).
        page_index  : Chỉ số trang (lấy từ thuộc tính 'index' trong XML).

    Trả về:
        Path đến file ảnh tìm thấy đầu tiên, hoặc None nếu không tìm thấy.
    """
    book_img_dir = images_dir / book_title

    if not book_img_dir.is_dir():
        return None

    # Các quy ước đặt tên file ảnh trong Manga109
    candidate_names = [
        f"{page_index:03d}.jpg",    # 000.jpg, 001.jpg, ... (phổ biến nhất)
        f"{page_index:03d}.png",    # 000.png
        f"{page_index:04d}.jpg",    # 0000.jpg
        f"{page_index:04d}.png",    # 0000.png
        f"{page_index}.jpg",        # 0.jpg (không padding)
        f"{page_index}.png",        # 0.png
    ]

    for name in candidate_names:
        candidate = book_img_dir / name
        if candidate.exists():
            return candidate

    return None  # Không tìm thấy file ảnh nào phù hợp


# ===========================================================================
# PHẦN 4: HÀM TỰ ĐỘNG TÌM THƯ MỤC MANGA109
# ===========================================================================

def find_manga109_source_dirs(raw_dir: Path) -> Tuple[Path, Path]:
    """
    Tự động dò tìm thư mục 'annotations' và 'images' bên trong raw_dir.
    Hỗ trợ cả trường hợp giải nén trực tiếp hoặc lồng thư mục.

    Tham số:
        raw_dir : Thư mục gốc chứa dữ liệu Manga109 tải về.

    Trả về:
        Tuple (annotations_dir, images_dir) - đường dẫn tuyệt đối đến 2 thư mục.

    Ném ra:
        FileNotFoundError nếu không tìm thấy thư mục 'annotations' hoặc 'images'.
    """
    # Tìm thư mục 'annotations' ở mọi cấp thư mục con
    annotation_candidates = list(raw_dir.glob("**/annotations"))
    # Tìm thư mục 'images' ở mọi cấp thư mục con
    image_candidates = list(raw_dir.glob("**/images"))

    if not annotation_candidates:
        raise FileNotFoundError(
            f"Khong tim thay thu muc 'annotations' trong '{raw_dir}'.\n"
            f"Hay dam bao ban da giai nen Manga109 vao dung thu muc nay."
        )
    if not image_candidates:
        raise FileNotFoundError(
            f"Khong tim thay thu muc 'images' trong '{raw_dir}'.\n"
            f"Hay dam bao ban da giai nen Manga109 vao dung thu muc nay."
        )

    # Lấy thư mục nông nhất (ít lồng nhất) trong cây thư mục
    annotations_dir = sorted(annotation_candidates, key=lambda p: len(p.parts))[0]
    images_dir      = sorted(image_candidates,      key=lambda p: len(p.parts))[0]

    return annotations_dir, images_dir


# ===========================================================================
# PHẦN 5: HÀM CHÍNH - XỬ LÝ TOÀN BỘ DATASET
# ===========================================================================

def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.15,
    seed: int = 42,
    copy_images: bool = True
):
    """
    Toàn bộ quy trình xử lý và chia tập dữ liệu Manga109 sang định dạng YOLO.

    Quy trình:
        1. Tìm thư mục annotations và images trong raw_dir.
        2. Đọc tất cả file XML, lấy danh sách trang truyện.
        3. Xáo trộn ngẫu nhiên (random.shuffle) danh sách trang, chia 85/15.
        4. Sao chép ảnh vào dataset/images/{train,val}.
        5. Ghi file nhãn .txt vào dataset/labels/{train,val}.

    Tham số:
        raw_dir     : Thư mục gốc chứa dữ liệu Manga109 gốc.
        output_dir  : Thư mục đầu ra chứa cấu trúc YOLO (thường là 'dataset/').
        val_ratio   : Tỷ lệ tập Validation (mặc định 0.15 = 15%).
        seed        : Hạt giống ngẫu nhiên để đảm bảo kết quả chia tập nhất quán.
        copy_images : True = sao chép ảnh; False = chỉ tạo file nhãn .txt.
    """
    # -----------------------------------------------------------------------
    # BƯỚC 1: Thiết lập random seed và tìm thư mục nguồn
    # -----------------------------------------------------------------------
    random.seed(seed)
    print("=" * 60)
    print("  MANGA109 -> YOLO  |  Data Preparation Script")
    print("=" * 60)

    annotations_dir, images_dir = find_manga109_source_dirs(raw_dir)
    print(f"\n[OK] Thu muc XML annotations : {annotations_dir}")
    print(f"[OK] Thu muc anh goc         : {images_dir}")

    # -----------------------------------------------------------------------
    # BƯỚC 2: Tạo cấu trúc thư mục đầu ra chuẩn YOLO
    # -----------------------------------------------------------------------
    train_img_dir = output_dir / "images" / "train"
    val_img_dir   = output_dir / "images" / "val"
    train_lbl_dir = output_dir / "labels" / "train"
    val_lbl_dir   = output_dir / "labels" / "val"

    # Tạo tất cả các thư mục cần thiết (kể cả thư mục cha)
    for directory in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Da tao cau truc thu muc tai: {output_dir}")

    # -----------------------------------------------------------------------
    # BƯỚC 3: Quét tất cả file XML và thu thập dữ liệu từng trang
    # -----------------------------------------------------------------------
    xml_files = sorted(list(annotations_dir.glob("*.xml")))
    if not xml_files:
        raise FileNotFoundError(
            f"Khong tim thay file .xml nao trong '{annotations_dir}'.\n"
            f"Hay dat file annotation Manga109 vao thu muc do."
        )

    print(f"\n[*] Tim thay {len(xml_files)} tua truyen (file XML).")
    print("[*] Dang doc annotation tu tat ca cac file XML...")

    # Danh sách chứa toàn bộ trang truyện từ tất cả tựa truyện
    all_pages = []

    for xml_file in xml_files:
        book_title = xml_file.stem  # Tên tựa truyện (không có phần mở rộng .xml)
        pages = parse_manga109_xml(xml_file)

        for page in pages:
            # Kiểm tra xem file ảnh tương ứng có tồn tại không
            src_img_path = _find_image_file(images_dir, book_title, page["page_index"])
            if src_img_path is None:
                continue  # Bỏ qua trang không có file ảnh

            all_pages.append({
                "book":         book_title,
                "page_index":   page["page_index"],
                "src_img_path": src_img_path,
                "yolo_labels":  page["yolo_labels"]
            })

    print(f"[OK] Tong so trang anh hop le (co ca XML + anh): {len(all_pages)}")

    # -----------------------------------------------------------------------
    # BƯỚC 4: Xáo trộn ngẫu nhiên và chia tập train/val theo tỷ lệ 85/15
    # -----------------------------------------------------------------------
    # random.shuffle xáo trộn tại chỗ (in-place), không tạo list mới
    random.shuffle(all_pages)

    val_size   = max(1, int(len(all_pages) * val_ratio))  # Đảm bảo ít nhất 1 ảnh val
    train_size = len(all_pages) - val_size

    train_pages = all_pages[:train_size]
    val_pages   = all_pages[train_size:]

    train_pct = train_size / len(all_pages) * 100
    val_pct   = val_size   / len(all_pages) * 100

    print(f"\n[*] Phan chia tap du lieu (seed={seed}):")
    print(f"    +-- Train : {train_size:>5} trang ({train_pct:.1f}%)")
    print(f"    +-- Val   : {val_size:>5}  trang ({val_pct:.1f}%)")

    # -----------------------------------------------------------------------
    # BƯỚC 5: Sao chép ảnh và ghi file nhãn YOLO cho từng tập
    # -----------------------------------------------------------------------
    splits = [
        ("train", train_pages, train_img_dir, train_lbl_dir),
        ("val",   val_pages,   val_img_dir,   val_lbl_dir),
    ]

    total_imgs_copied    = 0
    total_labels_written = 0
    total_text_boxes     = 0

    for split_name, pages, img_dir, lbl_dir in splits:
        print(f"\n[*] Dang xu ly tap '{split_name}' ({len(pages)} trang)...")
        count_imgs = 0
        count_lbls = 0

        for page_info in pages:
            book     = page_info["book"]
            page_idx = page_info["page_index"]
            src_img  = page_info["src_img_path"]
            labels   = page_info["yolo_labels"]

            # Đặt tên file thống nhất: <TenTruyen>_<SoTrang> để tránh trùng
            file_stem  = f"{book}_{page_idx:04d}"
            dest_img   = img_dir / f"{file_stem}{src_img.suffix}"
            dest_label = lbl_dir / f"{file_stem}.txt"

            # --- Sao chép file ảnh ---
            if copy_images and not dest_img.exists():
                try:
                    # Thử dùng hardlink trước (nhanh hơn và không tốn dung lượng)
                    os.link(src_img, dest_img)
                except OSError:
                    # Nếu hardlink không được (khác phân vùng đĩa), dùng shutil.copy2
                    shutil.copy2(str(src_img), str(dest_img))
                count_imgs += 1

            # --- Ghi file nhãn .txt ---
            # Ghi cả trang không có text (file rỗng) để mô hình học negative sample
            with open(dest_label, "w", encoding="utf-8") as f:
                for bbox in labels:
                    cls_id, xc, yc, w, h = bbox
                    # Định dạng chuẩn YOLO: <class_id> <x_center> <y_center> <width> <height>
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            count_lbls += 1
            total_text_boxes += len(labels)

        total_imgs_copied    += count_imgs
        total_labels_written += count_lbls
        print(f"    +-- Anh da sao chep  : {count_imgs}")
        print(f"    +-- File nhan da ghi : {count_lbls}")

    # -----------------------------------------------------------------------
    # BƯỚC 6: In báo cáo tổng kết
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  [OK] XU LY DU LIEU HOAN TAT THANH CONG!")
    print("=" * 60)
    print(f"  Tong anh da sao chep     : {total_imgs_copied}")
    print(f"  Tong file nhan .txt      : {total_labels_written}")
    print(f"  Tong vung chu (text box) : {total_text_boxes}")
    print(f"  Du lieu luu tai          : {output_dir.resolve()}")
    print("\n  Cau truc thu muc dau ra:")
    print(f"  {output_dir.name}/")
    print(f"  +-- images/")
    print(f"  |   +-- train/   ({train_size} anh)")
    print(f"  |   +-- val/     ({val_size} anh)")
    print(f"  +-- labels/")
    print(f"      +-- train/   ({train_size} file .txt)")
    print(f"      +-- val/     ({val_size} file .txt)")
    print("=" * 60)


# ===========================================================================
# PHẦN 6: PHÂN TÍCH THAM SỐ DÒNG LỆNH
# ===========================================================================

def parse_arguments():
    """
    Phân tích tham số dòng lệnh (Command Line Arguments).
    Cho phép người dùng tùy chỉnh đường dẫn và tỷ lệ chia tập mà không cần sửa code.
    """
    # Tự động xác định thư mục gốc của project (script này nằm trong scripts/)
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Xu ly du lieu Manga109 sang dinh dang nhan YOLO (Text Detection)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(repo_root / "data" / "raw"),
        help="Duong dan thu muc chua du lieu Manga109 goc (co thu muc 'annotations' va 'images')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "dataset"),
        help="Duong dan thu muc dau ra se chua cau truc YOLO (images/, labels/)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ti le tap Validation (vi du: 0.15 = 15%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed de chia tap train/val nhat quan"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Chi tao file nhan .txt, khong sao chep anh"
    )

    return parser.parse_args()


# ===========================================================================
# DIEM KHOI CHAY CHINH
# ===========================================================================

def main():
    """Hàm main: Điểm khởi chạy chính của script."""
    args = parse_arguments()

    raw_dir    = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    # Kiểm tra thư mục đầu vào có tồn tại không
    if not raw_dir.exists():
        print(f"\n[LOI] Thu muc du lieu goc khong ton tai: '{raw_dir}'")
        print("       Hay tao thu muc va dat du lieu Manga109 vao do truoc khi chay script.")
        sys.exit(1)

    # Kiểm tra tỷ lệ val hợp lệ
    if not (0.05 <= args.val_ratio <= 0.5):
        print(f"[LOI] val-ratio phai trong khoang [0.05, 0.5], nhan duoc: {args.val_ratio}")
        sys.exit(1)

    try:
        prepare_dataset(
            raw_dir=raw_dir,
            output_dir=output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            copy_images=not args.no_copy
        )
    except FileNotFoundError as e:
        print(f"\n[LOI] Khong tim thay file/thu muc:\n  {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[LOI] Loi khong mong doi trong qua trinh xu ly:\n  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
