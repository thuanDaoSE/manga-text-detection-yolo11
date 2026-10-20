#!/usr/bin/env python3
"""
full_pipeline.py
================
Tự động hóa 100% quy trình:
1. Tải Manga109-s từ Hugging Face
2. Giải nén và chuyển đổi sang chuẩn YOLO
3. Tạo file cấu hình manga_text.yaml
4. Huấn luyện YOLO11n đầy đủ 30 epochs (đã tối ưu tham số chống crash)
5. Tự động Validation, đánh giá mAP50, Precision, Recall và lưu kết quả.
"""

import os
import sys
import glob
import time
import shutil
import random
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

# 1. Cấu hình
HF_TOKEN = os.getenv("HF_TOKEN", "")
REPO_ID = "hal-utokyo/Manga109-s"

ZIP_FILENAME = "Manga109s_released_2026_05_21.zip"

BASE_DIR = Path("/root/data")
RAW_DIR = BASE_DIR / "raw"
DATASET_DIR = BASE_DIR / "dataset"
YAML_PATH = BASE_DIR / "manga_text.yaml"
RUNS_DIR = BASE_DIR / "runs" / "detect"
PROJECT_NAME = "manga_text_full"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# -------------------------------------------------------------
# BƯỚC 1: TẢI & GIẢI NÉN DATASET
# -------------------------------------------------------------
def download_and_extract():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Kiểm tra xem đã giải nén chưa
    xml_check = list(RAW_DIR.glob("**/*.xml"))
    if len(xml_check) >= 50:
        log(f"Dataset da ton tai ({len(xml_check)} file XML). Bo qua tai xuong.")
        return

    from huggingface_hub import hf_hub_download
    log(f"Dang tai {ZIP_FILENAME} tu Hugging Face repo {REPO_ID}...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=ZIP_FILENAME,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir=str(RAW_DIR)
    )
    log(f"Tai thanh cong: {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.1f} MB)")
    
    log("Dang giai nen file zip vao " + str(RAW_DIR) + "...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)
    log("Giai nen hoan tat!")

# -------------------------------------------------------------
# BƯỚC 2: CHUYỂN ĐỔI SANG YOLO FORMAT
# -------------------------------------------------------------
def convert_bbox(xmin, ymin, xmax, ymax, w, h):
    if w <= 0 or h <= 0:
        return None
    x1, x2 = min(float(xmin), float(xmax)), max(float(xmin), float(xmax))
    y1, y2 = min(float(ymin), float(ymax)), max(float(ymin), float(ymax))
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    xc = (x1 + x2) / (2.0 * w)
    yc = (y1 + y2) / (2.0 * h)
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return 0, round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)

def prepare_yolo_dataset(val_ratio=0.15, seed=42):
    train_img = DATASET_DIR / "images" / "train"
    val_img = DATASET_DIR / "images" / "val"
    train_lbl = DATASET_DIR / "labels" / "train"
    val_lbl = DATASET_DIR / "labels" / "val"

    if train_img.exists() and len(list(train_img.glob("*.jpg"))) > 5000:
        log("YOLO Dataset da ton tai va day du. Bo qua prepare.")
        return

    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    log("Dang tim thu muc annotations va images trong " + str(RAW_DIR))
    anno_dirs = list(RAW_DIR.glob("**/annotations"))
    img_dirs = list(RAW_DIR.glob("**/images"))
    if not anno_dirs or not img_dirs:
        raise RuntimeError("Khong tim thay annotations hoac images trong raw_dir!")
    
    anno_dir = sorted(anno_dirs, key=lambda p: len(p.parts))[0]
    img_dir = sorted(img_dirs, key=lambda p: len(p.parts))[0]
    
    xml_files = sorted(list(anno_dir.glob("*.xml")))
    log(f"Tim thay {len(xml_files)} file XML.")

    all_pages = []
    for xml_path in xml_files:
        book = xml_path.stem
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
        except Exception:
            continue
            
        pages_node = root.find("pages")
        page_elems = pages_node.findall("page") if pages_node is not None else root.findall("page")
        
        book_img_dir = img_dir / book
        for pe in page_elems:
            try:
                idx = int(pe.get("index", 0))
                pw = float(pe.get("width", 0))
                ph = float(pe.get("height", 0))
            except Exception:
                continue
            if pw <= 0 or ph <= 0:
                continue
                
            # Find image file
            src_img = None
            for cand in [f"{idx:03d}.jpg", f"{idx:03d}.png", f"{idx:04d}.jpg", f"{idx}.jpg"]:
                p = book_img_dir / cand
                if p.exists():
                    src_img = p
                    break
            if not src_img:
                continue
                
            labels = []
            for te in pe.findall("text"):
                try:
                    bbox = convert_bbox(te.get("xmin"), te.get("ymin"), te.get("xmax"), te.get("ymax"), pw, ph)
                    if bbox:
                        labels.append(bbox)
                except Exception:
                    continue
                    
            all_pages.append({
                "book": book,
                "idx": idx,
                "img": src_img,
                "labels": labels
            })

    log(f"Tong so trang hop le: {len(all_pages)}")
    random.seed(seed)
    random.shuffle(all_pages)
    
    val_n = int(len(all_pages) * val_ratio)
    splits = [
        ("val", all_pages[:val_n], val_img, val_lbl),
        ("train", all_pages[val_n:], train_img, train_lbl)
    ]
    
    for sname, pages, idir, ldir in splits:
        log(f"Dang ghi du lieu tap {sname} ({len(pages)} trang)...")
        for item in pages:
            name = f"{item['book']}_{item['idx']:04d}"
            dst_img = idir / f"{name}{item['img'].suffix}"
            dst_lbl = ldir / f"{name}.txt"
            
            if not dst_img.exists():
                try:
                    os.link(str(item['img']), str(dst_img))
                except Exception:
                    shutil.copy2(str(item['img']), str(dst_img))
                    
            with open(dst_lbl, "w", encoding="utf-8") as f:
                for b in item['labels']:
                    f.write(f"{b[0]} {b[1]} {b[2]} {b[3]} {b[4]}\n")
                    
    log("Hoan tat chuan bi dataset!")

# -------------------------------------------------------------
# BƯỚC 3: TẠO FILE YAML CẤU HÌNH
# -------------------------------------------------------------
def create_yaml():
    yaml_content = f"""path: {DATASET_DIR}
train: images/train
val: images/val
nc: 1
names: ['text']
"""
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    log(f"Da tao file cau hinh: {YAML_PATH}")

# -------------------------------------------------------------
# BƯỚC 4: HUẤN LUYỆN YOLO11n (FULL EPOCHS)
# -------------------------------------------------------------
def train_model(epochs=30):
    import torch
    from ultralytics import YOLO
    
    log(f"Khoi tao model YOLO11n tren device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    model = YOLO("yolo11n.pt")
    
    log(f"BAT DAU HUAN LUYEN: {epochs} EPOCHS...")
    results = model.train(
        data=str(YAML_PATH),
        epochs=epochs,
        imgsz=640,
        batch=16,
        workers=0,          # On dinh tren cloud VM
        amp=False,          # Tranh loi AMP tren GPU
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(RUNS_DIR),
        name=PROJECT_NAME,
        save=True,
        plots=True,
        patience=10,        # Early stopping sau 10 epoch neu khong cai thien
        exist_ok=True,
        cache='disk',       # Tranh tran RAM
        verbose=True
    )
    log("HUAN LUYEN HOAN TAT!")
    best_weights = RUNS_DIR / PROJECT_NAME / "weights" / "best.pt"
    log(f"Model tot nhat duoc luu tai: {best_weights}")
    return best_weights

# -------------------------------------------------------------
# BƯỚC 5: ĐÁNH GIÁ (VALIDATION & DEMO TEST)
# -------------------------------------------------------------
def evaluate_and_demo(weights_path):
    from ultralytics import YOLO
    log(f"Dang load weights de danh gia: {weights_path}")
    model = YOLO(str(weights_path))
    
    log("Chay validation tren tap val...")
    metrics = model.val(data=str(YAML_PATH), device=0)
    
    print("\n" + "="*50)
    print("           KET QUA DANH GIA CHAT LUONG")
    print("="*50)
    print(f"mAP50        : {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"mAP50-95     : {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    print(f"Precision    : {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
    print(f"Recall       : {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")
    print("="*50 + "\n")
    
    # Run test on 5 images
    val_images = sorted(list((DATASET_DIR / "images" / "val").glob("*.jpg")))[:5]
    if val_images:
        log("Chay thu nghiem tren 5 anh mau...")
        demo_res = model.predict(
            source=[str(p) for p in val_images],
            save=True,
            project=str(RUNS_DIR),
            name="demo_results",
            conf=0.25,
            exist_ok=True
        )
        for r in demo_res:
            log(f"  Anh: {os.path.basename(r.path)} -> Phat hien {len(r.boxes)} vung text")
        log(f"Anh demo da duoc ve bounding box tai: {RUNS_DIR / 'demo_results'}")

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    log(f"=== KHOI DONG PIPELINE YOLO11n ({epochs} EPOCHS) ===")
    
    download_and_extract()
    prepare_yolo_dataset()
    create_yaml()
    best_pt = train_model(epochs=epochs)
    evaluate_and_demo(best_pt)
    log("=== TAT CA DA HOAN TAT 100% ===")
