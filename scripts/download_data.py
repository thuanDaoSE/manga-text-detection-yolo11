#!/usr/bin/env python3
"""
scripts/download_data.py
Script tải và giải nén bộ dữ liệu Manga109-s từ Hugging Face Hub.
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, HfApi
from tqdm import tqdm


def get_token(args_token: str | None) -> str:
    """
    Lấy token Hugging Face từ CLI argument hoặc biến môi trường .env.
    """
    if args_token:
        return args_token
    
    # Load từ file .env ở thư mục gốc
    repo_root = Path(__file__).resolve().parent.parent
    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()
        
    token = os.getenv("HF_TOKEN")
    if not token or token == "your_huggingface_token_here":
        raise ValueError(
            "Không tìm thấy Hugging Face Token hợp lệ! "
            "Vui lòng thiết lập HF_TOKEN trong file .env hoặc truyền qua tham số --token."
        )
    return token


def download_manga109(
    repo_id: str = "hal-utokyo/Manga109-s",
    filename: str = "Manga109s_released_2023_12_07.zip",
    token: str | None = None,
    raw_dir: Path | None = None,
    extract: bool = True
) -> Path:
    """
    Tải tập nén Manga109 từ HuggingFace Hub và giải nén vào data/raw.
    """
    hf_token = get_token(token)
    
    if raw_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        raw_dir = repo_root / "data" / "raw"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Tự động tìm kiếm file zip nếu file chỉ định không tồn tại
    api = HfApi()
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset", token=hf_token)
        available_files = [f.rfilename for f in repo_info.siblings if f.rfilename.endswith(".zip")]
        if filename not in available_files and available_files:
            print(f"[!] File '{filename}' không còn trên repo. Tự động chuyển sang file mới nhất: '{available_files[0]}'")
            filename = available_files[0]
    except Exception as e:
        print(f"[*] Không thể kiểm tra danh sách repo: {e}")

    print(f"[*] Đang tải {filename} từ repository '{repo_id}'...")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(raw_dir)
    )
    zip_path = Path(zip_path)
    print(f"[✓] Tải thành công file: {zip_path} (Kích thước: {zip_path.stat().st_size / (1024*1024):.2f} MB)")
    
    if extract:
        print(f"[*] Đang tiến hành giải nén vào thư mục: {raw_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.infolist()
            for member in tqdm(members, desc="Giải nén", unit="files"):
                zip_ref.extract(member, raw_dir)
        print(f"[✓] Giải nén hoàn tất tại: {raw_dir}")
        
    return zip_path


def parse_args():
    parser = argparse.ArgumentParser(description="Tải dữ liệu Manga109-s từ Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="hal-utokyo/Manga109-s",
        help="Repository ID trên Hugging Face"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="Manga109s_released_2023_12_07.zip",
        help="Tên file zip trên Hugging Face dataset"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face Access Token (mặc định lấy từ .env)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Thư mục lưu trữ raw data (mặc định: data/raw/)"
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Chỉ tải file zip, không giải nén"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    try:
        download_manga109(
            repo_id=args.repo_id,
            filename=args.filename,
            token=args.token,
            raw_dir=raw_dir,
            extract=not args.no_extract
        )
    except Exception as e:
        print(f"[!] Lỗi khi tải dữ liệu: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
