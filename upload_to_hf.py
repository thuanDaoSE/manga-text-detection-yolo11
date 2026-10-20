import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

parser = argparse.ArgumentParser(description="Đẩy Space Demo lên Hugging Face")
parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token")
args = parser.parse_args()

# Load token
token = args.token
if not token:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    token = os.getenv("HF_TOKEN")

if not token:
    print("[!] Lỗi: Không tìm thấy Hugging Face Token!")
    print("    Vui lòng truyền qua: python upload_to_hf.py --token hf_xxx")
    sys.exit(1)

try:
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
except Exception as e:
    print(f"[!] Token không hợp lệ: {e}")
    sys.exit(1)

repo_name = "manga-text-detection-yolo11n"
repo_id = f"{username}/{repo_name}"

print("="*60)
print(f"[*] Tài khoản Hugging Face : {username}")
print(f"[*] Tên Space đích        : {repo_id}")
print("="*60)

# 1. Tạo Space nếu chưa tồn tại
try:
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False
    )
    print(f"[✓] Đã tạo hoặc xác nhận Space: {repo_id}")
except Exception as e:
    if "403" in str(e):
        print("\n[!] LỖI PHÂN QUYỀN (403 FORBIDDEN):")
        print("    Token hiện tại chỉ có quyền 'Read' (Đọc), không có quyền 'Write' (Ghi/Tạo Space).")
        print("    👉 Cách khắc phục:")
        print("       1. Vào: https://huggingface.co/settings/tokens")
        print("       2. Tạo một Token mới với Type: 'Write'")
        print("       3. Chạy lại: python upload_to_hf.py --token <token_write_cua_ban>")
        sys.exit(1)
    else:
        print(f"[*] Thông báo repo: {e}")

# 2. Upload toàn bộ thư mục hf_manga_space lên Space
source_folder = Path(__file__).resolve().parent / "hf_manga_space"
if not source_folder.exists():
    source_folder = Path(__file__).resolve().parent

print(f"[*] Đang tải toàn bộ file từ '{source_folder}' lên Hugging Face Space...")
api.upload_folder(
    folder_path=str(source_folder),
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=["upload_to_hf.py", "*.pyc", "__pycache__"]
)

print("\n" + "="*60)
print("🎉 ĐẨY LÊN HUGGING FACE THÀNH CÔNG 100%!")
print("="*60)
print(f"👉 Link Space Web Demo Trực Tuyến:")
print(f"   https://huggingface.co/spaces/{repo_id}")
print("="*60)
