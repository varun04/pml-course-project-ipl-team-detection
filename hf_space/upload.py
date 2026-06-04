"""Run this after: .venv/bin/huggingface-cli login

Usage:
    .venv/bin/python hf_space/upload.py
"""
from huggingface_hub import HfApi, create_repo

api  = HfApi()
REPO = "varun04tomarP/ipl-team-detection"

print(f"Creating Space: {REPO}")
create_repo(repo_id=REPO, repo_type="space", space_sdk="gradio",
            private=False, exist_ok=True)
print("Space created ✅")

print("Uploading README.md...")
api.upload_file(path_or_fileobj="hf_space/README.md",
                path_in_repo="README.md", repo_id=REPO, repo_type="space")

print("Uploading requirements.txt...")
api.upload_file(path_or_fileobj="hf_space/requirements.txt",
                path_in_repo="requirements.txt", repo_id=REPO, repo_type="space")

print("Uploading app.py...")
api.upload_file(path_or_fileobj="hf_space/app.py",
                path_in_repo="app.py", repo_id=REPO, repo_type="space")

print("Uploading src/ (phase3 modules)...")
api.upload_folder(folder_path="src", path_in_repo="src",
                  repo_id=REPO, repo_type="space",
                  ignore_patterns=["__pycache__", "*.pyc", "app.py"])

print("Uploading model (48 MB — takes ~1 min)...")
api.upload_file(
    path_or_fileobj="models/model_ipl_jersey_prediction_varun.pkl",
    path_in_repo="model_ipl_jersey_prediction_varun.pkl",
    repo_id=REPO, repo_type="space"
)

print("\nAll uploaded! ✅")
print(f"Space URL: https://huggingface.co/spaces/{REPO}")
print("Live in ~2-3 minutes once HF builds the container.")
