# V.S.

from huggingface_hub import HfApi
from pathlib import Path
import time

HF_REPO_ID = "Aurumz/RCT-Reviewer-pickle"  
LOCAL_DATA_DIR = Path("data")


def upload_clean_models():
    api = HfApi()
    
    print(f"🔄 Starting File-by-File Upload to: {HF_REPO_ID}")
    print("⚠️  MODE: Uploading files sequentially (slowest, but safest).\n")


    print("Scanning files...")
    all_files = list(LOCAL_DATA_DIR.rglob('*'))
    files_to_upload = [f for f in all_files if f.is_file()]
    total_files = len(files_to_upload)
    
    print(f"Found {total_files} files to upload.\n")

    count = 0


    for file_path in files_to_upload:
        count += 1
        

        path_in_repo = file_path.relative_to(LOCAL_DATA_DIR)

        print(f"[{count}/{total_files}] ⬆️ Uploading: {path_in_repo}", end="\r")

        try:
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=str(path_in_repo),
                repo_id=HF_REPO_ID,
                repo_type="model",
               
                commit_message=f"Upload file: {path_in_repo}"
            )
            

            time.sleep(0.5)

        except Exception as e:
            print(f"\n❌ Failed to upload {path_in_repo}: {e}")

    print("\n\n------------------------------------------------")
    print(f"✅ Finished! Processed {count} files.")
    print(f"👉 View here: https://huggingface.co/{HF_REPO_ID}/tree/main")
    print("------------------------------------------------")

if __name__ == "__main__":
    upload_clean_models()