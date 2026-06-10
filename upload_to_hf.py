# V.S.



from huggingface_hub import HfApi
from pathlib import Path


HF_REPO_ID = "Aurumz/RCT-Reviewer"  
LOCAL_DATA_DIR = Path("data")


def upload_clean_models():
    api = HfApi()
    
    print(f"🔄 Syncing optimized models to: {HF_REPO_ID}")
    print("⚠️ NOTE: This will DELETE unused CNN (.h5) files from the repo to save space.\n")


    allow_patterns = [

        "pico/*.npz",
        
    
        "bias/*.npz",
        

        "rct/*.npz",
        "rct/*.joblib",     
        "rct/*.json",       
        
 
        "drugbank/*.joblib",
        
  
        "bias_ab/*.npz",
        "bias_ab/*.joblib"
    ]


    delete_patterns = [
        "rct/cnn*.h5",    
        "rct/*.h5",     
        "*.hdf5",         
        "sample_size/*",   
        "minimap/*",       
        "pico_spans/*",   
        "*.pickle", "*.pck" 
    ]

    try:
        api.upload_folder(
            folder_path=str(LOCAL_DATA_DIR),
            repo_id=HF_REPO_ID,
            repo_type="model",
            allow_patterns=allow_patterns,   
            delete_patterns=delete_patterns, 
            commit_message="Clean repo: Remove CNN weights, keep core SVM models"
        )
        print("\n------------------------------------------------")
        print("✅ SUCCESS! Repo cleaned and synced.")
        print(f"👉 View here: https://huggingface.co/{HF_REPO_ID}/tree/main")
        print("------------------------------------------------")
        
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

if __name__ == "__main__":
    upload_clean_models()