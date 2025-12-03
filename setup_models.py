import os
import shutil
import glob
from huggingface_hub import hf_hub_download

# --- CONFIGURATION (FIXED) ---
SOURCE_DIR = os.path.expanduser("~/llama.cpp") 
DEST_DIR = "./models"
# We switch to 'bartowski' repo which is the gold standard for GGUF
BASE_MODEL_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
# Exact filename case-sensitive fix:
BASE_MODEL_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf" 

# Create destination
os.makedirs(DEST_DIR, exist_ok=True)

def find_and_copy(filename_pattern, description):
    print(f"🔍 Looking for {description} in {SOURCE_DIR}...")
    
    # Search for the file (handling case sensitivity)
    search_path = os.path.join(SOURCE_DIR, filename_pattern)
    found_files = glob.glob(search_path)
    
    # Also try exact match in case glob fails on case
    if not found_files:
        exact_path = os.path.join(SOURCE_DIR, BASE_MODEL_FILE)
        if os.path.exists(exact_path):
            found_files = [exact_path]

    if found_files:
        source_file = found_files[0]
        dest_file = os.path.join(DEST_DIR, os.path.basename(source_file))
        print(f"✅ FOUND! Copying '{os.path.basename(source_file)}' to models folder...")
        shutil.copy2(source_file, dest_file)
        print("   Copy complete.")
        return dest_file
    else:
        print(f"❌ Not found locally.")
        return None

# --- ACTION ---

# 1. GET THE BRAIN (Your Adapter)
brain_path = find_and_copy("khmer_brain.gguf", "Khmer Brain (Adapter)")
if not brain_path:
    print("⚠️ CRITICAL: 'khmer_brain.gguf' is missing locally.")
    print("   Make sure you ran the conversion script inside llama.cpp first!")

# 2. GET THE BODY (Base Qwen Model)
# Try finding locally first
body_path = find_and_copy(BASE_MODEL_FILE, "Qwen Base Body")

# If not found, download the correct file
if not body_path:
    print(f"\n📥 Downloading Base Model from {BASE_MODEL_REPO}...")
    try:
        body_path = hf_hub_download(
            repo_id=BASE_MODEL_REPO,
            filename=BASE_MODEL_FILE,
            local_dir=DEST_DIR,
            local_dir_use_symlinks=False
        )
        print("✅ Download complete!")
    except Exception as e:
        print(f"❌ Download failed: {e}")

print("\n✨ SYSTEM CHECK:")
print(f"   Brain: {'✅ Ready' if brain_path and os.path.exists(brain_path) else '❌ MISSING'}")
print(f"   Body:  {'✅ Ready' if body_path and os.path.exists(body_path) else '❌ MISSING'}")

if brain_path and body_path:
    print("\n🚀 You are ready to run main.py!")
else:
    print("\n⚠️  Fix the missing files before running main.py")