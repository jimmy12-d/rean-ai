import json
import os
import glob

def debug_rag_counts():
    rag_folder = "Subject Rag"
    source_files = glob.glob(os.path.join(rag_folder, "**/*.jsonl"), recursive=True)
    
    print(f"📂 Found {len(source_files)} files in '{rag_folder}':")
    for f in source_files:
        print(f" - {f}")
        
    concepts_count = 0
    exercises_count = 0
    
    for file_path in source_files:
        print(f"\nProcessing {file_path}...")
        file_c = 0
        file_e = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    meta = data.get('metadata', {})
                    item_id = data.get('id', '')
                    
                    # Logic from rag_utils.py
                    doc_type = meta.get('type', '')
                    
                    is_exercise = False
                    if item_id.startswith('EX_'):
                        is_exercise = True
                    elif doc_type in ['Q&A', 'Solved Example']:
                        is_exercise = True
                        
                    if is_exercise:
                        exercises_count += 1
                        file_e += 1
                    else:
                        concepts_count += 1
                        file_c += 1
                        # Print first few concepts to see what they are
                        if concepts_count <= 5:
                            print(f"   [Concept Sample] ID: {item_id}, Type: {doc_type}")

                except json.JSONDecodeError:
                    print(f"   ❌ Error decoding line {line_num}")
                    continue
        print(f"   -> Concepts: {file_c}, Exercises: {file_e}")

    print("\n" + "="*30)
    print(f"TOTAL Concepts: {concepts_count}")
    print(f"TOTAL Exercises: {exercises_count}")
    print("="*30)

if __name__ == "__main__":
    debug_rag_counts()
