import json
import os

def populate_data():
    print("🚀 Populating RAG Database with Full Data...")
    
    concepts = []
    exercises = []
    
    # Files to read from
    source_files = ["physics_concepts_rag.jsonl", "physics_exercise_rag.jsonl"]
    
    for file_path in source_files:
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: {file_path} not found. Skipping.")
            continue
            
        print(f"📖 Reading {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    item_id = data.get('id', '')
                    
                    # Categorize based on ID prefix
                    if item_id.startswith('EX_'):
                        exercises.append(data)
                    else:
                        # Assume everything else is a concept (TH_, EM_, WV_, etc.)
                        concepts.append(data)
                        
                except json.JSONDecodeError:
                    continue

    # Remove duplicates (based on ID)
    unique_concepts = {c['id']: c for c in concepts}.values()
    unique_exercises = {e['id']: e for e in exercises}.values()
    
    print(f"✅ Found {len(unique_concepts)} unique concepts.")
    print(f"✅ Found {len(unique_exercises)} unique exercises.")

    # Analyze prefixes for clarity
    from collections import Counter
    concept_prefixes = Counter([c['id'].split('_')[0] for c in unique_concepts])
    exercise_prefixes = Counter([e['id'].split('_')[0] for e in unique_exercises])
    
    print("\n📊 Breakdown by ID Prefix:")
    print("Concepts:")
    for prefix, count in concept_prefixes.items():
        print(f"  - {prefix}: {count}")
    print("Exercises:")
    for prefix, count in exercise_prefixes.items():
        print(f"  - {prefix}: {count}")
    print("")
    
    # Write to destination files
    print("💾 Writing to physics_concepts.jsonl...")
    with open("physics_concepts.jsonl", "w", encoding="utf-8") as f:
        for entry in unique_concepts:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print("💾 Writing to physics_exercises.jsonl...")
    with open("physics_exercises.jsonl", "w", encoding="utf-8") as f:
        for entry in unique_exercises:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print("🎉 Data Population Complete!")

if __name__ == "__main__":
    populate_data()
