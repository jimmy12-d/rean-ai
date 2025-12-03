import json
import os
import glob
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Global variables to hold the vector stores
db_concepts = None
db_exercises = None
embedding_model = None

def load_and_split_docs(file_paths):
    concepts_docs = []
    exercises_docs = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"⚠️ Warning: {file_path} not found.")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Combine title and content for the page_content
                    title = data.get('khmer_title', '')
                    body = data.get('content', '')
                    content = f"{title}\n{body}" if title else body
                    
                    meta = data.get('metadata', {})
                    item_id = data.get('id', '')
                    meta['id'] = item_id
                    
                    # If metadata is empty, try to populate it from root fields (for concepts)
                    if not meta:
                        for key in ['subject', 'chapter', 'topic', 'khmer_title']:
                            if key in data:
                                meta[key] = data[key]

                    doc = Document(page_content=content, metadata=meta)
                    
                    doc_type = meta.get('type', '')
                    if item_id.startswith('EX_') or doc_type in ['Solved Example']:
                        exercises_docs.append(doc)
                    else:
                        concepts_docs.append(doc)
                except json.JSONDecodeError:
                    continue
    return concepts_docs, exercises_docs

def initialize_rag_db():
    global db_concepts, db_exercises, embedding_model
    
    print("⏳ Initializing RAG Database...")
    
    # Initialize Embedding Model
    # Using BAAI/bge-m3 as requested
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    # Load Documents from the new RAG files
    # Recursively find all .jsonl files in "Subject Rag" folder
    rag_folder = "Subject Rag"
    source_files = glob.glob(os.path.join(rag_folder, "**/*.jsonl"), recursive=True)
    
    if not source_files:
        print(f"⚠️ No .jsonl files found in {rag_folder}")
    else:
        print(f"📂 Found {len(source_files)} RAG files: {source_files}")

    concepts_docs, exercises_docs = load_and_split_docs(source_files)
    
    if not concepts_docs:
        print("⚠️ No concept documents found.")
    if not exercises_docs:
        print("⚠️ No exercise documents found.")

    # Create FAISS Indices
    # We build them in memory for now. In a production app, you might load from disk.
    if concepts_docs:
        db_concepts = FAISS.from_documents(concepts_docs, embedding_model)
        print(f"✅ Concepts Index Built ({len(concepts_docs)} docs)")
    
    if exercises_docs:
        db_exercises = FAISS.from_documents(exercises_docs, embedding_model)
        print(f"✅ Exercises Index Built ({len(exercises_docs)} docs)")
        
    print("✅ RAG Database Ready!")

def retrieve_context(query, subject=None):
    global db_concepts, db_exercises
    
    concept_text = "No relevant concept found."
    exercise_text = "No relevant exercise found."
    
    # L2 Distance Threshold: Lower is better.
    # 0.0 = Identical, ~1.0 = Unrelated (Orthogonal for normalized vectors)
    # We filter out results with high distance to avoid irrelevant cross-subject matches.
    SCORE_THRESHOLD = 0.8
    
    # Prepare metadata filter if subject is known
    filter_args = {}
    if subject:
        filter_args['filter'] = {'subject': subject}
    
    if db_concepts:
        # Retrieve top 1 concept with score
        results = db_concepts.similarity_search_with_score(query, k=1, **filter_args)
        if results:
            doc, score = results[0]
            # Only use the result if it's sufficiently similar
            if score < SCORE_THRESHOLD:
                print(f"✅ Best Concept Match: {score:.4f}")
                concept_text = f"{doc.page_content}\n\n(Similarity Score: {score:.4f})"
            
    if db_exercises:
        # Retrieve top 1 exercise with score
        results = db_exercises.similarity_search_with_score(query, k=1, **filter_args)
        if results:
            doc, score = results[0]
            if score < SCORE_THRESHOLD:
                print(f"✅ Best Exercise Match: {score:.4f}")
                exercise_text = f"{doc.page_content}\n\n(Similarity Score: {score:.4f})"
            
    return concept_text, exercise_text

def format_rag_prompt(user_query, concept, exercise):
    # Check if we have valid content (not the default failure messages)
    has_concept = concept and not concept.startswith("No relevant concept")
    has_exercise = exercise and not exercise.startswith("No relevant exercise")

    prompt_parts = ["You are a Khmer Grade 12 Tutor."]

    if has_concept or has_exercise:
        prompt_parts.append("Use the following context to answer the user.")
        if has_concept:
            prompt_parts.append(f"CORE FORMULA:\n{concept}")
        if has_exercise:
            prompt_parts.append(f"SOLVED EXAMPLE (Follow this Strategy):\n{exercise}")
    else:
        # Fallback if no RAG context is found
        prompt_parts.append("Answer the user's question based on your general knowledge of the Khmer Grade 12 curriculum.")

    prompt_parts.append(f"USER QUESTION:\n{user_query}")

    return "\n\n".join(prompt_parts)
