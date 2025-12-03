from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import rag_utils  # Import the RAG utility module
import gc
import json
import threading
import time

# 1. Initialize the App
app = FastAPI(title="Khmer Grade 12 Tutor API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Configurations
MODELS = {
    "qwen": {
        "model_path": "./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "lora_path": "./models/khmer_brain.gguf",
        "alias": "Qwen 2.5 (Khmer Brain)"
    },
    "seallm": {
        "model_path": "/Users/jimmy/llama.cpp/SeaLLMs-v3-7B-Q4_K_M.gguf",
        "lora_path": "./models/khmer_seallm_brain.gguf",
        "alias": "Khmer SeaLLM"
    }
}

# Global Model Variable
llm = None
current_model_name = "qwen"
model_lock = threading.Lock()  # Prevent race conditions during model switching
is_loading = False

def load_model(model_key: str):
    global llm, current_model_name, is_loading
    
    if model_key not in MODELS:
        raise ValueError(f"Model '{model_key}' not found.")
    
    # Prevent concurrent model loading
    with model_lock:
        is_loading = True
        old_model_name = current_model_name
        
        # Unload existing model if any
        if llm is not None and model_key != current_model_name:
            print(f"♻️ Unloading {MODELS[old_model_name]['alias']}...")
            del llm
            llm = None
            gc.collect()
            time.sleep(0.5)  # Give time for cleanup
        
        config = MODELS[model_key]
        print(f"⏳ Loading {config['alias']}... (This may take a few seconds)")
        
        try:
            new_llm = Llama(
                model_path=config["model_path"],
                lora_path=config["lora_path"],
                lora_scale=1.0,
                n_ctx=2048,
                n_gpu_layers=-1,
                verbose=True
            )
            # Only update globals after successful load
            llm = new_llm
            current_model_name = model_key
            print(f"✅ {config['alias']} Loaded!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            # Restore old model name if load failed
            current_model_name = old_model_name
            raise e
        finally:
            is_loading = False

# 2. Load the Default Model on Startup
load_model("qwen")

# Initialize RAG Database
rag_utils.initialize_rag_db()

# 3. Define the Request Structure
class ChatRequest(BaseModel):
    instruction: str
    input_text: str = "" # Optional input (e.g., the math problem)

class ModelSwitchRequest(BaseModel):
    model: str

# Model Switching Endpoints
@app.post("/set_model")
def set_model(request: ModelSwitchRequest):
    try:
        load_model(request.model)
        return {"message": f"Switched to {MODELS[request.model]['alias']}", "current_model": request.model}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current_model")
def get_current_model():
    return {
        "current_model": current_model_name, 
        "alias": MODELS[current_model_name]["alias"],
        "available_models": list(MODELS.keys())
    }

# Helper: Intent Detection
def detect_intent(query: str) -> str:
    # Simple keyword matching for "Creation" intent
    creation_keywords = [
        "create", "generate", "make", "write", "compose", 
        "បង្កើត", "តែង", "សរសេរ", "រកនឹក"
    ]
    query_lower = query.lower()
    for word in creation_keywords:
        if word in query_lower:
            return "GENERATE"
    return "SOLVE"

# 4. The Endpoint
@app.post("/generate")
def generate_response(request: ChatRequest):
    global llm, is_loading
    
    # Prevent inference during model switching
    if is_loading:
        raise HTTPException(status_code=503, detail="Model is currently being loaded. Please wait.")
    
    if llm is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # Use the instruction as the user query for RAG
    user_query = request.instruction
    if request.input_text:
        user_query += f" {request.input_text}"

    try:
        # Retrieve context
        concept, exercise = rag_utils.retrieve_context(user_query)
        context_text = f"{concept}\n{exercise}"
        
        # Detect Intent
        intent = detect_intent(user_query)
        print(f"DEBUG: Detected Intent: {intent}")

        # --- DYNAMIC PROMPT STRATEGY ---
        if "seallm" in current_model_name.lower():
            # KHMER SEALLM STRATEGY (Khmer-Optimized)
            # SeaLLM uses <|im_start|> format but optimized for Southeast Asian languages
            if intent == "SOLVE":
                # Very low temperature for accuracy and instruction following
                temperature = 0.15
                repeat_penalty = 1.3
                
                # SeaLLM-specific: Put RAG instruction in FIRST USER TURN (not system)
                # System prompt should remain minimal as SeaLLM wasn't tuned for it
                prompt = f"""<|im_start|>system
អ្នកជាជំនួយការដែលមានប្រយោជន៍ និងត្រូវតែធ្វើតាមការណែនាំយ៉ាងតឹងរ៉ឹង។ អ្នកមិនត្រូវប្រើចំណេះដឹងខាងក្រៅ ឬសន្និដ្ឋានផ្ទាល់ខ្លួនឡើយ។ ប្រើតែព័ត៌មានពីឯកសារយោងប៉ុណ្ណោះ។</s><|im_start|>user
អ្នកជាគ្រូបង្រៀនថ្នាក់ទី១២ ជំនាញរូបវិទ្យា គណិតវិទ្យា ជីវវិទ្យា និងប្រវត្តិសាស្ត្រ។

ឯកសារយោង៖
{context_text}

សំណួរ៖ {user_query}

សេចក្តីណែនាំ៖ 
- សូមឆ្លើយសំណួរដោយប្រើតែព័ត៌មានពីឯកសារយោងខាងលើប៉ុណ្ណោះ។ កុំបន្ថែមព័ត៌មានខាងក្រៅ ឬសន្និដ្ឋានផ្ទាល់ខ្លួន។ បើឯកសារយោងមិនមានព័ត៌មានគ្រប់គ្រាន់ សូមឆ្លើយថា "ព័ត៌មានមិនគ្រប់គ្រាន់នៅក្នុងឯកសារយោង។"
- ចម្លើយត្រូវតែជាភាសាខ្មែរ។
- មុននឹងឆ្លើយ សូមគិតជាជំហាន៖ ១. រកព័ត៌មានពាក់ព័ន្ធពីឯកសារយោង។ ២. បញ្ជាក់ថាអ្នកបានយកពីឯកសារយោងណា។ ៣. បន្ទាប់មកឆ្លើយ។
- សម្រាប់គណិតវិទ្យា៖ បង្ហាញរូបមន្ត → ដោះស្រាយជាជំហាន → គណនា → ចម្លើយចុងក្រោយ។ ប្រើតែទិន្នន័យពីឯកសារយោង។
- សម្រាប់គំនិត ឬពន្យល់៖ ប្រើចំណុចសំខាន់ៗពីឯកសារយោង និងបញ្ជាក់ប្រភពពីឯកសារយោង។

ត្រូវតែធ្វើតាមសេចក្តីណែនាំនេះយ៉ាងតឹងរ៉ឹង បើមិនដូច្នោះទេ ចម្លើយមិនត្រឹមត្រូវ។</s><|im_start|>assistant
"""
            else:
                # Higher temperature for creativity in exercise generation
                temperature = 0.65
                repeat_penalty = 1.15
                
                # For generation, also use first user turn
                prompt = f"""<|im_start|>system
You are a helpful assistant.</s><|im_start|>user
អ្នកជាគ្រូបង្រៀនថ្នាក់ទី១២។ សូមបង្កើតលំហាត់ ឬពន្យល់គំនិតដូចខាងក្រោម៖

{user_query}

សេចក្តីណែនាំ៖ ឆ្លើយជាភាសាខ្មែរ។ ច្នៃប្រឌិតនិងមានលក្ខណៈអប់រំ។</s><|im_start|>assistant
"""

        else:
            # QWEN 2.5 STRATEGY (Instruction Following Focused)
            repeat_penalty = 1.1  # Default for Qwen
            if intent == "SOLVE":
                temperature = 0.1
                system_prompt = """You are an expert Khmer Grade 12 Tutor.
Your goal is to read the student’s question and provide an accurate solution.
Instructions:
1. Answer strictly in Khmer.
2. Use the provided [Context] to ensure accuracy.
3. If the problem involves calculation, follow this structure:
   - State the formula (តាមរូបមន្ត).
   - List given variables (ដោយ).
   - Perform calculation (យើងបាន).
   - State the final answer (ដូចនេះ).
4. If it is a conceptual question, explain clearly and concisely."""
            else:
                temperature = 0.7
                system_prompt = """You are a Khmer Grade 12 Teacher.
Your goal is to create new exercises or explain concepts clearly based on the user's request.
Instructions:
1. Answer strictly in Khmer.
2. Be creative and educational.
3. If creating an exercise, Do not provide the solution, only when user asks for it."""

            prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
[Context / ឯកសារយោង]
{context_text}

[Question / សំណួរ]
{user_query}
<|im_end|>
<|im_start|>assistant
"""
        
        print(f"DEBUG: Generated Prompt for {current_model_name}:\n{prompt}")

        def generate():
            # Yield prompt info first
            yield json.dumps({"type": "info", "prompt": prompt}) + "\n"

            # Prepare inference parameters
            inference_params = {
                "prompt": prompt,
                "max_tokens": 2048,
                "temperature": temperature,
                "stop": ["<|im_end|>", "<|endoftext|>", "</s>", "<|im_start|>"],
                "echo": False,
                "stream": True
            }
            
            # Add repeat_penalty for SeaLLM to prevent repetition
            if "seallm" in current_model_name.lower():
                inference_params["repeat_penalty"] = repeat_penalty
                inference_params["top_p"] = 0.9
                inference_params["top_k"] = 40
            
            # B. Run Inference
            stream = llm(**inference_params)
            
            for output in stream:
                token = output['choices'][0]['text']
                yield json.dumps({"type": "token", "text": token}) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Run Server (only if run directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)