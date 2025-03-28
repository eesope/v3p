from contextlib import asynccontextmanager
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# to run on cpu
from transformers import T5Tokenizer, T5ForConditionalGeneration

import os
import time

# https://huggingface.co/google/flan-t5-base
model_path = os.path.abspath("./")
os.environ["HUGGINGFACE_HUB_CACHE"] = model_path
model_name = "google/lan-t5-base"

# to make it lazily load
tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    print("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    print("Model and tokenizer loaded.")
    yield # server executing... 

    print("Cleaning up resources...") # coming back after server terminated
    del tokenizer 
    del model
    print("Resources cleaned up.")

app = FastAPI(lifespan=lifespan)

@app.get("/t2t")
async def generate_txt(prompt: str):   

    input_text = "translate English to French: What is your name?"
    # prompt = data.get("prompt", "Three cats are") # if @app.post
    start_inference = time.perf_counter()

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    result = tokenizer.decode(outputs[0])

    # output = model.generate(prompt, max_length=5, num_return_sequences=1)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    inference_time = time.perf_counter() - start_inference
    print(f"Inference for prompt '{prompt}' took {inference_time:.2f} seconds.")
    
    return {
        "prompt": input_text,
        "generated_text": result,
        "inference_time": inference_time
    }

@app.get("/")
async def health_check():
    return {"status": "OK"}
