from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

app = FastAPI()
model_path = os.path.abspath("./mpt-7b-storywriter")
os.environ["HUGGINGFACE_HUB_CACHE"] = model_path
model_name = "mosaicml/mpt-7b-storywriter"

# to make it lazily load
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

@app.get("/generate_story")
async def generate_story(prompt: str = "Three cats are"):
    tokenizer, model = load_model()
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=5, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {
        "prompt": prompt,
        "generated_text": generated_text
    }

@app.get("/")
async def health_check():
    return {"status": "OK"}


# manual start server:
# uvicorn model_api:app --reload --log-level debug

# query via url
# http://url/generate_story?prompt=Three%20cats%20are

# runtime command
# uvicorn model_api:app --host 0.0.0.0 --port 8000
