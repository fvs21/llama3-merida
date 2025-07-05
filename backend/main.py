from typing import Dict
from fastapi import FastAPI
from contextlib import asynccontextmanager
from model import load_model, Llama3Merida
import gc

model: Llama3Merida = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    yield
    model = None
    gc.collect()
    

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate(prompt: str) -> Dict:
    response = model.chat(prompt)

    return {
        "response": response
    }