"""
api/main.py

FastAPI app entrypoint. Everything expensive (the LLM, the embedding
model, the FAISS index) is loaded ONCE here, at startup, via the lifespan
context manager — never per-request. Routes pull these via Depends()
functions defined below, which just read from app.state.

LLM_BACKEND env var controls how generation is wired up:
  - "transformers" (default): loads Phi-4-mini-instruct via
    AutoModelForCausalLM + BitsAndBytes NF4 4-bit quantization, on your
    RTX 50-series GPU. This is what actually runs in production/dev on
    your machine.
  - "stub": swaps in a fake generate_fn that echoes the prompt instead of
    running a real model. Exists so this file (and anything built on top
    of it) can be tested on a machine with no GPU at all — e.g. CI, or
    this sandbox. Never set this in a real deployment.

This is the same swap-lever pattern as DATABASE_URL in api/db.py and
EMBEDDING_MODEL in pipeline/embed.py — one env var, no code changes.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from api.db import init_db
from pipeline.anonymize import PIIAnonymizer
from pipeline.embed import Embedder, load_index
from pipeline.rag import RAGPipeline
from pipeline.output_filter import OutputFilter

LLM_BACKEND = os.environ.get("LLM_BACKEND", "transformers")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "microsoft/Phi-4-mini-instruct")


def load_transformers_generate_fn():
    """Real backend: loads Phi-4-mini-instruct in 4-bit NF4 quantization.
    Only imports torch/transformers/bitsandbytes here, not at module level,
    so LLM_BACKEND=stub can run on a machine that doesn't even have a GPU
    build of torch installed."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading LLM: {LLM_MODEL_NAME} (NF4 4-bit) ...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
    )
    print("LLM loaded.")

    def generate_fn(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # generate() returns prompt + completion concatenated — strip the
        # echoed prompt so callers only get the new text.
        return full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

    return generate_fn


def load_stub_generate_fn():
    """Test-only backend — no model, no GPU required. Echoes back a fixed
    marker plus the tail of the prompt, purely so callers can assert
    something deterministic in tests."""
    print("LLM_BACKEND=stub — using fake generation, NOT calling any real model.")

    def generate_fn(prompt: str) -> str:
        return "[STUB SUGGESTION] " + prompt[-80:]

    return generate_fn


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: load every singleton once ---
    init_db()

    anonymizer = PIIAnonymizer()
    output_filter = OutputFilter(anonymizer)  # reuses the same compiled regexes
    embedder = Embedder()
    index, metadata = load_index()

    if LLM_BACKEND == "stub":
        generate_fn = load_stub_generate_fn()
    else:
        generate_fn = load_transformers_generate_fn()

    rag = RAGPipeline(embedder, index, metadata, generate_fn)

    app.state.anonymizer = anonymizer
    app.state.output_filter = output_filter
    app.state.rag = rag

    yield
    # --- Shutdown: nothing to clean up explicitly; process exit handles it ---


app = FastAPI(title="LoopSupport API", lifespan=lifespan)


# --- Dependency providers, used by api/routes/*.py via Depends() ---------

def get_anonymizer(request: Request) -> PIIAnonymizer:
    return request.app.state.anonymizer


def get_output_filter(request: Request) -> OutputFilter:
    return request.app.state.output_filter


def get_rag(request: Request) -> RAGPipeline:
    return request.app.state.rag


@app.get("/health")
def health():
    return {"status": "ok"}


# Routers are registered here once they exist:
# from api.routes import suggest, feedback, escalate, metrics, admin
# app.include_router(suggest.router)
# app.include_router(feedback.router)
# app.include_router(escalate.router)
# app.include_router(metrics.router)
# app.include_router(admin.router)
from api.routes import suggest, feedback  # noqa: E402
app.include_router(suggest.router)
app.include_router(feedback.router)