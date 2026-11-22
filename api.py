"""HTTP API for live scoring. Run: uvicorn api:app"""
import hashlib

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, create_model

from src.config import COLS, MODEL_FILE
from src.scorer import CycleOrderError, Scorer

# Loaded once when the server starts, not per request: loading takes far longer than predicting.
MODEL_VERSION = f"{MODEL_FILE.name}@{hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()[:12]}"
scorer = Scorer(joblib.load(MODEL_FILE))   # state lives in memory: lost when the server restarts

# One flight: an integer cycle and 24 required numbers. NaN and inf are refused (422), like missing fields.
Flight = create_model("Flight", __config__=ConfigDict(allow_inf_nan=False),
                      cycle=(int, ...), **{c: (float, ...) for c in COLS[2:]})

app = FastAPI(title="Turbofan RUL scoring")


@app.exception_handler(RequestValidationError)
def invalid_request(request, exc):
    # FastAPI's default 422 echoes the bad input back; a NaN can't be written as JSON, so that crashed with a 500.
    errors = [{k: v for k, v in e.items() if k != "input"} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/engines/{unit}/cycles")
def add_cycle(unit: int, flight: Flight):
    try:
        return scorer.score({"unit": unit, **flight.model_dump()})
    except CycleOrderError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/engines/{unit}")
def latest(unit: int):
    if unit not in scorer.latest:
        raise HTTPException(status_code=404, detail=f"engine {unit} has not sent any flights")
    return scorer.latest[unit]
