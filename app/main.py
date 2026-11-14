"""FastAPI app: POST /search, POST /search/multimodal, POST /sketches, GET /dinos and the static front-end."""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.search.ranker import Ranker, Result, build_ranker, load_config
from app.search.sketch import FusedResult, SketchReader, decode, ink_box, multimodal_search, png_1bit

STATIC = Path(__file__).resolve().parents[1] / "static"
# Drawings saved from the home board, one JSON object per line, to annotate (and maybe train on) later.
SKETCHES = Path(__file__).resolve().parents[1] / "data" / "desenhos.jsonl"
_sketches_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the models and embed the base once (cached on disk across restarts)."""
    cfg = load_config()
    app.state.ranker = build_ranker(cfg)
    app.state.sketch = SketchReader(cfg["sketch_model"], app.state.ranker.dinos)
    yield


app = FastAPI(title="Dino Search", lifespan=lifespan)
app.mount("/img", StaticFiles(directory=STATIC / "img"), name="img")


class SearchRequest(BaseModel):
    """Body of POST /search."""

    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=50)


def _ranker(request: Request) -> Ranker:
    """The Ranker built at startup."""
    return request.app.state.ranker


@app.post("/search")
def search(req: SearchRequest, request: Request) -> list[Result]:
    """Rank dinos by similarity to a free-text physical description."""
    return _ranker(request).search(req.query, req.top_k)


class MultimodalRequest(BaseModel):
    """Body of POST /search/multimodal: text, a drawing, or both."""

    query: str = Field(default="", max_length=500)
    desenho: str | None = Field(default=None, max_length=10_000_000, description="PNG data URL from the board")
    top_k: int = Field(default=5, ge=1, le=50)


class MultimodalResponse(BaseModel):
    """Fused ranking, plus what CLIP read from the drawing (null without one)."""

    grupo: str | None
    descricao_desenho: str | None
    resultados: list[FusedResult]


@app.post("/search/multimodal")
def search_multimodal(req: MultimodalRequest, request: Request) -> MultimodalResponse:
    """Fuse the drawing's group and description with the text search (weighted RRF)."""
    try:
        im = decode(req.desenho) if req.desenho else None
    except (ValueError, OSError) as e:
        raise HTTPException(422, f"desenho inválido: {e}") from e
    grupo, descricao, hits = multimodal_search(_ranker(request), request.app.state.sketch, req.query, im, req.top_k)
    return MultimodalResponse(grupo=grupo, descricao_desenho=descricao, resultados=hits)


class SketchRequest(BaseModel):
    """Body of POST /sketches."""

    desenho: str = Field(max_length=10_000_000, description="PNG data URL from the board")
    texto: str = Field(default="", max_length=500, description="what was in the search field")
    dinossauro: str = Field(default="", max_length=100, description="which dino it is; empty until annotated")


@app.post("/sketches", status_code=201)
def save_sketch(req: SketchRequest) -> dict:
    """Append a drawing to data/desenhos.jsonl; "dinossauro" stays empty until someone annotates it."""
    try:
        im = decode(req.desenho)
    except (ValueError, OSError) as e:
        raise HTTPException(422, f"desenho inválido: {e}") from e
    if not ink_box(im):
        raise HTTPException(422, "desenho em branco")
    line = {"salvo_em": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dinossauro": req.dinossauro.strip(),
            "texto": req.texto.strip(), "desenho": png_1bit(im)}
    with _sketches_lock, SKETCHES.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return {"salvo_em": line["salvo_em"]}


@app.get("/dinos")
def dinos(request: Request) -> list[dict]:
    """The full dataset as stored in data/dinos.json."""
    return [d.raw for d in _ranker(request).dinos]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The single-page front-end."""
    return FileResponse(STATIC / "index.html")


@app.get("/all", include_in_schema=False)
def all_dinos() -> FileResponse:
    """Grid with every dino in the dataset."""
    return FileResponse(STATIC / "all.html")


@app.get("/make-gt", include_in_schema=False)
def make_gt() -> FileResponse:
    """Draw each dino from its illustration to build a sketch ground truth."""
    return FileResponse(STATIC / "make-gt.html")
