"""End-to-end API check with the real model (slow: loads embeddings once)."""

import base64
import io
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_search_returns_ranked_explained_hits(client: TestClient) -> None:
    r = client.post("/search", json={"query": "três chifres e um escudo no pescoço", "top_k": 3})
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) == 3 and hits[0]["nome"] == "Triceratops"
    assert "chifres" in hits[0]["tracos_correspondentes"]
    assert hits[0]["score_final"] >= hits[1]["score_final"]
    assert 0 <= hits[0]["similaridade"] <= 1


def test_search_distinguishes_nasal_plates_from_back_plates(client: TestClient) -> None:
    response = client.post("/search", json={
        "query": "ceratopsiano sem chifres com uma placa grossa no nariz",
        "top_k": 1,
    })
    assert response.status_code == 200
    hit = response.json()[0]
    assert hit["nome"] == "Pachyrhinosaurus"
    assert set(hit["tracos_correspondentes"]) == {"ceratopsiano", "placa_nasal", "sem chifres"}


def test_reported_similarity_uses_raw_cosine_without_positive_attributes(client: TestClient) -> None:
    for query in ("qual é a capital da França?", "sem penas"):
        response = client.post("/search", json={"query": query, "top_k": 1})
        assert response.status_code == 200
        hit = response.json()[0]
        assert hit["penalidade"] == 0
        assert 0 < hit["similaridade"] == hit["score_embedding"] < 1


def test_search_validates_input(client: TestClient) -> None:
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422


def test_dinos_and_index(client: TestClient) -> None:
    assert len(client.get("/dinos").json()) == 64
    assert client.get("/").status_code == 200
    assert client.get("/all").status_code == 200
    assert client.get("/make-gt").status_code == 200


def test_every_dino_has_an_illustration_with_credits(client: TestClient) -> None:
    credits = client.get("/img/credits.json").json()
    for d in client.get("/dinos").json():
        r = client.get(f"/img/{d['nome']}.png")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png", d["nome"]
        assert credits[d["nome"]]["autor"] and credits[d["nome"]]["licenca"].startswith("CC")


def _sketch(nome: str) -> str:
    gt = json.loads(next(Path("eval/gt").glob("*.json")).read_text())
    return next(it["desenho"] for it in gt["itens"] if it["nome"] == nome)


def test_multimodal_reads_the_drawing_group(client: TestClient) -> None:
    r = client.post("/search/multimodal", json={"desenho": _sketch("Triceratops"), "top_k": 6})
    assert r.status_code == 200
    body = r.json()
    assert body["grupo"] == "Triceratops" and body["descricao_desenho"]
    group = set(json.loads(Path("data/map.json").read_text())["grupos"]["Triceratops"])
    assert {h["nome"] for h in body["resultados"]} == group  # the whole group comes first
    assert set(body["resultados"][0]["posicoes"]) == {"nome", "descrição"}


def test_multimodal_text_only_matches_search(client: TestClient) -> None:
    q = {"query": "pescoço longo e cauda chicote", "top_k": 5}
    fused = client.post("/search/multimodal", json=q).json()
    assert fused["grupo"] is None
    assert [h["nome"] for h in fused["resultados"]] == [h["nome"] for h in client.post("/search", json=q).json()]


def test_multimodal_blank_or_bad_drawing(client: TestClient) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (64, 32), "white").save(buf, "PNG")
    blank = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    assert client.post("/search/multimodal", json={"desenho": blank}).json() == {
        "grupo": None, "descricao_desenho": None, "resultados": []}
    assert client.post("/search/multimodal", json={"desenho": "data:image/png;base64,bm9wZQ=="}).status_code == 422


def test_sketches_append_one_line_per_drawing(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.main

    monkeypatch.setattr(app.main, "SKETCHES", tmp_path / "desenhos.jsonl")
    for texto in ("três chifres", ""):
        r = client.post("/sketches", json={"desenho": _sketch("Triceratops"), "texto": texto})
        assert r.status_code == 201
    lines = [json.loads(x) for x in (tmp_path / "desenhos.jsonl").read_text().splitlines()]
    assert [x["texto"] for x in lines] == ["três chifres", ""]
    assert lines[0]["dinossauro"] == "" and lines[0]["desenho"].startswith("data:image/png;base64,")
    im = Image.open(io.BytesIO(base64.b64decode(lines[0]["desenho"].split(",", 1)[1])))
    assert im.mode == "1"

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, "PNG")
    blank = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    assert client.post("/sketches", json={"desenho": blank}).status_code == 422
    assert len((tmp_path / "desenhos.jsonl").read_text().splitlines()) == 2
