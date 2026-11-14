"""Evaluate sketch recognition on the make-gt ground truth (eval/gt/*.json).

Instead of picking one of 64 dinos, a sketch is assigned to one of the 8 popular dinos
(POPULAR in app.search.ranker). data/map.json puts every dino in the group of the popular
one it looks most like, and a sketch is right when it lands in its dino's group. Dinos too
unlike any popular one are listed under "sem_grupo": their sketches are still classified
and shown, but left out of the metrics since no group is right for them.

Three rankings of the 8 groups, all zero-shot CLIP, nothing trained:
  nome              sketch vs. "a drawing of a <popular>, a dinosaur"
  descrição         CLIP reads posture and only the traits the 8 popular dinos have; they
                    become a Portuguese description that the app's hybrid search scores
  nome + descrição  both scores min-max normalized per sketch and summed

Usage:
    python -m eval.run_sketch_eval eval/gt/gt-desenhos-20261114-011535.json
    python -m eval.run_sketch_eval eval/gt/*.json --model clip-ViT-B-32 --json out.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.search.dinos import TRAITS
from app.search.extractor import extract_rules
from app.search.ranker import POPULAR, build_ranker, load_config
from eval.run_eval import fmt, metrics, rank_of_first_hit

ROOT = Path(__file__).resolve().parents[1]
DINOS = ROOT / "data" / "dinos.json"
MAP = ROOT / "data" / "map.json"

# Every trait the semantic search scores, and nothing else: CLIP prompt (English works
# best) -> Portuguese phrase the search's extractor reads back as that same trait.
SEARCH_TRAITS = {
    "chifres": ("horns on its face", "chifres"),
    "escudo": ("a big bony frill behind its head", "escudo"),
    "crista": ("a crest on its head", "crista"),
    "placas": ("tall plates along its back", "placas nas costas"),
    "placa_nasal": ("a thick bony lump on its nose", "placa no nariz"),
    "espinhos": ("spikes", "espinhos"),
    "espinhos_cauda": ("spikes on its tail", "espinhos na cauda"),
    "vela_dorsal": ("a sail on its back", "vela nas costas"),
    "armadura": ("armored body", "armadura"),
    "clava_cauda": ("a club at the end of its tail", "clava na cauda"),
    "penas": ("feathers", "penas"),
    "asas": ("wings", "asas"),
    "pescoco_longo": ("a very long neck", "pescoço longo"),
    "cauda_chicote": ("a long whip-like tail", "cauda chicote"),
    "patas_dianteiras_longas": ("front legs longer than its back legs", "patas dianteiras longas"),
    "garras_grandes": ("big claws", "garras grandes"),
    "espinho_polegar": ("a spike on its thumb", "espinho no polegar"),
    "cabeca_domo": ("a thick domed head", "cabeça em domo"),
    "bico": ("a beak", "bico"),
    "bico_de_pato": ("a duck bill", "bico de pato"),
    "bracos_curtos": ("tiny short arms", "braços curtos"),
    "cabeca_grande": ("a huge head", "cabeça grande"),
    "focinho_longo": ("a long crocodile-like snout", "focinho longo"),
    "pernas_longas": ("long slender legs", "pernas longas"),
    "semiaquatico": ("a body built for swimming", "semiaquático"),
    "ceratopsiano": ("a parrot beak, a frill and horns like a ceratopsian", "ceratopsiano"),
}
assert SEARCH_TRAITS.keys() == set(TRAITS), "SEARCH_TRAITS must list exactly the search's TRAITS"
assert all(extract_rules(pt).tracos == {t} for t, (_, pt) in SEARCH_TRAITS.items()), "phrase not read back as its trait"
# Postures the search scores; "facultativo" (two or four legs) can't be read off one pose.
POSTURES = {"bípede": "a dinosaur walking on two legs", "quadrúpede": "a dinosaur walking on four legs"}
NEUTRAL = "a drawing of a dinosaur"
TOP_TRAITS = 3  # beat thresholding on p(trait vs neutral), which picked ~8 traits per sketch


def describe(margin: np.ndarray, posture_scores: np.ndarray, keys: list[str]) -> tuple[str, set[str]]:
    """Portuguese query and trait set for one sketch, from its centered margins over `keys`."""
    picked = [keys[i] for i in np.argsort(-margin)[:TOP_TRAITS]]
    posture = list(POSTURES)[int(np.argmax(posture_scores))]
    phrases = [SEARCH_TRAITS[t][1] for t in picked]
    query = posture + (" com " + ", ".join(phrases[:-1]) + (" e " if len(phrases) > 1 else "") + phrases[-1] if phrases else "")
    return query, set(picked)


def decode(data_url: str) -> Image.Image:
    """PNG data URL from the make-gt canvas to an RGB image."""
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")


def frame(im: Image.Image, margin: float = 0.06) -> Image.Image:
    """Crop to the non-white content and pad it to a white square.

    CLIP resizes the short side to 224 and center-crops, so a 4:3 sketch loses its
    sides (head and tail) and a wide illustration loses almost half its width.
    Cropping first also makes a small doodle fill the frame, keeping thin strokes visible.
    """
    ink = im.convert("L").point(lambda p: 255 if p < 245 else 0).getbbox()
    if ink:
        im = im.crop(ink)
    w, h = im.size
    side = int(max(w, h) * (1 + 2 * margin))
    out = Image.new("RGB", (side, side), "white")
    out.paste(im, ((side - w) // 2, (side - h) // 2))
    return out


def load_groups(dataset: list[dict]) -> dict[str, str | None]:
    """data/map.json as dino -> popular group (None for "sem_grupo"), checking every dino
    sits in exactly one group or in "sem_grupo"."""
    m = json.loads(MAP.read_text())
    groups, ungrouped = m["grupos"], m["sem_grupo"]
    assert groups.keys() == POPULAR, "map.json groups must be exactly the POPULAR dinos"
    member_of: dict[str, str | None] = {d: g for g, members in groups.items() for d in members}
    assert all(member_of[g] == g for g in groups), "each popular dino must lead its own group"
    member_of |= dict.fromkeys(ungrouped)
    assert sum(map(len, groups.values())) + len(ungrouped) == len(member_of), "a dino is listed twice"
    assert member_of.keys() == {d["nome"] for d in dataset}, "map.json must cover exactly data/dinos.json"
    return member_of


def minmax(x: np.ndarray) -> np.ndarray:
    """Rescale each row to [0, 1] so scores on different scales can be summed."""
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    return (x - lo) / np.where(hi > lo, hi - lo, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path, help="ground-truth JSON files from /make-gt")
    ap.add_argument("--model", default="clip-ViT-L-14", help="sentence-transformers CLIP model")
    ap.add_argument("--no-frame", action="store_true", help="feed images uncropped (CLIP center-crops them)")
    ap.add_argument("--json", type=Path, help="also write per-sketch results here")
    args = ap.parse_args()
    prep = (lambda im: im) if args.no_frame else frame

    items = [it for f in args.files for it in json.loads(f.read_text())["itens"]]
    dataset = json.loads(DINOS.read_text())
    names = [d["nome"] for d in dataset]
    group_of = load_groups(dataset)
    popular = sorted(POPULAR)
    traits = {d["nome"]: {t["traco"] for t in d["tracos"]} for d in dataset}
    # Only traits some popular dino has: the ones that can tell the 8 groups apart.
    keys = [t for t in SEARCH_TRAITS if any(t in traits[p] for p in popular)]
    model = SentenceTransformer(args.model)
    enc = lambda xs: model.encode(xs, normalize_embeddings=True, convert_to_numpy=True)  # noqa: E731

    sketches = enc([prep(decode(it["desenho"])) for it in items])
    trait_emb = enc([f"{NEUTRAL} with {SEARCH_TRAITS[t][0]}" for t in keys])
    neutral = enc([NEUTRAL])[0]
    posture_emb = enc([f"a drawing of {p}" for p in POSTURES.values()])
    # How much each trait prompt beats the neutral one, centered per trait over all sketches:
    # some prompts ("tiny short arms") win on every doodle, so only above-average wins count.
    # ponytail: centering needs the whole batch; a live endpoint would center on stored sketch stats.
    margins = sketches @ trait_emb.T - (sketches @ neutral)[:, None]
    margins -= margins.mean(axis=0)
    described = [describe(mg, posture_emb @ s, keys) for mg, s in zip(margins, sketches)]
    ranker = build_ranker(load_config())
    search = [{r.nome: r.score_final for r in ranker.search(q, top_k=len(names))} for q, _ in described]
    desc_scores = np.array([[row[p] for p in popular] for row in search])
    name_scores = sketches @ enc([f"a drawing of a {p}, a dinosaur" for p in popular]).T
    scores = {
        "nome": name_scores,
        "descrição": desc_scores,
        "nome + descrição": minmax(name_scores) + minmax(desc_scores),
    }
    rankings = {m: [[popular[j] for j in np.argsort(-row)] for row in sc] for m, sc in scores.items()}
    expected = [group_of[it["nome"]] for it in items]
    ranks = {m: [rank_of_first_hit(r, [g]) if g else None for r, g in zip(rows, expected)] for m, rows in rankings.items()}
    scored = {m: [r for r, g in zip(rk, expected) if g] for m, rk in ranks.items()}
    n_out = expected.count(None)

    print(f"modelo: {args.model}  |  {len(items) - n_out} desenhos com grupo (+{n_out} sem grupo, fora das métricas)  |  "
          f"{len(popular)} grupos (acaso: top-1 {1 / len(popular):.3f})")
    print(f"traços que o CLIP procura ({len(keys)}): {', '.join(keys)}\n")
    print("| método | Top-1 | Top-3 | MRR |\n|---|---|---|---|")
    for m, rk in scored.items():
        print(f"| {m} | {fmt(metrics(rk))} |")

    # How well CLIP read the traits, against the dataset's own.
    truth = {n: ts & set(keys) for n, ts in traits.items()}
    tp = sum(len(picked & truth[it["nome"]]) for (_, picked), it in zip(described, items))
    n_picked = sum(len(picked) for _, picked in described)
    n_true = sum(len(truth[it["nome"]]) for it in items)
    print(f"\ntraços lidos pelo CLIP: precisão {tp / max(n_picked, 1):.2f} ({tp}/{n_picked}), "
          f"revocação {tp / max(n_true, 1):.2f} ({tp}/{n_true})\n")
    print("| dinossauro | grupo | nome | descrição | nome + descrição | descrição gerada | traços reais |\n"
          "|---|---|---|---|---|---|---|")
    for i, ((q, _), it) in enumerate(zip(described, items)):
        pos = " | ".join(f"{rankings[m][i][0]}{'' if ranks[m][i] in (1, None) else f' (#{ranks[m][i]})'}" for m in scores)
        print(f"| {it['nome']} | {expected[i] or 'sem grupo'} | {pos} | {q} | {', '.join(sorted(truth[it['nome']])) or '-'} |")

    if args.json:
        args.json.write_text(json.dumps({
            "modelo": args.model,
            "resumo": {m: metrics(rk) for m, rk in scored.items()},
            "itens": [{
                "nome": it["nome"], "grupo": expected[i], "query": q, "lidos": sorted(picked, key=keys.index),
                "reais": sorted(truth[it["nome"]]),
                "metodos": {m: {"posicao": ranks[m][i], "top3": rankings[m][i][:3]} for m in scores},
            } for i, ((q, picked), it) in enumerate(zip(described, items))],
        }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
