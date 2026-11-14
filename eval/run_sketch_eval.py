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
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.search.ranker import POPULAR, build_ranker, load_config
from app.search.sketch import NEUTRAL, POSTURES, SEARCH_TRAITS, decode, describe, frame, load_groups
from eval.run_eval import fmt, metrics, rank_of_first_hit

ROOT = Path(__file__).resolve().parents[1]
DINOS = ROOT / "data" / "dinos.json"


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
    group_of = load_groups(names)
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
