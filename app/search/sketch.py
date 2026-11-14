"""Sketch search: zero-shot CLIP reads a drawing, and its rankings are fused with the text search.

A drawing gives two rankings, nothing trained:
  nome       the drawing vs. "a drawing of a <popular>, a dinosaur" for the 8 popular dinos;
             every dino takes the rank of its group in data/map.json
  descrição  CLIP reads posture and the traits the popular dinos have; they become a
             Portuguese description that the hybrid search ranks
Typed text adds a third, busca. Weighted reciprocal rank fusion merges them.

Trait margins are centered by their mean over the ground-truth sketches, stored in
data/sketch_trait_means.json. Regenerate it after changing the model or the prompts:
    python -m app.search.sketch eval/gt/*.json
"""

from __future__ import annotations

import base64
import io
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.search.dinos import TRAITS, Dino
from app.search.extractor import extract_rules
from app.search.ranker import POPULAR, Ranker, Result

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "data" / "map.json"
MEANS = ROOT / "data" / "sketch_trait_means.json"

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

# Reciprocal rank fusion: score = sum of weight / (RRF_K + rank) over the rankings a dino is in.
# nome must decide the group and the others only order dinos inside it, so one rank down in
# nome has to cost more than descrição and busca can ever add: 100 * (1/67 - 1/68) > 2 * (1/61 - 1/124).
RRF_K = 60
RRF_WEIGHTS = {"nome": 100.0, "descrição": 1.0, "busca": 1.0}


def describe(margin: np.ndarray, posture_scores: np.ndarray, keys: list[str]) -> tuple[str, set[str]]:
    """Portuguese query and trait set for one sketch, from its centered margins over `keys`."""
    picked = [keys[i] for i in np.argsort(-margin)[:TOP_TRAITS]]
    posture = list(POSTURES)[int(np.argmax(posture_scores))]
    phrases = [SEARCH_TRAITS[t][1] for t in picked]
    query = posture + (" com " + ", ".join(phrases[:-1]) + (" e " if len(phrases) > 1 else "") + phrases[-1] if phrases else "")
    return query, set(picked)


def decode(data_url: str) -> Image.Image:
    """PNG data URL from a canvas to an RGB image."""
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[-1]))).convert("RGB")


def png_1bit(im: Image.Image) -> str:
    """Black-on-white 1-bit PNG data URL, the format of the make-gt ground truth."""
    buf = io.BytesIO()
    im.convert("L").point(lambda v: 255 if v >= 128 else 0).convert("1", dither=Image.Dither.NONE).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ink_box(im: Image.Image) -> tuple[int, int, int, int] | None:
    """Bounding box of the non-white pixels, None for a blank drawing."""
    return im.convert("L").point(lambda p: 255 if p < 245 else 0).getbbox()


def frame(im: Image.Image, margin: float = 0.06) -> Image.Image:
    """Crop to the non-white content and pad it to a white square.

    CLIP resizes the short side to 224 and center-crops, so a 4:3 sketch loses its
    sides (head and tail) and a wide illustration loses almost half its width.
    Cropping first also makes a small doodle fill the frame, keeping thin strokes visible.
    """
    if ink := ink_box(im):
        im = im.crop(ink)
    w, h = im.size
    side = int(max(w, h) * (1 + 2 * margin))
    out = Image.new("RGB", (side, side), "white")
    out.paste(im, ((side - w) // 2, (side - h) // 2))
    return out


def load_groups(names: list[str]) -> dict[str, str | None]:
    """data/map.json as dino -> popular group (None for "sem_grupo"), checking every dino
    sits in exactly one group or in "sem_grupo"."""
    m = json.loads(MAP.read_text())
    groups, ungrouped = m["grupos"], m["sem_grupo"]
    assert groups.keys() == POPULAR, "map.json groups must be exactly the POPULAR dinos"
    member_of: dict[str, str | None] = {d: g for g, members in groups.items() for d in members}
    assert all(member_of[g] == g for g in groups), "each popular dino must lead its own group"
    member_of |= dict.fromkeys(ungrouped)
    assert sum(map(len, groups.values())) + len(ungrouped) == len(member_of), "a dino is listed twice"
    assert member_of.keys() == set(names), "map.json must cover exactly data/dinos.json"
    return member_of


def trait_keys(dinos: list[Dino]) -> list[str]:
    """Traits some popular dino has: the ones that can tell the 8 groups apart."""
    return [t for t in SEARCH_TRAITS if any(t in d.tracos for d in dinos if d.nome in POPULAR)]


def rrf(rankings: dict[str, dict[str, int]]) -> dict[str, float]:
    """Weighted reciprocal rank fusion of {ranking: {dino: 1-based rank}}; a dino missing
    from a ranking gets nothing from it."""
    scores: dict[str, float] = defaultdict(float)
    for name, ranks in rankings.items():
        for dino, rank in ranks.items():
            scores[dino] += RRF_WEIGHTS[name] / (RRF_K + rank)
    return scores


class SketchReader:
    """CLIP with every prompt embedded once; reads one drawing at a time."""

    def __init__(self, model: str, dinos: list[Dino]) -> None:
        self.model = SentenceTransformer(model)
        self.popular = sorted(POPULAR)
        self.group_of = load_groups([d.nome for d in dinos])
        self.keys = trait_keys(dinos)
        means = json.loads(MEANS.read_text())
        assert means["modelo"] == model and list(means["tracos"]) == self.keys, f"stale {MEANS.name}: rerun python -m app.search.sketch"
        self.means = np.array(list(means["tracos"].values()))
        self.name_emb = self.encode([f"a drawing of a {p}, a dinosaur" for p in self.popular])
        self.trait_emb = self.encode([f"{NEUTRAL} with {SEARCH_TRAITS[t][0]}" for t in self.keys])
        self.neutral = self.encode([NEUTRAL])[0]
        self.posture_emb = self.encode([f"a drawing of {p}" for p in POSTURES.values()])

    def encode(self, xs: list) -> np.ndarray:
        """Normalized CLIP embeddings of texts or images."""
        return self.model.encode(xs, normalize_embeddings=True, convert_to_numpy=True)

    def read(self, im: Image.Image) -> tuple[list[str], str] | None:
        """The 8 groups from most to least alike and the Portuguese description, None if blank."""
        if not ink_box(im):
            return None
        s = self.encode([frame(im)])[0]
        groups = [self.popular[j] for j in np.argsort(-(self.name_emb @ s))]
        margin = self.trait_emb @ s - self.neutral @ s - self.means
        return groups, describe(margin, self.posture_emb @ s, self.keys)[0]


@dataclass
class FusedResult(Result):
    """A text-search Result plus its fusion score and its rank in each ranking (also the API schema)."""

    score_rrf: float = 0.0
    posicoes: dict[str, int] = field(default_factory=dict)


def multimodal_search(ranker: Ranker, reader: SketchReader, query: str, im: Image.Image | None,
                      top_k: int) -> tuple[str | None, str | None, list[FusedResult]]:
    """(group CLIP read, description CLIP wrote, fused top-k) for text and/or a drawing."""
    n = len(ranker.dinos)
    rankings: dict[str, dict[str, int]] = {}
    group = description = None
    shown: list[Result] = []
    if im is not None and (read := reader.read(im)):
        groups, description = read
        group = groups[0]
        rankings["nome"] = {d: groups.index(g) + 1 for d, g in reader.group_of.items() if g}
        shown = ranker.search(description, n)
        rankings["descrição"] = {r.nome: i + 1 for i, r in enumerate(shown)}
    if query.strip():
        shown = ranker.search(query, n)  # cards explain the typed text when there is one
        rankings["busca"] = {r.nome: i + 1 for i, r in enumerate(shown)}
    scores = rrf(rankings)
    by_name = {r.nome: r for r in shown}
    top = sorted(scores, key=lambda d: -scores[d])[:top_k]
    return group, description, [
        FusedResult(**asdict(by_name[d]), score_rrf=round(scores[d], 5),
                    posicoes={k: ranks[d] for k, ranks in rankings.items() if d in ranks})
        for d in top
    ]


def main() -> None:
    """Write data/sketch_trait_means.json from ground-truth files given as arguments."""
    from app.search.dinos import load_dinos
    from app.search.ranker import load_config

    model = load_config()["sketch_model"]
    keys = trait_keys(load_dinos())
    clip = SentenceTransformer(model)
    enc = lambda xs: clip.encode(xs, normalize_embeddings=True, convert_to_numpy=True)  # noqa: E731
    items = [it for f in sys.argv[1:] for it in json.loads(Path(f).read_text())["itens"]]
    s = enc([frame(decode(it["desenho"])) for it in items])
    margins = s @ enc([f"{NEUTRAL} with {SEARCH_TRAITS[t][0]}" for t in keys]).T - (s @ enc([NEUTRAL])[0])[:, None]
    MEANS.write_text(json.dumps({"modelo": model, "desenhos": len(items),
                                 "tracos": dict(zip(keys, map(float, margins.mean(axis=0))))}, ensure_ascii=False, indent=1))
    print(f"{MEANS.relative_to(ROOT)}: {len(items)} desenhos, {len(keys)} traços")


if __name__ == "__main__":
    main()
