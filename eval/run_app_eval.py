"""Evaluate the app's own sketch search (CLIP + weighted RRF) on the make-gt ground truth.

Every drawing goes through app.search.sketch.multimodal_search exactly as POST
/search/multimodal runs it with an empty text field, and the result is written as a
self-contained HTML page: metrics, per-group hits, and for each drawing the group CLIP read,
the description it wrote and the fused ranking with each dino's rank in every RRF ranking.

    python -m eval.run_app_eval eval/gt/*.json            # writes eval/eval.html
    python -m eval.run_app_eval eval/gt/*.json --html out.html

Caveat: data/sketch_trait_means.json is computed on these same drawings, so the numbers are
somewhat optimistic (cross-validated group top-1 was 0.750 against 0.818 in-sample).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app.search.ranker import build_ranker, load_config
from app.search.sketch import RRF_WEIGHTS, SketchReader, decode, multimodal_search
from eval.run_eval import metrics

ROOT = Path(__file__).resolve().parents[1]
SHOWN = 8  # fused results listed per drawing


def evaluate(files: list[Path]) -> dict:
    """Run the app search on every drawing and collect what the page shows."""
    cfg = load_config()
    ranker = build_ranker(cfg)
    reader = SketchReader(cfg["sketch_model"], ranker.dinos)
    n = len(ranker.dinos)
    rows = []
    for f in files:
        for it in json.loads(f.read_text())["itens"]:
            grupo, descricao, hits = multimodal_search(ranker, reader, "", decode(it["desenho"]), n)
            names = [h.nome for h in hits]
            expected = reader.group_of[it["nome"]]
            group_order = list(dict.fromkeys(reader.group_of[d] for d in names if reader.group_of[d]))
            rows.append({
                "arquivo": f.name, "nome": it["nome"], "grupo": expected, "desenho": it["desenho"],
                "grupo_lido": grupo, "descricao": descricao,
                "posicao_grupo": group_order.index(expected) + 1 if expected else None,
                "posicao_especie": names.index(it["nome"]) + 1,
                "resultados": [{"nome": h.nome, "grupo": reader.group_of[h.nome], "score_rrf": h.score_rrf,
                                "posicoes": h.posicoes} for h in hits[:SHOWN]],
            })
    grouped = [r for r in rows if r["grupo"]]
    per_group: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in grouped:
        g = per_group[r["grupo"]]
        g[0] += 1
        g[1] += r["posicao_grupo"] == 1
        g[2] += r["posicao_especie"] == 1
    return {
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "modelo": cfg["sketch_model"], "pesos": RRF_WEIGHTS,
        "arquivos": [f.name for f in files], "desenhos": len(rows),
        "grupo": metrics([r["posicao_grupo"] for r in grouped]),
        "especie": metrics([r["posicao_especie"] for r in rows]),
        "representante_primeiro": sum(r["resultados"][0]["nome"] == r["grupo"] for r in grouped if r["posicao_grupo"] == 1),
        "grupos_certos": sum(r["posicao_grupo"] == 1 for r in grouped),
        "por_grupo": {g: {"desenhos": a, "grupo_top1": b, "especie_top1": c}
                      for g, (a, b, c) in sorted(per_group.items(), key=lambda x: -x[1][0])},
        "itens": rows,
    }


PAGE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dino Search eval</title>
<style>
  :root { --bg: #f7f6f2; --card: #ffffff; --ink: #1d1c1a; --muted: #6b6860; --line: #e3e0d8;
          --accent: #b45309; --chip: #fbefe3; --ok: #2f7d4f; --bad: #b0213a; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #151614; --card: #1e201d; --ink: #ecebe6; --muted: #9a978f; --line: #30332e;
            --accent: #f59e57; --chip: #33261b; --ok: #6cc28f; --bad: #f0707f; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1100px; margin: 0 auto; padding: 40px 16px 64px; }
  h1 { font-size: 1.6rem; margin: 0 0 4px; }
  h2 { font-size: 1.1rem; margin: 32px 0 12px; }
  .sub, .muted { color: var(--muted); }
  .sub { margin: 0 0 24px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; }
  .stat { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
  .stat b { display: block; font-size: 1.6rem; font-variant-numeric: tabular-nums; }
  .stat span { color: var(--muted); font-size: .85rem; }
  table { width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
  th { color: var(--muted); font-weight: 600; font-size: .85rem; }
  tr:last-child td { border-bottom: 0; }
  .bar { display: inline-block; height: 6px; border-radius: 3px; background: var(--accent); vertical-align: middle; margin-left: 8px; }
  .filters { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 16px; }
  .filters button, .filters select { font: inherit; padding: 6px 12px; border-radius: 999px; border: 1px solid var(--line);
                                     background: transparent; color: var(--muted); cursor: pointer; }
  .filters button[aria-pressed="true"] { color: var(--bg); background: var(--accent); border-color: var(--accent); }
  .card { background: var(--card); border: 1px solid var(--line); border-left: 4px solid var(--ok); border-radius: 10px;
          padding: 16px; margin-bottom: 12px; display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 18px; }
  .card.bad { border-left-color: var(--bad); }
  .card img { width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #fff; border: 1px solid var(--line); border-radius: 8px; }
  .card h3 { margin: 0; font-size: 1.05rem; }
  .tag { display: inline-block; font-size: .78rem; font-weight: 600; padding: 1px 8px; border-radius: 999px; margin-left: 6px; }
  .tag.ok { color: var(--ok); border: 1px solid currentColor; }
  .tag.bad { color: var(--bad); border: 1px solid currentColor; }
  .read { margin: 6px 0 10px; color: var(--muted); font-size: .9rem; }
  .read strong { color: var(--ink); }
  ol { margin: 0; padding-left: 22px; }
  ol li { padding: 2px 0; }
  li.hit { font-weight: 700; }
  li.same { color: var(--ink); }
  li.other { color: var(--muted); }
  .pos { color: var(--muted); font-size: .8rem; font-weight: 400; font-variant-numeric: tabular-nums; }
  .small { font-size: .8rem; }
  @media (max-width: 640px) { .card { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <h1>Eval da busca por desenho</h1>
  <p class="sub" id="sub"></p>
  <div class="stats" id="stats"></div>
  <h2>Por grupo</h2>
  <table id="groups"></table>
  <h2>Desenhos</h2>
  <div class="filters" id="filters">
    <button type="button" data-f="all" aria-pressed="true">Todos</button>
    <button type="button" data-f="bad" aria-pressed="false">Grupo errado</button>
    <button type="button" data-f="species" aria-pressed="false">Grupo certo, espécie errada</button>
    <select id="group-filter" aria-label="Filtrar por grupo"><option value="">Todos os grupos</option></select>
  </div>
  <div id="cards"></div>
</main>
<script>
const DATA = __DATA__;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => (x * 100).toFixed(1) + "%";
const short = { "nome": "nome", "descrição": "desc", "busca": "busca", "popularidade": "pop" };

$("sub").textContent = `${DATA.desenhos} desenhos de ${DATA.arquivos.join(", ")} · ${DATA.modelo} · RRF k=60, pesos ` +
  Object.entries(DATA.pesos).map(([k, v]) => `${k} ${v}`).join(", ") + ` · gerado em ${DATA.gerado_em}. ` +
  "Mesma busca do app (POST /search/multimodal), só com o desenho e o texto vazio.";

const stat = (v, label) => `<div class="stat"><b>${v}</b><span>${label}</span></div>`;
$("stats").innerHTML = [
  stat(pct(DATA.grupo.top1), "grupo Top-1"), stat(pct(DATA.grupo.top3), "grupo Top-3"), stat(DATA.grupo.mrr.toFixed(3), "grupo MRR"),
  stat(pct(DATA.especie.top1), "espécie Top-1"), stat(pct(DATA.especie.top3), "espécie Top-3"), stat(DATA.especie.mrr.toFixed(3), "espécie MRR"),
  stat(`${DATA.representante_primeiro}/${DATA.grupos_certos}`, "representante em 1º quando o grupo acerta"),
].join("");

$("groups").innerHTML = "<tr><th>grupo</th><th>desenhos</th><th>grupo certo</th><th>espécie exata</th></tr>" +
  Object.entries(DATA.por_grupo).map(([g, v]) => `<tr><td>${esc(g)}</td><td>${v.desenhos}</td>
    <td>${v.grupo_top1}/${v.desenhos}<span class="bar" style="width:${60 * v.grupo_top1 / v.desenhos}px"></span></td>
    <td>${v.especie_top1}/${v.desenhos}</td></tr>`).join("");
$("group-filter").append(...Object.keys(DATA.por_grupo).map((g) => new Option(g, g)));

function card(r) {
  const okGroup = r.posicao_grupo === 1;
  const list = r.resultados.map((h) => {
    const cls = h.nome === r.nome ? "hit" : h.grupo === r.grupo ? "same" : "other";
    const pos = Object.entries(h.posicoes).map(([k, p]) => `${short[k] || k} #${p}`).join(" · ");
    return `<li class="${cls}">${esc(h.nome)} <span class="pos">${h.score_rrf.toFixed(4)} · ${pos}</span></li>`;
  }).join("");
  return `<article class="card ${okGroup ? "" : "bad"}">
    <div><img src="${r.desenho}" alt="Desenho de ${esc(r.nome)}" loading="lazy"><div class="small muted">${esc(r.arquivo)}</div></div>
    <div>
      <h3>${esc(r.nome)}<span class="tag ${okGroup ? "ok" : "bad"}">${okGroup ? "grupo certo" : `grupo #${r.posicao_grupo ?? "-"}`}</span>
        <span class="tag ${r.posicao_especie === 1 ? "ok" : "bad"}">espécie #${r.posicao_especie}</span></h3>
      <div class="read">Grupo esperado <strong>${esc(r.grupo ?? "sem grupo")}</strong> · lido como <strong>${esc(r.grupo_lido)}</strong><br>
        Descrição do CLIP: ${esc(r.descricao)}</div>
      <ol>${list}</ol>
      ${r.posicao_especie > r.resultados.length ? `<div class="small muted">${esc(r.nome)} ficou em #${r.posicao_especie}.</div>` : ""}
    </div>
  </article>`;
}

let filter = "all";
function render() {
  const g = $("group-filter").value;
  const rows = DATA.itens.filter((r) => (!g || r.grupo === g) &&
    (filter === "all" || (filter === "bad" ? r.posicao_grupo !== 1 : r.posicao_grupo === 1 && r.posicao_especie !== 1)));
  $("cards").innerHTML = rows.map(card).join("") || '<p class="muted">Nenhum desenho com esse filtro.</p>';
}
for (const b of document.querySelectorAll("[data-f]")) {
  b.onclick = () => {
    filter = b.dataset.f;
    for (const o of document.querySelectorAll("[data-f]")) o.setAttribute("aria-pressed", o === b);
    render();
  };
}
$("group-filter").onchange = render;
render();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path, help="ground-truth JSON files from /make-gt or /annotate")
    ap.add_argument("--html", type=Path, default=ROOT / "eval" / "eval.html", help="page to write")
    args = ap.parse_args()
    data = evaluate(args.files)
    args.html.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    g, e = data["grupo"], data["especie"]
    print(f"{data['desenhos']} desenhos | grupo top1 {g['top1']:.3f} top3 {g['top3']:.3f} MRR {g['mrr']:.3f} | "
          f"espécie top1 {e['top1']:.3f} top3 {e['top3']:.3f} MRR {e['mrr']:.3f} | {args.html}")


if __name__ == "__main__":
    main()
