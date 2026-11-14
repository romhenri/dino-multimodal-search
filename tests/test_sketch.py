from app.search.sketch import RRF_K, rrf


def test_nome_decides_the_group_and_the_others_only_break_ties() -> None:
    last = 64
    # A's group is one rank ahead in nome, but B leads descrição and busca: A still wins.
    for g in range(1, 8):
        s = rrf({"nome": {"A": g, "B": g + 1}, "descrição": {"A": last, "B": 1}, "busca": {"A": last, "B": 1}})
        assert s["A"] > s["B"], g
    # Same group: the other rankings decide.
    s = rrf({"nome": {"A": 1, "B": 1}, "descrição": {"A": 2, "B": 1}, "busca": {"A": 3, "B": 2}})
    assert s["B"] > s["A"]


def test_missing_from_a_ranking_scores_nothing_there() -> None:
    assert rrf({"busca": {"A": 1}, "nome": {}}) == {"A": 1 / (RRF_K + 1)}
