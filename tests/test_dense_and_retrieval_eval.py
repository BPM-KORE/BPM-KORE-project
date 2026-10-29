import json

import pytest

from kore.cli import main
from kore.dense import HybridRetriever
from kore.pipeline import KOREPipeline, PipelineConfig, build_retriever, run_experiment, set_encoder
from kore.retrieval import BM25Retriever
from kore.retrieval_eval import compare_retrievers, parse_variant
from kore.schemas import EvalQuestion, KnowledgeItem

SAMPLE_K, SAMPLE_Q = "data/sample_knowledge.csv", "data/sample_eval.csv"

# 같은 뜻의 낱말을 같은 차원에 모으는 가짜 인코더. "표현이 바뀐 질문"을 흉내 낸다.
CONCEPTS = [("임금", "급여", "월급"), ("해고", "퇴직", "그만두"), ("휴가", "연차", "쉬는"), ("화성", "붉은 행성"), ("커피",)]


class ConceptEncoder:
    name = "fake-concepts"

    def __init__(self):
        self.calls = []

    def encode(self, texts, kind):
        self.calls.append(kind)
        out = []
        for t in texts:
            v = [float(sum(t.count(w) for w in words)) for words in CONCEPTS]
            out.append(v if any(v) else [0.0] * (len(CONCEPTS) - 1) + [0.01])
        return out


def item(i, text, domain="law"):
    return KnowledgeItem(i, domain, i, text, "local:t", "test")


DOCS = [
    item("d_pay", "사용자는 근로자에게 매달 급여를 지급해야 한다. 지급일은 정해 둔다."),
    item("d_fire", "사용자가 근로자를 퇴직시키려면 30일 전에 알려야 한다."),
    item("d_leave", "1년 동안 80% 이상 출근한 근로자에게는 15일의 연차를 준다."),
    item("d_mars", "붉은 행성은 태양계의 네 번째 행성이다.", domain="wiki"),
    item("d_noise", "근로자 근로자 근로자 사업장 안내문", domain="guide"),
]


def test_hybrid_finds_a_paraphrased_document_that_bm25_misses():
    q = "임금은 언제 주나요?"  # 문서에는 '임금'이 없고 '급여'가 있다
    bm25 = BM25Retriever(DOCS)
    assert "d_pay" not in [d.id for d in bm25.retrieve(q, k=1)]
    hy = HybridRetriever(DOCS, bm25, ConceptEncoder(), alpha=0.5)
    assert hy.retrieve(q, k=1)[0].id == "d_pay"


def test_alpha_one_matches_bm25_order_and_scores_are_bounded():
    bm25 = BM25Retriever(DOCS)
    hy = HybridRetriever(DOCS, bm25, ConceptEncoder(), alpha=1.0)
    q = "근로자 연차"
    assert [d.id for d in hy.retrieve(q, k=3)] == [d.id for d in bm25.retrieve(q, k=3)]
    hy2 = HybridRetriever(DOCS, bm25, ConceptEncoder(), alpha=0.3)
    got = hy2.retrieve(q, k=5)
    assert all(0.0 < d.retrieval_score <= 1.0 for d in got)
    assert [d.retrieval_score for d in got] == sorted((d.retrieval_score for d in got), reverse=True)


def test_hybrid_respects_domains_and_delegates_bm25_helpers():
    bm25 = BM25Retriever(DOCS)
    hy = HybridRetriever(DOCS, bm25, ConceptEncoder())
    assert {d.domain for d in hy.retrieve("붉은 행성 급여", k=5, domains=["wiki"])} == {"wiki"}
    assert hy.term_weights("근로자") == bm25.term_weights("근로자")
    assert hy.get("d_pay").id == "d_pay" and len(hy.items) == len(DOCS)
    with pytest.raises(ValueError):
        HybridRetriever(DOCS, bm25, ConceptEncoder(), alpha=1.5)


def test_passages_are_encoded_once_and_queries_each_time():
    enc = ConceptEncoder()
    hy = HybridRetriever(DOCS, BM25Retriever(DOCS), enc)
    hy.retrieve("급여", k=2)
    hy.retrieve("연차", k=2)
    assert enc.calls == ["passage", "query", "query"]


def test_pipeline_with_hybrid_config_shares_candidates_and_records_the_retriever(tmp_path):
    cfg = PipelineConfig(retriever="hybrid", dense_model="fake/concepts")
    set_encoder("fake/concepts", ConceptEncoder())
    assert isinstance(build_retriever(DOCS, cfg), HybridRetriever)
    assert cfg.to_dict()["retriever"] == "hybrid"
    assert "retriever" not in PipelineConfig().to_dict()  # 기본값은 예전 설정 해시를 그대로 둔다
    qs = [EvalQuestion("q1", "임금은 언제 주나요?", ["d_pay"], "law", "easy", "dev", rubric="급여")]
    res = run_experiment(DOCS, qs, tmp_path / "r", config=cfg, conditions=("vanilla_rag", "kore"))
    rows = [json.loads(line) for line in (tmp_path / "r" / "run_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({tuple(r["candidate_ids"]) for r in rows}) == 1  # 두 조건이 같은 후보를 본다
    assert "d_pay" in rows[0]["candidate_ids"] and res["results"]
    with pytest.raises(ValueError):
        PipelineConfig(retriever="dense").validate()


def test_compare_retrievers_reports_paired_differences_and_refuses_test():
    qs = [
        EvalQuestion("q1", "임금은 언제 주나요?", ["d_pay"], "law", "easy", "dev", rubric="x"),
        EvalQuestion("q2", "퇴직 알림은 며칠 전?", ["d_fire"], "law", "easy", "train", rubric="x"),
        EvalQuestion("q3", "휴가 며칠?", ["d_leave"], "law", "easy", "dev", rubric="x"),
        EvalQuestion("q4", "비밀", ["d_mars"], "wiki", "easy", "test", rubric="x"),
    ]
    set_encoder("fake/concepts", ConceptEncoder())
    base = PipelineConfig(dense_model="fake/concepts", candidate_k=2, k=1)
    res = compare_retrievers(DOCS, qs, ["bm25", "hybrid::0.5"], base=base)
    assert res["n_questions"] == 3  # test 질문은 빠진다
    bm, hy = res["rows"]
    assert hy["recall@1"] >= bm["recall@1"] and "diff_recall@3" in hy and "diff_recall@3" not in bm
    with pytest.raises(ValueError, match="test"):
        compare_retrievers(DOCS, qs, ["bm25"], splits=("dev", "test"))
    with pytest.raises(ValueError):
        parse_variant("bm25:nope", PipelineConfig())
    with pytest.raises(ValueError):
        parse_variant("bm25:ngram:0.5:extra", PipelineConfig())
    assert parse_variant("hybrid:word:0.2", PipelineConfig()).hybrid_alpha == 0.2


def test_cli_retrieval_eval_on_sample(tmp_path, capsys):
    out = tmp_path / "re.json"
    assert main(["retrieval-eval", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "검색기 비교" in text and "시도한 설정 2개" in text
    assert json.loads(out.read_text(encoding="utf-8"))["baseline"] == "bm25:ngram"
    with pytest.raises(SystemExit):
        main(["retrieval-eval", "--knowledge", SAMPLE_K, "--questions", SAMPLE_Q, "--splits", "test"])


def test_cli_search_accepts_config(capsys):
    assert main(["search", "Which planet is the largest?", "--knowledge", SAMPLE_K, "--config", "configs/default.json"]) == 0
    assert "선택" in capsys.readouterr().out
