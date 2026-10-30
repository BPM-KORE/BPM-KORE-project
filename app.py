"""Gradio 데모. `pip install -e .[demo]` 후 `python app.py`.

KORE_KNOWLEDGE 환경변수로 지식 파일을 바꾼다(기본: data/sample_knowledge.csv).
데모는 검색·선택 결과를 보여 줄 뿐이고, 성능 향상을 증명하지 않는다.
"""

from __future__ import annotations

import os
from pathlib import Path

from kore.io import load_knowledge
from kore.retrieval import BM25Retriever
from kore.selector import select_knowledge

DEFAULT_KNOWLEDGE = Path(__file__).parent / "data" / "sample_knowledge.csv"
HEADERS = ["id", "분야", "제목", "선택 점수", "검색 점수", "본문"]


def build_retriever(path: str | os.PathLike[str] | None = None) -> BM25Retriever:
    return BM25Retriever(load_knowledge(path or os.environ.get("KORE_KNOWLEDGE", DEFAULT_KNOWLEDGE)))


def demo_search(
    retriever: BM25Retriever, question: str, k: int = 3, strategy: str = "kore", candidate_k: int = 10
) -> list[list[object]]:
    """질문 하나에 대한 선택 결과를 표 행으로 돌려준다."""
    if not question.strip():
        return []
    cands = retriever.retrieve(question, k=max(int(candidate_k), int(k)))
    picked = select_knowledge(
        question, cands, k=int(k), strategy=strategy, term_weights=retriever.term_weights(question)
    )
    return [
        [p.id, p.domain, p.title, round(p.selection_score, 3), round(p.retrieval_score, 3), p.text]
        for p in picked
    ]


def build_ui():  # pragma: no cover - gradio가 있어야 실행된다
    import gradio as gr

    retriever = build_retriever()

    def run(question: str, k: int, strategy: str):
        return demo_search(retriever, question, k=k, strategy=strategy)

    with gr.Blocks(title="KORE 데모") as ui:
        gr.Markdown("# KORE 데모\n연구 프로토타입이다. 성능 향상은 아직 검증되지 않았다.")
        question = gr.Textbox(label="질문")
        with gr.Row():
            k = gr.Slider(1, 10, value=3, step=1, label="선택 개수 k")
            strategy = gr.Radio(["kore", "similarity"], value="kore", label="선택 방식")
        table = gr.Dataframe(headers=HEADERS, wrap=True)
        gr.Button("검색").click(run, [question, k, strategy], table)
    return ui


if __name__ == "__main__":  # pragma: no cover
    build_ui().launch()
