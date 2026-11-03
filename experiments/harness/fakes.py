"""모델 없이 배관만 시험하는 가짜 생성기.

ExtractiveGenerator(근거에서 문장을 뽑는 결정적 기준선)를 그대로 쓴다. 언어모델이 아니고, 점수는 연구 결과가 아니다.
프롬프트 틀은 이 생성기의 답을 바꾸지 못한다(이름에만 반영된다). 그래서 prompt_template 편집의 효과는 여기서 잴 수 없다.
"""

from __future__ import annotations

import hashlib

from harness.components import HarnessConfig
from kore.generation import ExtractiveGenerator


class TemplateExtractiveGenerator(ExtractiveGenerator):
    def __init__(self, template: str | None = None) -> None:
        self.template = template
        tag = "" if not template else "-t" + hashlib.sha256(template.encode("utf-8")).hexdigest()[:6]
        self.name = "fake-extractive-v0" + tag


def make_fake_generator(cfg: HarnessConfig) -> TemplateExtractiveGenerator:
    return TemplateExtractiveGenerator(cfg.prompt_template)
