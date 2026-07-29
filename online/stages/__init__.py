"""The six explicit stages of the online BetterCallAgent pipeline."""

from online.stages.stage_01_understand import run as understand
from online.stages.stage_02_generate_queries import run as generate_queries
from online.stages.stage_03_retrieve import run as retrieve
from online.stages.stage_04_rerank import run as rerank
from online.stages.stage_05_control_citations import run as control_citations
from online.stages.stage_06_answer import run as answer

__all__ = [
    "answer",
    "control_citations",
    "generate_queries",
    "rerank",
    "retrieve",
    "understand",
]
