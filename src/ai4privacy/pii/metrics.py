from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
from collections import Counter

from .taxonomy import tags_to_spans, EntitySpan

@dataclass
class SpanPRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

def _safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0

def span_level_prf(true_tags: List[str], pred_tags: List[str]) -> SpanPRF:
    true_spans = set(tags_to_spans(true_tags))
    pred_spans = set(tags_to_spans(pred_tags))

    tp = len(true_spans & pred_spans)
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(int(round(2 * p * r * 1e9)), int(round((p + r) * 1e9))) if (p + r) else 0.0  # stable
    return SpanPRF(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn)

def span_level_prf_by_label(true_tags: List[str], pred_tags: List[str]) -> Dict[str, SpanPRF]:
    from .taxonomy import tags_to_spans

    true_spans = tags_to_spans(true_tags)
    pred_spans = tags_to_spans(pred_tags)

    labels = sorted({s.label for s in true_spans} | {s.label for s in pred_spans})
    out: Dict[str, SpanPRF] = {}

    for lab in labels:
        tset = set([s for s in true_spans if s.label == lab])
        pset = set([s for s in pred_spans if s.label == lab])

        tp = len(tset & pset)
        fp = len(pset - tset)
        fn = len(tset - pset)

        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        out[lab] = SpanPRF(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn)

    return out
