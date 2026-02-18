from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

def base_label(tag: str) -> str:
    # "B-EMAIL" -> "EMAIL", "O" -> "O"
    if tag == "O":
        return "O"
    if "-" in tag:
        return tag.split("-", 1)[1]
    return tag

def bio_prefix(tag: str) -> str:
    if tag == "O":
        return "O"
    if "-" in tag:
        return tag.split("-", 1)[0]
    return "O"

@dataclass(frozen=True)
class EntitySpan:
    label: str
    start: int   # inclusive token index
    end: int     # exclusive token index

def tags_to_spans(tags: List[str]) -> List[EntitySpan]:
    spans: List[EntitySpan] = []
    i = 0
    n = len(tags)

    while i < n:
        t = tags[i]
        if t == "O":
            i += 1
            continue

        pref = bio_prefix(t)
        lab = base_label(t)

        # Start new span on B-*, or on I-* that isn't continuing a valid prior span
        start = i
        i += 1
        while i < n:
            t2 = tags[i]
            if t2 == "O":
                break
            if base_label(t2) != lab:
                break
            # Continue while I-* of same label; also tolerate B-* of same label as restart
            if bio_prefix(t2) == "I":
                i += 1
                continue
            if bio_prefix(t2) == "B":
                break
            i += 1

        spans.append(EntitySpan(label=lab, start=start, end=i))
    return spans

def labels_present(tags: List[str]) -> Set[str]:
    return {base_label(t) for t in tags if t != "O"}
