"""
app/core/consensus.py
---------------------
Exact consensus panel rule implementation per Plan v2 Section 4.5.
Contains zero Streamlit imports.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from app.config import METHOD_NAMES
from app.core.methods import MethodResult

@dataclass
class ConsensusResult:
    panel: List[str]                      # Selected consensus Entrez IDs
    symbols: List[str]                    # Selected consensus gene symbols
    rule: str                             # "single_method" | "intersection_all_selected" | "most_occurring" | "fallback_top1"
    description: str                      # Plain-English explanation for the UI
    vote_count: Optional[int] = None      # Vote count (if applicable)
    source_method: Optional[str] = None   # Source method name (for single_method or fallback_top1)
    all_votes: Dict[str, int] = field(default_factory=dict) # Full vote dictionary across all candidates

def build_consensus(
    results: Dict[str, MethodResult],
    id_to_symbol: Dict[str, str],
    selected_methods_user: Optional[List[str]] = None
) -> ConsensusResult:
    """
    Builds the consensus panel following the exact 4-branch rule in Section 4.5:
    1. Single method selected: panel = top 10 of that method.
    2. Strict intersection: all selected methods agree on >= 1 gene.
    3. Most occurring: maximum vote tier among genes with votes > 1.
    4. Total disjoint fallback: top gene from the first selected method in fixed METHOD_NAMES order.
    """
    if not results:
        raise ValueError("Cannot build consensus from empty results dictionary.")

    # Determine fixed deterministic ordering of selected methods
    if selected_methods_user is not None:
        ordered_selected = [m for m in METHOD_NAMES if m in selected_methods_user and m in results]
    else:
        ordered_selected = [m for m in METHOD_NAMES if m in results]

    # Branch 1: Only one method selected
    if len(results) == 1:
        first_method = ordered_selected[0]
        r = results[first_method]
        panel_ids = r.top10_ids
        panel_symbols = [id_to_symbol.get(str(g), str(g)) for g in panel_ids]
        return ConsensusResult(
            panel=panel_ids,
            symbols=panel_symbols,
            rule="single_method",
            description=f"Single method selected ({first_method}); using top {len(panel_ids)} features directly without consensus filtering.",
            vote_count=1,
            source_method=first_method,
            all_votes={g: 1 for g in panel_ids}
        )

    # Count votes across all selected methods
    vote_counts = Counter()
    for r in results.values():
        vote_counts.update(r.top10_ids)

    n_selected = len(results)
    all_votes_dict = dict(vote_counts)

    # Branch 2: Strict intersection across ALL selected methods
    intersection = [g for g, v in vote_counts.items() if v == n_selected]
    if intersection:
        panel_symbols = [id_to_symbol.get(str(g), str(g)) for g in intersection]
        return ConsensusResult(
            panel=intersection,
            symbols=panel_symbols,
            rule="intersection_all_selected",
            description=f"All {n_selected} selected methods agreed on {len(intersection)} gene(s).",
            vote_count=n_selected,
            all_votes=all_votes_dict
        )

    # Branch 3: Most occurring tier (votes > 1)
    max_votes = max(vote_counts.values()) if vote_counts else 0
    if max_votes > 1:
        top_tier = [g for g, v in vote_counts.items() if v == max_votes]
        panel_symbols = [id_to_symbol.get(str(g), str(g)) for g in top_tier]
        return ConsensusResult(
            panel=top_tier,
            symbols=panel_symbols,
            rule="most_occurring",
            description=f"No gene was selected by all {n_selected} methods. Showing the {len(top_tier)} gene(s) chosen by the most methods ({max_votes} of {n_selected}).",
            vote_count=max_votes,
            all_votes=all_votes_dict
        )

    # Branch 4: Total disjoint fallback
    first_method = ordered_selected[0]
    fallback_gene = results[first_method].top10_ids[0]
    fallback_symbol = id_to_symbol.get(str(fallback_gene), str(fallback_gene))
    return ConsensusResult(
        panel=[fallback_gene],
        symbols=[fallback_symbol],
        rule="fallback_top1",
        description=f"No overlap between the selected methods' top 10 lists. Showing the #1 top-ranked gene from {first_method} as a deterministic fallback.",
        vote_count=1,
        source_method=first_method,
        all_votes=all_votes_dict
    )
