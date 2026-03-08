#!/usr/bin/env python3

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class Placement:
    edge_num: int
    likelihood: float
    distal_length: float
    pendant_length: float
    split_key: Optional[Tuple[str, ...]] = None


@dataclass
class QueryPlacements:
    top1: Placement
    all_rows: List[Placement]


@dataclass
class ParsedNode:
    name: Optional[str]
    children: List["ParsedNode"]
    edge_num: Optional[int] = None


class NewickParser:
    def __init__(self, text: str) -> None:
        self.text = text.strip()
        self.pos = 0

    def parse(self) -> ParsedNode:
        node = self._parse_subtree()
        self._skip_ws()
        if self.pos >= len(self.text) or self.text[self.pos] != ";":
            raise ValueError("Invalid jplace tree: missing terminating ';'")
        return node

    def _parse_subtree(self) -> ParsedNode:
        self._skip_ws()
        if self.pos < len(self.text) and self.text[self.pos] == "(":
            self.pos += 1
            children = [self._parse_subtree()]
            while True:
                self._skip_ws()
                if self.pos >= len(self.text):
                    raise ValueError("Invalid jplace tree: unexpected end inside internal node")
                ch = self.text[self.pos]
                if ch == ",":
                    self.pos += 1
                    children.append(self._parse_subtree())
                    continue
                if ch == ")":
                    self.pos += 1
                    break
                raise ValueError(f"Invalid jplace tree: unexpected character '{ch}'")
            node = ParsedNode(name=None, children=children)
        else:
            node = ParsedNode(name=self._parse_name(), children=[])
        self._parse_edge_annotation(node)
        return node

    def _parse_name(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ",():;":
            self.pos += 1
        name = self.text[start:self.pos].strip()
        if not name:
            raise ValueError("Invalid jplace tree: empty tip name")
        return name

    def _parse_edge_annotation(self, node: ParsedNode) -> None:
        self._skip_ws()
        if self.pos >= len(self.text) or self.text[self.pos] != ":":
            return
        self.pos += 1
        while self.pos < len(self.text) and self.text[self.pos] not in "{,);":
            self.pos += 1
        self._skip_ws()
        if self.pos < len(self.text) and self.text[self.pos] == "{":
            self.pos += 1
            start = self.pos
            while self.pos < len(self.text) and self.text[self.pos] != "}":
                self.pos += 1
            if self.pos >= len(self.text):
                raise ValueError("Invalid jplace tree: unterminated edge annotation")
            node.edge_num = int(self.text[start:self.pos])
            self.pos += 1

    def _skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1


def canonical_split(leaves: Set[str], universe: Set[str]) -> Tuple[str, ...]:
    complement = universe - leaves
    chosen = leaves
    if len(complement) < len(leaves):
        chosen = complement
    elif len(complement) == len(leaves):
        chosen = min(tuple(sorted(leaves)), tuple(sorted(complement)))
        return tuple(chosen)
    return tuple(sorted(chosen))


def build_edge_split_map(tree_text: str) -> Dict[int, Tuple[str, ...]]:
    root = NewickParser(tree_text).parse()
    all_leaves: Set[str] = set()

    def collect_all_leaves(node: ParsedNode) -> Set[str]:
        if not node.children:
            assert node.name is not None
            all_leaves.add(node.name)
            return {node.name}
        leaves: Set[str] = set()
        for child in node.children:
            leaves |= collect_all_leaves(child)
        return leaves

    collect_all_leaves(root)
    edge_to_split: Dict[int, Tuple[str, ...]] = {}

    def assign_splits(node: ParsedNode) -> Set[str]:
        if not node.children:
            assert node.name is not None
            leaves = {node.name}
        else:
            leaves = set()
            for child in node.children:
                leaves |= assign_splits(child)
        if node.edge_num is not None:
            edge_to_split[node.edge_num] = canonical_split(leaves, all_leaves)
        return leaves

    assign_splits(root)
    return edge_to_split


def load_jplace(path: Path) -> Dict[str, QueryPlacements]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    tree_text = data.get("tree")
    fields = data.get("fields")
    placements = data.get("placements")
    if not isinstance(tree_text, str) or not isinstance(fields, list) or not isinstance(placements, list):
        raise ValueError(f"{path}: missing 'tree', 'fields' or 'placements'")

    required = ["edge_num", "likelihood", "distal_length", "pendant_length"]
    field_index = {}
    for name in required:
        if name not in fields:
            raise ValueError(f"{path}: missing required field '{name}'")
        field_index[name] = fields.index(name)
    edge_splits = build_edge_split_map(tree_text)

    result: Dict[str, QueryPlacements] = {}
    for entry in placements:
        names = entry.get("n")
        rows = entry.get("p")
        if not isinstance(names, list) or not names:
            continue
        if not isinstance(rows, list) or not rows:
            continue

        parsed_rows: List[Placement] = []
        for row in rows:
            if not isinstance(row, list):
                continue
            edge_num = int(row[field_index["edge_num"]])
            parsed_rows.append(
                Placement(
                    edge_num=edge_num,
                    likelihood=float(row[field_index["likelihood"]]),
                    distal_length=float(row[field_index["distal_length"]]),
                    pendant_length=float(row[field_index["pendant_length"]]),
                    split_key=edge_splits.get(edge_num),
                )
            )
        if not parsed_rows:
            continue
        query_placements = QueryPlacements(top1=parsed_rows[0], all_rows=parsed_rows)

        for query_name in names:
            result[str(query_name)] = query_placements

    return result


def mean(values: List[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / float(len(values))


def find_rank_by_split(rows: List[Placement], split_key: Optional[Tuple[str, ...]]) -> Optional[int]:
    if split_key is None:
        return None
    for index, placement in enumerate(rows, start=1):
        if placement.split_key == split_key:
            return index
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare top-1 placements between two jplace files."
    )
    parser.add_argument("--truth", required=True, help="Reference jplace, e.g. epa-ng output")
    parser.add_argument("--pred", required=True, help="Predicted jplace, e.g. MLIPPER output")
    parser.add_argument("--truth-topk", type=int, default=1, help="Compare predicted top-1 against truth top-k placements")
    parser.add_argument("--max-mismatches", type=int, default=50, help="Maximum mismatch rows to print")
    parser.add_argument("--max-topk-hits", type=int, default=20, help="Maximum top-k-but-not-top1 examples to print")
    args = parser.parse_args()

    truth = load_jplace(Path(args.truth))
    pred = load_jplace(Path(args.pred))

    shared_queries = sorted(set(truth) & set(pred))
    truth_only = sorted(set(truth) - set(pred))
    pred_only = sorted(set(pred) - set(truth))

    if not shared_queries:
        raise SystemExit("No shared query names between the two jplace files.")

    exact_matches = 0
    exact_split_matches = 0
    topk_raw_matches = 0
    topk_split_matches = 0
    distal_abs_errors_all: List[float] = []
    pendant_abs_errors_all: List[float] = []
    distal_abs_errors_matched: List[float] = []
    pendant_abs_errors_matched: List[float] = []
    mismatches = []
    topk_not_top1 = []
    topk_rank_histogram: Dict[int, int] = {}

    for query_name in shared_queries:
        truth_q = truth[query_name]
        pred_q = pred[query_name]
        truth_p = truth_q.top1
        pred_p = pred_q.top1
        truth_topk = truth_q.all_rows[: max(1, args.truth_topk)]
        truth_topk_edges = {placement.edge_num for placement in truth_topk}
        truth_topk_splits = {
            placement.split_key for placement in truth_topk if placement.split_key is not None
        }

        distal_err = abs(pred_p.distal_length - truth_p.distal_length)
        pendant_err = abs(pred_p.pendant_length - truth_p.pendant_length)
        distal_abs_errors_all.append(distal_err)
        pendant_abs_errors_all.append(pendant_err)

        if pred_p.edge_num == truth_p.edge_num:
            exact_matches += 1
        if pred_p.split_key is not None and pred_p.split_key == truth_p.split_key:
            exact_split_matches += 1
            distal_abs_errors_matched.append(distal_err)
            pendant_abs_errors_matched.append(pendant_err)
        if pred_p.edge_num in truth_topk_edges:
            topk_raw_matches += 1
        if pred_p.split_key is not None and pred_p.split_key in truth_topk_splits:
            topk_split_matches += 1
            if not (pred_p.split_key is not None and pred_p.split_key == truth_p.split_key):
                pred_rank = find_rank_by_split(truth_q.all_rows, pred_p.split_key)
                truth_best_lk = truth_q.top1.likelihood
                pred_truth_lk = truth_q.all_rows[pred_rank - 1].likelihood if pred_rank is not None else math.nan
                topk_not_top1.append(
                    (
                        query_name,
                        pred_rank,
                        truth_p.edge_num,
                        pred_p.edge_num,
                        truth_best_lk,
                        pred_truth_lk,
                        truth_best_lk - pred_truth_lk if pred_rank is not None else math.nan,
                    )
                )
                if pred_rank is not None:
                    topk_rank_histogram[pred_rank] = topk_rank_histogram.get(pred_rank, 0) + 1
        if not (pred_p.split_key is not None and pred_p.split_key == truth_p.split_key):
            mismatches.append(
                (
                    query_name,
                    truth_p.edge_num,
                    pred_p.edge_num,
                    truth_p.likelihood,
                    pred_p.likelihood,
                )
            )

    print(f"truth file: {args.truth}")
    print(f"pred file:  {args.pred}")
    print(f"shared queries: {len(shared_queries)}")
    print(f"truth-only queries: {len(truth_only)}")
    print(f"pred-only queries: {len(pred_only)}")
    print()

    exact_rate = exact_matches / float(len(shared_queries))
    exact_split_rate = exact_split_matches / float(len(shared_queries))
    topk_raw_rate = topk_raw_matches / float(len(shared_queries))
    topk_split_rate = topk_split_matches / float(len(shared_queries))
    print(f"top-1 exact edge match (raw edge_num): {exact_matches}/{len(shared_queries)} ({exact_rate:.2%})")
    print(f"top-1 exact edge match (tree split):   {exact_split_matches}/{len(shared_queries)} ({exact_split_rate:.2%})")
    if args.truth_topk > 1:
        print(f"pred top-1 in truth top-{args.truth_topk} (raw edge_num): {topk_raw_matches}/{len(shared_queries)} ({topk_raw_rate:.2%})")
        print(f"pred top-1 in truth top-{args.truth_topk} (tree split):   {topk_split_matches}/{len(shared_queries)} ({topk_split_rate:.2%})")
        if topk_not_top1:
            avg_gap = mean([item[6] for item in topk_not_top1 if not math.isnan(item[6])])
            print(f"avg truth-top1 minus pred-edge truth-lk gap (top-{args.truth_topk} but not top-1): {avg_gap:.12f}")
            print("truth rank histogram for pred top-1 when inside truth top-k but not top-1:")
            for rank in sorted(topk_rank_histogram):
                print(f"  rank {rank}: {topk_rank_histogram[rank]}")
    print(f"mean abs distal length error (all shared):   {mean(distal_abs_errors_all):.12f}")
    print(f"mean abs pendant length error (all shared):  {mean(pendant_abs_errors_all):.12f}")
    if exact_split_matches > 0:
        print(f"mean abs distal length error (matched):      {mean(distal_abs_errors_matched):.12f}")
        print(f"mean abs pendant length error (matched):     {mean(pendant_abs_errors_matched):.12f}")
    else:
        print("mean abs distal length error (matched):      n/a")
        print("mean abs pendant length error (matched):     n/a")

    if mismatches:
        print()
        print("top-1 edge mismatches:")
        for query_name, truth_edge, pred_edge, truth_lk, pred_lk in mismatches[: args.max_mismatches]:
            print(
                f"  {query_name}: truth edge={truth_edge}, pred edge={pred_edge}, "
                f"truth ll={truth_lk:.12f}, pred ll={pred_lk:.12f}"
            )
        remaining = len(mismatches) - min(len(mismatches), args.max_mismatches)
        if remaining > 0:
            print(f"  ... {remaining} more mismatches omitted")

    if args.truth_topk > 1 and topk_not_top1:
        print()
        print(f"pred top-1 inside truth top-{args.truth_topk} but not truth top-1:")
        for query_name, pred_rank, truth_edge, pred_edge, truth_best_lk, pred_truth_lk, lk_gap in topk_not_top1[: args.max_topk_hits]:
            print(
                f"  {query_name}: truth top1 edge={truth_edge}, pred edge={pred_edge}, "
                f"truth rank of pred={pred_rank}, truth top1 ll={truth_best_lk:.12f}, "
                f"truth ll at pred edge={pred_truth_lk:.12f}, gap={lk_gap:.12f}"
            )
        remaining = len(topk_not_top1) - min(len(topk_not_top1), args.max_topk_hits)
        if remaining > 0:
            print(f"  ... {remaining} more top-k hits omitted")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
