#!/usr/bin/env python3
"""Explain invalid CornPuzzle curriculum prefixes and systematic SGF failures."""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


def normalize(cells):
    cells = list(cells)
    if not cells:
        return tuple()
    mr = min(r for r, _ in cells)
    mc = min(c for _, c in cells)
    return tuple(sorted((r - mr, c - mc) for r, c in cells))


def rotate90(cells):
    max_r = max(r for r, _ in cells)
    return normalize((c, max_r - r) for r, c in cells)


def rotate180(cells):
    return rotate90(rotate90(normalize(cells)))


def key(cells):
    return ";".join(f"{r},{c}" for r, c in normalize(cells)) + ";"


def canonical_key(cells):
    cells = list(cells)
    return min(key(cells), key(rotate180(cells)))


def parse_puzzle(path: Path):
    rows, cols = 7, 14
    pieces, current = [], []
    for raw in path.read_text().splitlines() + [""]:
        line = raw.rstrip("\r")
        if line.startswith("#"):
            fields = line[1:].split()
            if len(fields) >= 2 and fields[0] in ("rows", "active_rows"):
                rows = int(fields[1])
            elif len(fields) >= 2 and fields[0] in ("cols", "active_cols"):
                cols = int(fields[1])
            continue
        if not line:
            if current:
                cells = [(r, c) for r, row in enumerate(current)
                         for c, ch in enumerate(row) if ch in "1#"]
                if cells:
                    pieces.append(normalize(cells))
                current = []
        else:
            current.append(line)
    return rows, cols, pieces


def parse_solution(path: Path):
    values = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            values.extend(int(x) for x in line.split())
    return values


def diagnose_prefix(task_id, puzzle_path, solution_path, prefix):
    puzzle, solution = Path(puzzle_path), Path(solution_path)
    if not puzzle.exists():
        return f"puzzle file 不存在: {puzzle}"
    if not solution.exists():
        return f"solution file 不存在: {solution}"
    rows, cols, pieces = parse_puzzle(puzzle)
    labels = parse_solution(solution)
    if len(labels) < rows * cols:
        return f"solution 格數不足: {len(labels)} < {rows * cols}"
    remaining = Counter(canonical_key(p) for p in pieces)
    occupied = 0
    for label in range(1, min(prefix, len(pieces)) + 1):
        cells = [(r, c) for r in range(rows) for c in range(cols)
                 if labels[r * cols + c] == label]
        if not cells:
            return f"solution 缺少 label {label}"
        matched = None
        tried = set()
        for shift in range(cols):
            candidate = canonical_key((r, (c + shift) % cols) for r, c in cells)
            tried.add(candidate)
            if remaining[candidate] > 0:
                matched = candidate
                break
        if matched is None:
            available_same_area = sorted(
                (k, count) for k, count in remaining.items()
                if count > 0 and k.count(";") == len(cells)
            )
            return (f"label {label} 的 {len(cells)} 格形狀找不到可用拼圖片；"
                    f"同面積剩餘形狀數={len(available_same_area)}，"
                    f"solution cells={cells}")
        remaining[matched] -= 1
        occupied += len(cells)
    remaining_area = sum(k.count(";") * count for k, count in remaining.items())
    empty_area = rows * cols - occupied
    if remaining_area != empty_area:
        return f"prefix 後面積不一致: 空格={empty_area}, 剩餘拼圖片面積={remaining_area}"
    return "OK"


def games_from_sgf(path: Path):
    text = path.read_text(errors="replace")
    return re.findall(r"\(;GM\[cornpuzzle\].*?(?=\n\(;GM\[cornpuzzle\]|\Z)", text, re.S)


def action_name(raw):
    try:
        action = int(raw)
    except ValueError:
        return raw or "empty"
    if action == 128:
        return "null"
    return f"P{action // 2 + 1}R{180 if action % 2 else 0}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--sgf")
    p.add_argument("--only-problems", action="store_true")
    args = p.parse_args()

    manifest = {}
    for raw in Path(args.manifest).read_text().splitlines():
        if not raw or raw.startswith("#"):
            continue
        task, puzzle, solution, prefix = raw.split("\t")[:4]
        manifest[task] = (puzzle, solution, int(prefix))

    by_task = defaultdict(list)
    if args.sgf:
        for game in games_from_sgf(Path(args.sgf)):
            task = re.search(r"CTASK\[([^]]+)\]", game)
            result = re.search(r"RE\[([-+0-9.eE]+)\]", game)
            if not task or not result:
                continue
            actions = [action_name(x) for x in re.findall(r";B\[([^]]*)\]", game)]
            by_task[task.group(1)].append((float(result.group(1)), actions))

    print("task\tprefix_check\tgames\tsolved\tmean_return\tcommon_action_sequence")
    for task, (puzzle, solution, prefix) in manifest.items():
        reason = diagnose_prefix(task, puzzle, solution, prefix)
        games = by_task.get(task, [])
        solved = sum(ret >= 1.0 - 1e-8 for ret, _ in games)
        mean = sum(ret for ret, _ in games) / len(games) if games else 0.0
        sequences = Counter(" ".join(actions) for _, actions in games)
        common = sequences.most_common(1)[0][0] if sequences else "-"
        is_problem = reason != "OK" or (games and solved < len(games)) or (args.sgf and not games)
        if not args.only_problems or is_problem:
            print(f"{task}\t{reason}\t{len(games)}\t{solved}\t{mean:.6f}\t{common}")


if __name__ == "__main__":
    main()
