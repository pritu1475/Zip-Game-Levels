#!/usr/bin/env python3
"""
Zip Puzzle Level Validator
Validates levels.json for a Zip-style puzzle where:
  1. Movement is orthogonal (up/down/left/right).
  2. Every cell must be visited exactly once.
  3. Checkpoints must be visited in numerical order.
  4. Walls block movement across the specified cell-to-cell edge.

Usage:
    python validate_levels.py levels.json

Exit code:
    0 = all levels valid
    1 = one or more invalid levels
"""

import json
import sys
from collections import defaultdict


def parity(r, c):
    return (r + c) & 1


def edge_key(a, b):
    return tuple(sorted((a, b)))


def parse_wall(wall):
    return (
        (wall["r1"], wall["c1"]),
        (wall["r2"], wall["c2"]),
    )


def validate_structure(level):
    errors = []

    required = ["id", "rows", "cols", "checkpoints", "walls"]
    for key in required:
        if key not in level:
            errors.append(f"missing required field '{key}'")

    if errors:
        return errors

    rows = level["rows"]
    cols = level["cols"]

    if not isinstance(rows, int) or not isinstance(cols, int):
        errors.append("rows and cols must be integers")
        return errors

    if rows <= 0 or cols <= 0:
        errors.append("rows and cols must be greater than zero")

    checkpoints = level["checkpoints"]

    if not checkpoints:
        errors.append("no checkpoints defined")
        return errors

    values = []
    positions = set()

    for cp in checkpoints:
        for key in ("row", "col", "value"):
            if key not in cp:
                errors.append(f"checkpoint missing '{key}'")
                continue

        if not all(k in cp for k in ("row", "col", "value")):
            continue

        r, c, value = cp["row"], cp["col"], cp["value"]

        if not (0 <= r < rows and 0 <= c < cols):
            errors.append(
                f"checkpoint {value} at ({r},{c}) is outside the board"
            )

        pos = (r, c)
        if pos in positions:
            errors.append(f"duplicate checkpoint position {pos}")
        positions.add(pos)

        values.append(value)

    if values:
        expected = list(range(1, len(values) + 1))
        if sorted(values) != expected:
            errors.append(
                f"checkpoint values must be exactly {expected}; found {sorted(values)}"
            )

    for wall in level["walls"]:
        try:
            a, b = parse_wall(wall)
        except KeyError as exc:
            errors.append(f"wall missing field {exc}")
            continue

        for r, c in (a, b):
            if not (0 <= r < rows and 0 <= c < cols):
                errors.append(f"wall endpoint ({r},{c}) is outside the board")

        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            errors.append(
                f"wall {a}->{b} does not separate adjacent cells"
            )

    return errors


def build_blocked_edges(level):
    blocked = set()

    for wall in level["walls"]:
        a, b = parse_wall(wall)
        blocked.add(edge_key(a, b))

    return blocked


def neighbors(cell, rows, cols, blocked):
    r, c = cell

    candidates = [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    ]

    for nxt in candidates:
        nr, nc = nxt
        if 0 <= nr < rows and 0 <= nc < cols:
            if edge_key(cell, nxt) not in blocked:
                yield nxt


def solve_level(level):
    """
    Backtracking Hamiltonian-path solver with checkpoint-order constraints.

    The path starts at checkpoint 1 and ends at the final checkpoint.
    Every board cell must be visited exactly once.
    Checkpoints must be encountered in increasing order.
    """

    rows = level["rows"]
    cols = level["cols"]
    total = rows * cols

    checkpoint_by_pos = {
        (cp["row"], cp["col"]): cp["value"]
        for cp in level["checkpoints"]
    }

    checkpoint_positions = {
        cp["value"]: (cp["row"], cp["col"])
        for cp in level["checkpoints"]
    }

    max_checkpoint = max(checkpoint_positions)
    start = checkpoint_positions[1]
    finish = checkpoint_positions[max_checkpoint]
    blocked = build_blocked_edges(level)

    # Fast parity check.
    if total % 2 == 0 and parity(*start) == parity(*finish):
        return None, (
            "impossible by checkerboard parity: "
            f"start {start} and final {finish} have the same parity "
            f"on an even-cell board"
        )

    # Build adjacency once.
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    adjacency = {
        cell: list(neighbors(cell, rows, cols, blocked))
        for cell in cells
    }

    path = [start]
    visited = {start}

    # Next checkpoint value required.
    next_checkpoint = 2

    # If start is also unexpectedly another checkpoint, structure validation
    # should already have caught duplicate positions.
    solution = None

    def checkpoint_allowed(cell, current_next):
        value = checkpoint_by_pos.get(cell)

        if value is None:
            return True, current_next

        if value == current_next:
            return True, current_next + 1

        # A checkpoint may not be entered before its turn, and may not be
        # revisited after its turn.
        return False, current_next

    def connectivity_possible(current):
        """
        Lightweight pruning:
        Every unvisited cell must remain reachable from the current endpoint.
        """
        remaining = set(cells) - visited
        if not remaining:
            return True

        # The next move must enter the remaining region from current.
        stack = [current]
        seen = {current}

        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if v in remaining and v not in seen:
                    seen.add(v)
                    stack.append(v)

        return remaining.issubset(seen)

    def forced_degree_prune(current):
        """
        Avoid obvious dead ends:
        An unvisited non-final cell cannot have zero available neighbors.
        A remaining cell with exactly one available neighbor is only acceptable
        when it can be the final endpoint or can be connected appropriately.
        """
        remaining = set(cells) - visited

        for cell in remaining:
            available = 0
            for nxt in adjacency[cell]:
                if nxt == current or nxt in remaining:
                    available += 1

            if available == 0:
                return False

        return True

    def dfs(current, current_next):
        nonlocal solution

        if len(path) == total:
            if current == finish and current_next == max_checkpoint + 1:
                solution = path.copy()
                return True
            return False

        # Final cell cannot be entered before all cells are visited.
        candidates = []

        for nxt in adjacency[current]:
            if nxt in visited:
                continue

            if nxt == finish and len(path) != total - 1:
                continue

            allowed, new_next = checkpoint_allowed(nxt, current_next)
            if not allowed:
                continue

            candidates.append((nxt, new_next))

        # Warn/prune if no legal continuation.
        if not candidates:
            return False

        # Warn/prune for disconnected remaining cells.
        if not connectivity_possible(current):
            return False

        if not forced_degree_prune(current):
            return False

        # Warnsdorff-style ordering: try cells with fewer onward choices first.
        def onward_score(item):
            cell, _ = item
            count = 0
            for n in adjacency[cell]:
                if n not in visited and n != finish:
                    count += 1
            return count

        candidates.sort(key=onward_score)

        for nxt, new_next in candidates:
            visited.add(nxt)
            path.append(nxt)

            if dfs(nxt, new_next):
                return True

            path.pop()
            visited.remove(nxt)

        return False

    if dfs(start, next_checkpoint):
        return solution, None

    return None, (
        "no complete Hamiltonian path satisfies the walls, "
        "checkpoint order, and full-board requirement"
    )


def validate_level(level):
    errors = validate_structure(level)

    if errors:
        return False, errors, None

    solution, solver_error = solve_level(level)

    if solver_error:
        return False, [solver_error], None

    return True, [], solution


def format_path(path):
    if not path:
        return ""
    return " -> ".join(f"({r},{c})" for r, c in path)


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "levels.json"

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: file not found: {filename}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON: {exc}")
        sys.exit(1)

    levels = data.get("levels")

    if not isinstance(levels, list):
        print("ERROR: JSON must contain a 'levels' array")
        sys.exit(1)

    print("=" * 72)
    print("ZIP PUZZLE LEVEL VALIDATION")
    print("=" * 72)

    invalid_count = 0

    for level in levels:
        level_id = level.get("id", "?")
        title = level.get("title", "Untitled")
        rows = level.get("rows", "?")
        cols = level.get("cols", "?")

        print(f"\nLevel {level_id}: {title} ({rows}x{cols})")

        valid, errors, solution = validate_level(level)

        if valid:
            print("  STATUS: PASS")
            print(f"  Cells: {rows * cols}")
            print(f"  Solution length: {len(solution)}")
            print(f"  Checkpoints: {len(level['checkpoints'])}")
        else:
            invalid_count += 1
            print("  STATUS: FAIL")
            for error in errors:
                print(f"  - {error}")

    print("\n" + "=" * 72)
    print(
        f"RESULT: {len(levels) - invalid_count}/{len(levels)} levels valid"
    )

    if invalid_count:
        print(f"INVALID LEVELS: {invalid_count}")
        print("=" * 72)
        sys.exit(1)

    print("ALL LEVELS PASSED")
    print("=" * 72)
    sys.exit(0)


if __name__ == "__main__":
    main()
