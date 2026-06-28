"""Allocate AMPN activation bits from a sensitivity CSV with dynamic programming."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from ampn_utils import ADALN_OP, AMPN_OPS, OP_COSTS, save_json


def load_sensitivity(path, ops):
    table = defaultdict(dict)
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            block_idx = int(row["block_idx"])
            op_name = row["op_name"]
            if op_name not in ops:
                continue
            bit = int(row["activation_bits"])
            loss = float(row["loss_diff"])
            key = (block_idx, op_name)
            if bit not in table[key] or loss < table[key][bit]:
                table[key][bit] = loss
    return table


def solve_dp(table, total_blocks, ops, bits, target_avg_bits, beta):
    items = [(block_idx, op) for block_idx in range(total_blocks) for op in ops]
    total_cost = sum(OP_COSTS[op] for _, op in items)
    target = int(round(target_avg_bits * total_cost * beta))
    max_budget = int(round(max(bits) * total_cost * beta))

    dp = [defaultdict(lambda: float("inf")) for _ in range(len(items) + 1)]
    choice = [{} for _ in range(len(items) + 1)]
    dp[0][0] = 0.0

    for i, item in enumerate(items, start=1):
        block_idx, op = item
        candidates = table.get(item)
        if not candidates:
            raise ValueError(f"No sensitivity rows for block={block_idx}, op={op}")
        for prev_budget, prev_loss in dp[i - 1].items():
            for bit in bits:
                if bit not in candidates:
                    continue
                cost = int(round(bit * OP_COSTS[op] * beta))
                new_budget = prev_budget + cost
                if new_budget > max_budget:
                    continue
                new_loss = prev_loss + candidates[bit]
                if new_loss < dp[i][new_budget]:
                    dp[i][new_budget] = new_loss
                    choice[i][new_budget] = (prev_budget, bit)

    feasible = [(abs(budget - target), loss, budget) for budget, loss in dp[len(items)].items()]
    if not feasible:
        raise RuntimeError("No feasible AMPN allocation found.")
    _, total_loss, best_budget = min(feasible)

    by_block = defaultdict(dict)
    budget = best_budget
    total_actual = 0.0
    for i in range(len(items), 0, -1):
        block_idx, op = items[i - 1]
        prev_budget, bit = choice[i][budget]
        loss = table[(block_idx, op)][bit]
        by_block[str(block_idx)][op] = {"bitwidth": bit, "loss": loss}
        total_actual += bit * OP_COSTS[op]
        budget = prev_budget

    actual_avg = total_actual / total_cost
    return {
        "by_block": dict(by_block),
        "total_loss": total_loss,
        "target_avg_bitwidth": target_avg_bits,
        "avg_weighted_bitwidth": actual_avg,
        "op_weights": {op: OP_COSTS[op] for op in ops},
    }


def add_fixed_ops(solution, total_blocks, fixed_adaln_bits):
    if fixed_adaln_bits is None:
        return solution
    for block_idx in range(total_blocks):
        solution["by_block"].setdefault(str(block_idx), {})
        solution["by_block"][str(block_idx)][ADALN_OP] = {"bitwidth": fixed_adaln_bits, "loss": 0.0, "fixed": True}
    return solution


def main(args):
    bits = [int(item) for item in args.bits.split(",") if item]
    ops = [op.strip() for op in args.ops.split(",") if op.strip()]
    table = load_sensitivity(args.sensitivity_csv, ops)
    solution = solve_dp(
        table,
        total_blocks=args.total_blocks,
        ops=ops,
        bits=bits,
        target_avg_bits=args.target_avg_bits,
        beta=args.beta,
    )
    solution = add_fixed_ops(solution, args.total_blocks, args.fixed_adaln_bits)
    save_json(args.output, solution)
    print(json.dumps({
        "output": args.output,
        "target_avg_bitwidth": solution["target_avg_bitwidth"],
        "avg_weighted_bitwidth": solution["avg_weighted_bitwidth"],
        "total_loss": solution["total_loss"],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensitivity-csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--target-avg-bits", type=float, required=True)
    parser.add_argument("--bits", type=str, default="1,2,3,4")
    parser.add_argument("--ops", type=str, default=",".join(AMPN_OPS))
    parser.add_argument("--total-blocks", type=int, default=28)
    parser.add_argument("--beta", type=int, default=1000)
    parser.add_argument("--fixed-adaln-bits", type=int, default=4)
    main(parser.parse_args())
