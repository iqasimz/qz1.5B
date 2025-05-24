import json
import csv

INPUT = "data/cand_sft.jsonl"
OUT_CSV = "data/annotation_tasks.csv"

with open(INPUT) as fin, open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fout:
    writer = csv.writer(fout, quoting=csv.QUOTE_ALL)
    # Header: id, pair_id, dialogue, reply_A, reply_B
    writer.writerow(["id", "pair_id", "dialogue", "reply_A", "reply_B"])
    for ex in map(json.loads, fin):
        ctx = "\n".join(
            f"{'User' if t['speaker']=='user' else 'Assistant'}: {t['text']}"
            for t in ex["dialogue"]
        )
        base_id = ex["id"]

        # 1) Gold vs. Model
        writer.writerow([
            base_id,
            f"{base_id}_gold_vs_model",
            ctx,
            ex["response_preferred"],
            ex["response_model"]
        ])

        # 2) Model vs. Neutral
        writer.writerow([
            base_id,
            f"{base_id}_model_vs_neutral",
            ctx,
            ex["response_model"],
            ex["response_non_preferred"]
        ])

print("➡️  annotation_tasks.csv written with", sum(1 for _ in open(OUT_CSV)) - 1, "tasks")