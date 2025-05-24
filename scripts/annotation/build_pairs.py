import csv
import json

IN_CSV  = "data/annotation_tasks_labeled.csv"
OUT_JSON = "data/preference_pairs.jsonl"

with open(IN_CSV, encoding="utf-8") as fin, open(OUT_JSON, "w", encoding="utf-8") as fout:
    reader = csv.DictReader(fin)
    for row in reader:
        better = row["better"].strip().upper()
        resp_A = row["reply_A"]
        resp_B = row["reply_B"]
        winner = resp_A if better == "A" else resp_B
        loser  = resp_B if better == "A" else resp_A

        pair = {
            "id": row["pair_id"],
            "dialogue": row["dialogue"].split("\n"),
            "response_preferred": winner,
            "response_non_preferred": loser
        }
        fout.write(json.dumps(pair, ensure_ascii=False) + "\n")

print("✅ preference_pairs.jsonl written")