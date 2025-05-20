import json
import re
import argparse
from pathlib import Path

def validate_line(line, line_num, id_pattern, seen_ids):
    errors = []

    try:
        entry = json.loads(line)
    except json.JSONDecodeError as e:
        return [f"Line {line_num}: JSON decode error: {e}"]

    # Validate top-level keys
    expected_keys = {"id", "dialogue", "response_preferred", "response_non_preferred"}
    entry_keys = set(entry.keys())
    missing = expected_keys - entry_keys
    extra = entry_keys - expected_keys
    if missing:
        errors.append(f"Line {line_num}: Missing keys: {missing}")
    if extra:
        errors.append(f"Line {line_num}: Unexpected keys: {extra}")

    # Validate id
    id_val = entry.get("id")
    if not isinstance(id_val, str) or not re.match(id_pattern, id_val):
        errors.append(f"Line {line_num}: 'id' must match pattern {id_pattern}, got: {id_val}")
    elif id_val in seen_ids:
        errors.append(f"Line {line_num}: Duplicate id: {id_val}")
    else:
        seen_ids.add(id_val)

    # Validate dialogue
    dialogue = entry.get("dialogue")
    if not isinstance(dialogue, list) or len(dialogue) < 3:
        errors.append(f"Line {line_num}: 'dialogue' must be a list with at least 3 turns")
    if isinstance(dialogue, list) and len(dialogue) >= 3 and len(dialogue) % 2 == 0:
        errors.append(f"Line {line_num}: 'dialogue' should have an odd number of turns (user start and end)")
    else:
        # First and last speaker must be user
        if dialogue[0].get("speaker") != "user":
            errors.append(f"Line {line_num}: dialogue[0] speaker must be 'user'")
        if dialogue[-1].get("speaker") != "user":
            errors.append(f"Line {line_num}: last dialogue turn must be 'user'")
        # Check alternating speakers and valid fields
        last_speaker = None
        for idx, turn in enumerate(dialogue):
            if not isinstance(turn, dict):
                errors.append(f"Line {line_num}, dialogue[{idx}]: Turn must be a dict")
                continue
            # Validate dialogue turn keys
            expected_turn_keys = {"speaker", "text"}
            turn_keys = set(turn.keys())
            missing_turn_keys = expected_turn_keys - turn_keys
            extra_turn_keys = turn_keys - expected_turn_keys
            if missing_turn_keys:
                errors.append(f"Line {line_num}, dialogue[{idx}]: Missing keys in turn: {missing_turn_keys}")
            if extra_turn_keys:
                errors.append(f"Line {line_num}, dialogue[{idx}]: Unexpected keys in turn: {extra_turn_keys}")
            speaker = turn.get("speaker")
            text = turn.get("text")
            if speaker not in {"user", "assistant"}:
                errors.append(f"Line {line_num}, dialogue[{idx}]: 'speaker' must be 'user' or 'assistant', got: {speaker}")
            if not isinstance(text, str) or not text.strip():
                errors.append(f"Line {line_num}, dialogue[{idx}]: 'text' must be a non-empty string")
            if last_speaker is not None and speaker == last_speaker:
                errors.append(f"Line {line_num}, dialogue[{idx}]: speaker did not alternate (two '{speaker}' in a row)")
            last_speaker = speaker

    # Validate responses
    for resp_key in ["response_preferred", "response_non_preferred"]:
        resp_val = entry.get(resp_key)
        if not isinstance(resp_val, str) or not resp_val.strip():
            errors.append(f"Line {line_num}: '{resp_key}' must be a non-empty string")

    return errors

def main(file_path):
    id_pattern = r"^nrg_\d{4}$"
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: File not found: {file_path}")
        return

    errors_found = False
    seen_ids = set()
    with path.open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            errors = validate_line(line, line_num, id_pattern, seen_ids)
            if errors:
                errors_found = True
                for error in errors:
                    print(error)

    if not errors_found:
        print("All lines are valid!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate JSONL dataset for nuclear energy dialogues")
    parser.add_argument("file", help="Path to the JSONL file to validate")
    args = parser.parse_args()
    main(args.file)