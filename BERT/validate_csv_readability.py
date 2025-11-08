# validate_csv_readability.py
import os
import csv

BASE = r"E:\TAAD\TextFooler-master\traindata\ag"
FILES = ["train_tok.csv", "test_tok.csv"]
VALID_LABELS = {"1", "2", "3", "4"}  # Adjust based on your dataset

def validate_csv(path):
    print(f"\n=== Validating file: {path} ===")
    total = 0
    valid = 0
    invalid = 0
    
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for ln, line in enumerate(f, 1):
            total += 1
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            # Try splitting at last comma as text,label
            try:
                text, label = line.rsplit(",", 1)
                text, label = text.strip(), label.strip()
            except ValueError:
                print(f"[Line {ln}] ❌ Cannot split by last comma: {line}")
                invalid += 1
                continue
            
            # Check label
            if label not in VALID_LABELS:
                print(f"[Line {ln}] ❌ Invalid label '{label}': {line}")
                invalid += 1
                continue
            
            valid += 1
            
            # Print a few sample lines
            if valid <= 5:
                print(f"[Line {ln}] ✅ Text: {text[:50]}... Label: {label}")

    print(f"\n🔍 Summary: {os.path.basename(path)}")
    print(f"  Total lines:       {total}")
    print(f"  Valid rows:        {valid}")
    print(f"  Invalid rows:      {invalid}")
    print(f"  Skipped/Empty rows: {total - valid - invalid}")
    print("-" * 50)


if __name__ == "__main__":
    for fn in FILES:
        validate_csv(os.path.join(BASE, fn))
