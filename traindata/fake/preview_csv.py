import os

base_dir = r"E:\TAAD\\TextFooler-2025\traindata\ag"
files = ["train_tok.csv", "test_tok.csv"]

for fname in files:
    file_path = os.path.join(base_dir, fname)
    print(f"\n=== {fname} ===")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i in range(10):
                line = f.readline()
                if not line:
                    break
                print(f"{i+1}: {line.strip()}")
    except Exception as e:
        print(f"Error reading {fname}: {e}")

