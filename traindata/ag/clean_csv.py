import os
import shutil

def clean_csv(input_path):
    # Prepare paths
    bak_path = input_path + ".bak"
    output_path = input_path

    # Backup original file
    if not os.path.exists(bak_path):
        shutil.copyfile(input_path, bak_path)
        print(f"馃敀 Backup created: {bak_path}")
    else:
        print(f"鈿狅笍 Backup already exists: {bak_path}")

    cleaned_lines = []
    with open(bak_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Remove surrounding quotes only if they wrap the entire field
            if line.startswith('"') and line.rfind('",') != -1:
                idx = line.rfind('",')
                text = line[1:idx]  # Remove starting "
                label = line[idx+2:].strip()
            else:
                # Split by last comma
                if ',' not in line:
                    continue  # Skip malformed lines
                text, label = line.rsplit(',', 1)

            # Final cleanup: strip spaces
            text = text.strip()
            label = label.strip()

            cleaned_lines.append(f"{text},{label}")

    # Write cleaned file (overwrite original path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')

    print(f"鉁?Cleaned data written to: {output_path}")
    print(f"馃Ч Total cleaned lines: {len(cleaned_lines)}\n")


# Paths to clean
train_csv = r"E:\TAAD\\TextFooler-2025\traindata\ag\train_tok.csv"
test_csv = r"E:\TAAD\\TextFooler-2025\traindata\ag\test_tok.csv"

# Clean both datasets
clean_csv(train_csv)
clean_csv(test_csv)

