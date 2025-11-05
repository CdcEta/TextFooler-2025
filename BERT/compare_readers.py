import os
from run_classifier import AGProcessor  # 使用你当前的run_classifier.py
import io

# 新的、更宽容的版本B: 允许内容中有逗号、去掉多余空白等
def read_corpus_v2(file_path):
    samples = []
    with io.open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # 방식: 尝试分离最后一次逗号后的标签
            try:
                text, label = line.rsplit(',', 1)
                text = text.strip().strip('"')
                label = label.strip()
                samples.append((text, label))
            except ValueError:
                print(f"[V2][Line {idx}] ❌ Could not split validly: {line}")
                continue
    return samples

def preview_samples(samples, n=5):
    for idx, (text, label) in enumerate(samples[:n]):
        print(f"  [{idx+1}] ✅ Text: {text[:50]}... Label: {label}")
    print(f"  Total loaded: {len(samples)}")
    print('-' * 40)

if __name__ == "__main__":
    data_dir = os.path.join("E:/TAAD/TextFooler-master/traindata/ag")
    train_path = os.path.join(data_dir, "train_tok.csv")
    test_path = os.path.join(data_dir, "test_tok.csv")

    # Version A: Original run_classifier.py approach
    print("\n=== 🅰️ Version A: run_classifier AGProcessor._read_corpus ===")
    processor = AGProcessor()
    train_samples_A = processor._read_corpus(train_path)
    preview_samples(train_samples_A)

    # Version B: Improved approach
    print("\n=== 🅱️ Version B: custom read_corpus_v2 ===")
    train_samples_B = read_corpus_v2(train_path)
    preview_samples(train_samples_B)
