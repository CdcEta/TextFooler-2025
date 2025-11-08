import os

data_dir = "E:/TAAD/TextFooler-master/data/mnli"

command = (
    f'python run_classifier.py --data_dir "{data_dir}" '
    '--bert_model bert-base-uncased '
    '--task_name mnli '
    '--output_dir results/mnli '
    '--cache_dir pytorch_cache --do_eval --do_lower_case '
)

print(">>> Running command:\n", command)
os.system(command)