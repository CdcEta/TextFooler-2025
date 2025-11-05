import os

data_dir = "E:/TAAD/TextFooler-master/traindata/ag"

command = f'python run_classifier.py --data_dir "{data_dir}" ' \
          '--bert_model bert-base-uncased ' \
          '--task_name ag ' \
          f'--output_dir results/ag ' \
          '--cache_dir pytorch_cache --do_train --do_eval --do_lower_case '

print(">>> Running command:\n", command)
os.system(command)

