# SNLI（斯坦福自然语言推理）任务的 BERT 微调脚本（调用 run_classifier.py）。
# 注意：将 --data_dir 设置为 SNLI 数据路径；可按需开启断点续训（--do_resume）。
import os

# 组装命令并执行；训练与评估结果输出到 results/SNLI_retrain。
command = 'python run_classifier.py --data_dir /data/medg/misc/jindi/nlp/datasets/SNLI/snli_1.0 ' \
          '--bert_model bert-base-uncased ' \
          '--task_name snli --output_dir results/SNLI_retrain --cache_dir pytorch_cache  --do_train --do_eval --do_lower_case ' \
          # '--do_resume'

os.system(command)