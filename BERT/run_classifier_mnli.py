# MNLI（多领域自然语言推理）任务的 BERT 微调/评估脚本（调用 run_classifier.py）。
# 注意：将 --data_dir 设置为 MNLI 数据路径；如需从断点继续，保留 --do_resume 参数。
import os

# 仅评估示例；如需训练可添加 --do_train 并配置其他参数。
command = 'python run_classifier.py --data_dir /data/medg/misc/jindi/nlp/datasets/MNLI ' \
          '--bert_model bert-base-uncased ' \
          '--task_name mnli --output_dir results/MNLI --cache_dir pytorch_cache --do_eval --do_lower_case ' \
          '--do_resume'

os.system(command)