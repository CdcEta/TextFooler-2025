# MR（Movie Review）情感分类的 BERT 微调启动脚本（调用 run_classifier.py）。
# 注意：将 --data_dir 调整为你的 MR 数据所在目录。
import os

# 根据需要开启/关闭训练与评估，并设置输出目录与缓存目录。
command = 'python run_classifier.py --data_dir /data/medg/misc/jindi/nlp/datasets/mr ' \
          '--bert_model bert-base-uncased ' \
          '--task_name mr --output_dir results/mr_retrain --cache_dir pytorch_cache --do_train --do_eval ' \
          '--do_lower_case '

os.system(command)