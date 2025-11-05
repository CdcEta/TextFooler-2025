# Yelp 评论分类的 BERT 微调启动脚本（调用 run_classifier.py）。
# 注意：将 --data_dir 设置为本地实际数据路径；可调整 max_seq_length、batch_size、epochs 等超参。
import os

# 组装命令并执行；训练与评估输出在 results/yelp。
command = 'python run_classifier.py --data_dir /afs/csail.mit.edu/u/z/zhijing/proj/to_di/data/yelp ' \
          '--bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 ' \
          '--task_name yelp --output_dir results/yelp --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case ' \
          '--num_train_epochs 2.'

os.system(command)