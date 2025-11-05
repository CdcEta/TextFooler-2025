# 假新闻/评论数据集的 BERT 微调启动脚本（调用 run_classifier.py）。
# 注意：将 --data_dir 设置为本地实际数据路径。
import os

# 组装命令并调用主运行脚本；核心参数：数据目录、预训练模型、任务名、输出与缓存目录等。
command = 'python run_classifier.py --data_dir /afs/csail.mit.edu/u/z/zhijing/proj/to_di/data/fake ' \
          '--bert_model bert-base-uncased --max_seq_length 256 --train_batch_size 16 ' \
          '--task_name fake --output_dir results/fake --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case '

os.system(command)