# IMDB 情感分类的 BERT 微调启动脚本（调用 run_classifier.py）。
# 注意：将 --data_dir 调整为本地实际路径；其他超参可按需修改。
import os

# 训练周期、批大小、序列长度等参数可根据资源调整。
command = 'python run_classifier.py --data_dir /data/medg/misc/jindi/nlp/datasets/imdb ' \
          '--bert_model bert-base-uncased --max_seq_length 256 --train_batch_size 32 ' \
          '--task_name imdb --output_dir results/imdb --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case ' \
          '--num_train_epochs 3.'

os.system(command)