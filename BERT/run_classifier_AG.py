# AG 新闻分类任务的 BERT 微调启动脚本（调用 run_classifier.py）。
# 注意：请将 data_dir 修改为本地实际数据路径，例如本仓库的 traindata/ag。
import os

data_dir = "E:/TAAD/TextFooler-master/traindata/ag"

# 组装命令并调用主运行脚本；参数含义详见 run_classifier.py 的 --help。
command = f'python run_classifier.py --data_dir "{data_dir}" ' \
          '--bert_model bert-base-uncased ' \
          '--task_name ag ' \
          f'--output_dir results/ag ' \
          '--cache_dir pytorch_cache --do_train --do_eval --do_lower_case '

print(">>> Running command:\n", command)
os.system(command)

