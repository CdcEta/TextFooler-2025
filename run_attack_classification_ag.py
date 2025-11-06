import os

cmd = (
    'python -u attack_classification.py '
    '--dataset_path "E:\\TAAD\\TextFooler-master\\data\\ag" '
    '--target_model bert '
    '--target_model_path "E:\\TAAD\\TextFooler-master\\BERT\\results\\ag" '
    '--counter_fitting_embeddings_path "E:\\TAAD\\TextFooler-master\\Embeddings\\counter-fitted-vectors.txt" '
    '--counter_fitting_cos_sim_path "E:\\TAAD\\TextFooler-master\\Embeddings\\cos_sim_counter_fitting.npy" '
    '--USE_cache_path "E:\\TAAD\\TextFooler-master\\USE_cache_path"'
    ' --auto_gpu --num_workers 6 --prefetch_factor 3 --fp16 --max_seq_length 256 '
)

print("🚀 Running command:\n ", cmd)
panel_cmd = 'python -u panel_runner_gui.py --cmd "' + cmd.replace('"', '\\"') + '"'
os.system(panel_cmd)
