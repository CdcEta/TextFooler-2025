import os

cmd = (
    'python attack_classification.py '
    '--dataset_path "E:\\TAAD\\\TextFooler-2025\\data\\ag" '
    '--target_model bert '
    '--target_model_path "E:\\TAAD\\\TextFooler-2025\\BERT\\results\\ag-result" '
    '--counter_fitting_embeddings_path "E:\\TAAD\\\TextFooler-2025\\Embeddings\\counter-fitted-vectors.txt" '
    '--counter_fitting_cos_sim_path "E:\\TAAD\\\TextFooler-2025\\Embeddings\\cos_sim_counter_fitting.npy" '
    '--USE_cache_path "E:\\TAAD\\\TextFooler-2025\\USE_cache_path"'
)

print("馃殌 Running command:\n", cmd)
os.system(cmd)

