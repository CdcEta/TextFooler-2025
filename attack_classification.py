# 本文件：用于对单句文本分类模型执行对抗攻击（TextFooler）。
# 关键步骤：词重要性计算、同义词检索、USE 语义相似度过滤、词性约束（POS）。
import argparse
import os
# Suppress verbose TensorFlow logs (INFO/WARNING) to reduce GPU DLL noise
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
# Reduce CUDA allocator fragmentation on small-memory GPUs
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:64')
import numpy as np
import dataloader
from train_classifier import Model
import criteria
import random
import platform
import subprocess

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import tensorflow_hub as hub

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from concurrent.futures import ThreadPoolExecutor

from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig




class USE(object):
    """USE 句向量模块：加载 TFHub 模块并提供语义相似度计算。

    - 通过 `TFHUB_CACHE_DIR` 指定缓存目录
    - 构建图并在会话中计算两段文本的余弦相似度与角度相似度
    """
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        # Prefer GPU if available; enable memory growth to avoid OOM at init
        device = "/CPU:0"
        try:
            # Default to CPU to avoid TF/PyTorch GPU contention; allow opt-in via USE_ON_GPU=1
            force_cpu = os.environ.get('USE_ON_CPU', '1') == '1'
            force_gpu = os.environ.get('USE_ON_GPU') == '1'
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and not force_cpu and force_gpu:
                try:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                except Exception:
                    pass
                device = "/GPU:0"
                print(f"[USE] TensorFlow GPU detected ({gpus[0].name}). Running USE on GPU.")
            elif force_cpu:
                device = "/CPU:0"
                print("[USE] USE_ON_CPU=1 detected or USE_ON_GPU not set. Using CPU for USE.")
            else:
                print("[USE] No TensorFlow GPU detected. Running USE on CPU.")
        except Exception:
            # In TF1 compatibility environments, tf.config may be unavailable; fall back gracefully
            print("[USE] TensorFlow device query failed; defaulting to CPU.")
        config = tf.compat.v1.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        self._device = device
        with tf.device(self._device):
            self.embed = hub.Module(module_url)
        self.sess = tf.compat.v1.Session(config=config)
        self.build_graph()
        self.sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

    def build_graph(self):
        with tf.device(self._device):
            self.sts_input1 = tf.compat.v1.placeholder(tf.string, shape=(None))
            self.sts_input2 = tf.compat.v1.placeholder(tf.string, shape=(None))

            sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
            sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
            self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
            clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
            self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)
            # Raw embedding op for caching-based batched embedding
            self.raw_input = tf.compat.v1.placeholder(tf.string, shape=(None))
            self.raw_encode = tf.nn.l2_normalize(self.embed(self.raw_input), axis=1)
        # Simple embedding cache to avoid recompute on identical windows
        self._emb_cache = {}
        # Increase cache capacity to reduce TFHub calls; keeps memory reasonable
        self._emb_cache_max = 50000

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores

    def semantic_sim_cached(self, sents1, sents2, batch_size=128):
        """Compute semantic similarity using cached embeddings to reduce redundant TFHub calls.

        - Embeds unique strings only once, caches results.
        - Computes cosine similarity and converts to the same score space as semantic_sim.
        """
        # Collect unique strings
        uniq = []
        seen = set()
        for s in sents1:
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        for s in sents2:
            if s not in seen:
                uniq.append(s)
                seen.add(s)

        # Embed missing strings in chunks
        missing = [s for s in uniq if s not in self._emb_cache]
        try:
            for i in range(0, len(missing), batch_size):
                chunk = missing[i:i+batch_size]
                if not chunk:
                    continue
                emb = self.sess.run(self.raw_encode, feed_dict={self.raw_input: chunk})
                for s, v in zip(chunk, emb):
                    # capacity control
                    if len(self._emb_cache) >= self._emb_cache_max:
                        self._emb_cache.clear()
                    self._emb_cache[s] = v
        except Exception:
            # Fallback to original method if embedding fails
            return self.semantic_sim(sents1, sents2)[0]

        # Build arrays for pairwise cosine similarity
        e1 = np.stack([self._emb_cache[s] for s in sents1], axis=0)
        e2 = np.stack([self._emb_cache[s] for s in sents2], axis=0)
        cos = np.sum(e1 * e2, axis=1)
        cos = np.clip(cos, -1.0, 1.0)
        scores = 1.0 - np.arccos(cos)
        return scores

def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    在相似度矩阵中为每个词选取近邻候选词（排除自身）。
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32,
                 auto_gpu=False,
                 gpu_fill_target=0.80,
                 num_workers=2,
                 prefetch_factor=2,
                 fp16=False):
        super(NLI_infer_BERT, self).__init__()

        # Preferred: use model's from_pretrained API (handles config/weights)
        try:
            self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses).cuda()
            print(f"[ModelLoad] Loaded BertForSequenceClassification from '{pretrained_dir}' with num_labels={nclasses}.")
        except Exception as e:
            print(f"[ModelLoad] from_pretrained failed: {e}. Falling back to manual config/weights load.")
            # Fallback: try reading config json; else default vocab size
            cfg_path = os.path.join(pretrained_dir, 'bert_config.json')
            try:
                config = BertConfig.from_json_file(cfg_path)
                print(f"[ModelLoad] Loaded config from {cfg_path}.")
            except Exception as e_cfg:
                print(f"[ModelLoad] Config json load failed: {e_cfg}. Using default vocab size 30522.")
                config = BertConfig(30522)
            self.model = BertForSequenceClassification(config, num_labels=nclasses).cuda()
            # Try to load weights; ignore classifier shape mismatches
            sd_path = os.path.join(pretrained_dir, 'pytorch_model.bin')
            if os.path.exists(sd_path):
                try:
                    state_dict = torch.load(sd_path, map_location='cpu')
                    cw = state_dict.get('classifier.weight')
                    cb = state_dict.get('classifier.bias')
                    if cw is not None and hasattr(cw, 'shape') and cw.shape[0] != nclasses:
                        print(f"[ModelLoad] Drop mismatched classifier.weight {tuple(cw.shape)} for nclasses={nclasses}")
                        state_dict.pop('classifier.weight', None)
                    if cb is not None and hasattr(cb, 'shape') and cb.shape[0] != nclasses:
                        print(f"[ModelLoad] Drop mismatched classifier.bias {tuple(cb.shape)} for nclasses={nclasses}")
                        state_dict.pop('classifier.bias', None)
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"[ModelLoad] Loaded weights from {sd_path} with strict=False.")
                except Exception as e_sd:
                    print(f"[ModelLoad] Warning: failed to load state dict from {sd_path}: {e_sd}")
            else:
                print(f"[ModelLoad] Warning: state dict not found at {sd_path}, using randomly initialized weights.")

        # Optional: compile model for faster inference if supported
        self._compiled = False
        self.model_eager = self.model
        try:
            enable_compile = str(os.environ.get("TORCH_COMPILE", "1")).lower() not in ("0", "false", "no")
            # Detect Triton availability to avoid runtime failures
            triton_ok = False
            try:
                from torch._inductor.utils import has_triton
                triton_ok = bool(has_triton())
            except Exception:
                try:
                    import importlib.util as _ilu
                    triton_ok = _ilu.find_spec("triton") is not None
                except Exception:
                    triton_ok = False
            if enable_compile and hasattr(torch, 'compile') and triton_ok:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._compiled = True
                print("[Perf] torch.compile enabled (mode=reduce-overhead).")
            else:
                if not enable_compile:
                    print("[Perf] torch.compile disabled by TORCH_COMPILE env.")
                elif not triton_ok:
                    print("[Perf] Triton not found; skipping torch.compile.")
        except Exception as e:
            print(f"[Perf] torch.compile unavailable or failed: {e}")

        # Enable TF32/high precision matmul optimizations using new APIs when available
        try:
            if hasattr(torch.backends, 'fp32_precision'):
                # Prefer new granular APIs (PyTorch 2.9+)
                torch.backends.fp32_precision = "tf32"
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.fp32_precision = "tf32"
                # Optional: be explicit for conv/rnn if needed
                if hasattr(torch.backends.cudnn, 'conv'):
                    torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                # Fallback to old flags (pre-2.9)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            # Avoid deprecated API warnings on PyTorch 2.9+; rely on new backends settings
            # Only use legacy API if new fp32_precision backend is not available
            if (not hasattr(torch.backends, 'fp32_precision')) and hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)
        self.auto_gpu = auto_gpu
        self.gpu_fill_target = gpu_fill_target
        # Keep a safe free-memory margin to avoid spill into shared GPU memory
        self.reserve_free_ratio = 0.15
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.fp16 = fp16
        self.current_step = "-"
        # Background CPU pipeline for tokenization/feature prep
        try:
            self._executor = ThreadPoolExecutor(max_workers=max(1, self.num_workers))
        except Exception:
            self._executor = None

    def _tune_batch_size(self, text_data, init_bs):
        if not torch.cuda.is_available():
            return init_bs
        try:
            self.model.eval()
            probe_bs = max(1, min(init_bs, 16))
            sample = text_data[0] if len(text_data) > 0 else ["test"]
            probe_chunk = [sample] * probe_bs
            feats = self.dataset.convert_examples_to_features(probe_chunk, self.dataset.max_seq_length, self.dataset.tokenizer)
            ids = torch.tensor([f.input_ids for f in feats], dtype=torch.long).cuda(non_blocking=True)
            msk = torch.tensor([f.input_mask for f in feats], dtype=torch.long).cuda(non_blocking=True)
            seg = torch.tensor([f.segment_ids for f in feats], dtype=torch.long).cuda(non_blocking=True)
            free_before, total = torch.cuda.mem_get_info()
            with torch.no_grad():
                _ = self.model(ids, seg, msk)
            free_after, _ = torch.cuda.mem_get_info()
            used = max(1, int(free_before - free_after))
            per_sample = max(1, used // probe_bs)
            target = int(free_after * float(self.gpu_fill_target))
            bs_fit = max(1, min(4096, target // per_sample))
            return bs_fit
        except Exception:
            # Fallback: doubling until OOM then backoff
            bs = max(1, init_bs)
            max_bs = bs
            while True:
                try:
                    sample = text_data[:bs] if len(text_data) >= bs else text_data
                    feats = self.dataset.convert_examples_to_features(sample, self.dataset.max_seq_length, self.dataset.tokenizer)
                    ids = torch.tensor([f.input_ids for f in feats], dtype=torch.long).cuda(non_blocking=True)
                    msk = torch.tensor([f.input_mask for f in feats], dtype=torch.long).cuda(non_blocking=True)
                    seg = torch.tensor([f.segment_ids for f in feats], dtype=torch.long).cuda(non_blocking=True)
                    with torch.no_grad():
                        _ = self.model(ids, seg, msk)
                    max_bs = bs
                    bs = bs * 2
                    if bs > 4096:
                        break
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                        bs = max(1, bs // 2)
                        if bs <= max_bs:
                            break
                    else:
                        break
            return max_bs

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # decide effective batch size based on GPU memory
        effective_bs = batch_size
        if self.auto_gpu:
            effective_bs = max(batch_size, self._tune_batch_size(text_data, batch_size))

        total_items = len(text_data)
        processed = 0
        probs_all = []

        # Helper to prepare CPU pinned tensors for a chunk
        def _prepare(chunk_local):
            feats_local = self.dataset.convert_examples_to_features(chunk_local, self.dataset.max_seq_length, self.dataset.tokenizer)
            ids_cpu_local = torch.tensor([f.input_ids for f in feats_local], dtype=torch.long).pin_memory()
            msk_cpu_local = torch.tensor([f.input_mask for f in feats_local], dtype=torch.long).pin_memory()
            seg_cpu_local = torch.tensor([f.segment_ids for f in feats_local], dtype=torch.long).pin_memory()
            return ids_cpu_local, msk_cpu_local, seg_cpu_local

        prefetch_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        i = 0
        # Prepare first batch synchronously, and schedule next in background
        bs = min(effective_bs, total_items - i)
        cur_chunk = text_data[i:i+bs]
        ids_cpu, msk_cpu, seg_cpu = _prepare(cur_chunk)
        next_future = None
        if self._executor is not None and (i + bs) < total_items:
            bs_next = min(effective_bs, total_items - (i + bs))
            next_future = self._executor.submit(_prepare, text_data[i+bs:i+bs+bs_next])

        # Move current batch to GPU
        cur_ids = ids_cpu.to(device='cuda', non_blocking=True)
        cur_mask = msk_cpu.to(device='cuda', non_blocking=True)
        cur_seg = seg_cpu.to(device='cuda', non_blocking=True)
        i += bs

        while True:
            # Compute current batch
            free_before, total_mem = (0, 1)
            try:
                free_before, total_mem = torch.cuda.mem_get_info()
            except Exception:
                pass
            with torch.inference_mode():
                def _forward_batch(ids_t, seg_t, msk_t):
                    if self.fp16:
                        with torch.amp.autocast(device_type='cuda'):
                            return self.model(ids_t, seg_t, msk_t)
                    else:
                        return self.model(ids_t, seg_t, msk_t)
                try:
                    logits = _forward_batch(cur_ids, cur_seg, cur_mask)
                    probs = nn.functional.softmax(logits, dim=-1)
                    probs_all.append(probs)
                except torch.OutOfMemoryError as oom:
                    new_bs = max(1, effective_bs // 2)
                    print(f"[AutoGPU] OOM caught. Splitting batch {effective_bs} -> {new_bs} and retrying.")
                    effective_bs = new_bs
                    local_probs = []
                    offset = 0
                    while offset < cur_ids.size(0):
                        end = min(offset + new_bs, cur_ids.size(0))
                        try:
                            logits_part = _forward_batch(cur_ids[offset:end], cur_seg[offset:end], cur_mask[offset:end])
                        except Exception as e2:
                            if getattr(self, "_compiled", False):
                                print(f"[Perf] torch.compile runtime failed during micro-batch; reverting to eager: {e2}")
                                self.model = self.model_eager
                                self._compiled = False
                                logits_part = _forward_batch(cur_ids[offset:end], cur_seg[offset:end], cur_mask[offset:end])
                            else:
                                raise
                        local_probs.append(nn.functional.softmax(logits_part, dim=-1))
                        offset = end
                    probs_all.append(torch.cat(local_probs, dim=0))
                except Exception as e:
                    if getattr(self, "_compiled", False):
                        print(f"[Perf] torch.compile runtime failed; reverting to eager: {e}")
                        self.model = self.model_eager
                        self._compiled = False
                        logits = _forward_batch(cur_ids, cur_seg, cur_mask)
                        probs = nn.functional.softmax(logits, dim=-1)
                        probs_all.append(probs)
                    else:
                        raise
            free_after, total_mem2 = (0, total_mem)
            try:
                free_after, total_mem2 = torch.cuda.mem_get_info()
            except Exception:
                pass
            processed += cur_ids.size(0)
            print(f"STEP_PROGRESS {self.current_step} {processed}/{total_items}")
            # Adjust batch size based on memory ratio
            if self.auto_gpu and total_mem2:
                free_ratio = (free_after / float(total_mem2))
                used_ratio = 1.0 - free_ratio
                target = float(self.gpu_fill_target)
                growth, backoff = 1.15, 0.6
                # Do not grow if free memory is below reserve margin
                if free_ratio < self.reserve_free_ratio:
                    effective_bs = max(1, int(effective_bs * backoff))
                    print(f"[AutoGPU] Reserve backoff -> {effective_bs} (free_ratio={free_ratio:.2f})")
                else:
                    if used_ratio < target * 0.98:
                        effective_bs = max(1, min(4096, int(effective_bs * growth)))
                        print(f"[AutoGPU] Increase batch size -> {effective_bs} (mem_ratio={used_ratio:.2f})")
                    elif used_ratio > target * 1.02:
                        effective_bs = max(1, int(effective_bs * backoff))
                        print(f"[AutoGPU] Decrease batch size -> {effective_bs} (mem_ratio={used_ratio:.2f})")

            # If finished, break
            if i >= total_items:
                break

            # Ensure next batch CPU tensors are ready
            if next_future is not None:
                try:
                    ids_cpu, msk_cpu, seg_cpu = next_future.result()
                except Exception as e:
                    # Fallback: prepare synchronously
                    bs_ready = min(effective_bs, total_items - i)
                    ids_cpu, msk_cpu, seg_cpu = _prepare(text_data[i:i+bs_ready])
                    next_future = None
            else:
                bs_ready = min(effective_bs, total_items - i)
                ids_cpu, msk_cpu, seg_cpu = _prepare(text_data[i:i+bs_ready])

            # Defer GPU transfer until after current batch completes to avoid double-residency
            next_ids, next_mask, next_seg = None, None, None

            # Schedule the following batch in background while current advances
            if self._executor is not None and (i + ids_cpu.size(0)) < total_items:
                bs_next = min(effective_bs, total_items - (i + ids_cpu.size(0)))
                try:
                    next_future = self._executor.submit(_prepare, text_data[i+ids_cpu.size(0):i+ids_cpu.size(0)+bs_next])
                except Exception:
                    next_future = None
            else:
                next_future = None

            # Swap: move next batch to GPU now to avoid overlapping residency
            next_ids = ids_cpu.to(device='cuda', non_blocking=True)
            next_mask = msk_cpu.to(device='cuda', non_blocking=True)
            next_seg = seg_cpu.to(device='cuda', non_blocking=True)
            cur_ids, cur_mask, cur_seg = next_ids, next_mask, next_seg
            i += ids_cpu.size(0)
        
        # End of while loop

        # Compute any remaining batch only if not fully processed
        if cur_ids is not None and processed < total_items:
            try:
                _free_before, _total_mem = torch.cuda.mem_get_info()
            except Exception:
                pass
            with torch.inference_mode():
                if self.fp16:
                    with torch.amp.autocast(device_type='cuda'):
                        logits = self.model(cur_ids, cur_seg, cur_mask)
                else:
                    logits = self.model(cur_ids, cur_seg, cur_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)
            processed += cur_ids.size(0)
            print(f"STEP_PROGRESS {self.current_step} {processed}/{total_items}")
        print(f"STEP_DONE {self.current_step}")
        return torch.cat(probs_all, dim=0)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    BERT 分类数据封装：将文本切分为 token，并构造 `input_ids`/`input_mask`/`segment_ids`。
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        # Feature cache keyed by (text, max_seq_length) to avoid repeated tokenization
        self._feat_cache = {}
        self._feat_cache_max = 20000

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """将文本序列转换为 BERT 可接受的特征列表。"""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            key = (' '.join(text_a), max_seq_length)
            cached = self._feat_cache.get(key)
            if cached is not None:
                features.append(cached)
                continue
            tokens_a = tokenizer.tokenize(key[0])

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            feat = InputFeatures(input_ids=input_ids,
                                 input_mask=input_mask,
                                 segment_ids=segment_ids)
            features.append(feat)
            # Cache with capacity control
            try:
                if len(self._feat_cache) >= self._feat_cache_max:
                    # Simple reset to keep memory in check
                    self._feat_cache.clear()
                self._feat_cache[key] = feat
            except Exception:
                pass
        return features

    def transform_text(self, data, batch_size=32):
        """将文本批量转换为 `TensorDataset` 并构建 `DataLoader`。"""
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader


def attack(text_ls, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):
    """
    执行 TextFooler 攻击流程：
    - 计算词重要性（leave-one-out）并按重要性降序尝试替换；
    - 基于相似度矩阵为词检索同义词候选；
    - 使用 USE 在滑动窗口上计算语义相似度，过滤低相似候选；
    - 结合 POS 约束过滤语法不一致候选；
    返回：对抗文本、修改词数、原标签、攻击后标签、查询次数。
    """
    # 先检查原始文本的预测结果
    # Substep: original prediction check
    try:
        owner = getattr(predictor, "__self__", None)
        if owner is not None:
            owner.current_step = "Check original prediction"
            print("STEP_START Check original prediction total=1")
    except Exception:
        pass
    orig_probs = predictor([text_ls]).squeeze()
    # 强制拉平成 [num_classes]，避免后续 index_select 广播问题
    if orig_probs.dim() > 1:
        orig_probs = orig_probs.view(-1)
    # 使用 Python int 作为列索引，避免 0-dim Tensor 索引触发不期望的广播
    orig_label = int(torch.argmax(orig_probs).item())
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)  # 词性序列，用于后续 POS 约束过滤

        # get importance score
        leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        # Substep: importance scoring
        try:
            owner = getattr(predictor, "__self__", None)
            if owner is not None:
                owner.current_step = "Importance scoring"
                print(f"STEP_START Importance scoring total={len(leave_1_texts)}")
        except Exception:
            pass
        leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
        # 统一为 [len_text, num_classes] 形状，若输出为 [num_classes, len_text] 则转置
        if leave_1_probs.dim() == 1:
            leave_1_probs = leave_1_probs.view(1, -1)
        elif (
            leave_1_probs.dim() == 2
            and leave_1_probs.size(-1) != orig_probs.size(0)
            and leave_1_probs.size(0) == orig_probs.size(0)
        ):
            leave_1_probs = leave_1_probs.transpose(0, 1)
        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        # 逐样本：max_class_prob - 原样本在该预测类别上的概率
        leave1_max = leave_1_probs.max(dim=-1)[0]
        orig_per_pred = orig_probs.index_select(0, leave_1_probs_argmax)
        import_scores = (
            (orig_prob - leave_1_probs[:, orig_label])
            + (leave_1_probs_argmax != orig_label).float() * (leave1_max - orig_per_pred)
        ).detach().cpu().numpy()

        # 根据重要性得分筛选需要扰动的词（过滤停用词）
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

        # 为待扰动词查找同义词候选
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # 开始逐词替换并攻击
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            # Substep: synonym evaluation at position
            try:
                owner = getattr(predictor, "__self__", None)
                if owner is not None:
                    owner.current_step = f"Synonym evaluation@{idx}"
                    print(f"STEP_START Synonym evaluation@{idx} total={len(new_texts)}")
            except Exception:
                pass
            new_probs = predictor(new_texts, batch_size=batch_size)

            # 计算语义相似度（滑动窗口），用于过滤低相似的替换
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            ref = ' '.join(text_cache[text_range_min:text_range_max])
            cand = list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts))
            # Cached USE similarity to reduce redundant embedding of identical ref window
            semantic_sims = sim_predictor.semantic_sim_cached([ref] * len(new_texts), cand)
            # Ensure semantic_sims length matches candidate count; fallback to non-cached if mismatch
            semantic_sims = np.asarray(semantic_sims).reshape(-1)
            if semantic_sims.shape[0] != len(new_texts):
                try:
                    semantic_sims = sim_predictor.semantic_sim([ref] * len(new_texts), cand)[0]
                    semantic_sims = np.asarray(semantic_sims).reshape(-1)
                except Exception:
                    if semantic_sims.shape[0] > len(new_texts):
                        semantic_sims = semantic_sims[:len(new_texts)]
                    else:
                        # Pad with zeros (conservative filter) if we cannot recompute
                        pad = len(new_texts) - semantic_sims.shape[0]
                        if pad > 0:
                            semantic_sims = np.concatenate([semantic_sims, np.zeros(pad, dtype=np.float32)], axis=0)

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy().astype(np.float32)
            # 过滤语义不相似的候选
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # 过滤词性不兼容的候选
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        # Substep: final evaluation
        try:
            owner = getattr(predictor, "__self__", None)
            if owner is not None:
                owner.current_step = "Final evaluation"
                print("STEP_START Final evaluation total=1")
        except Exception:
            pass
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


def random_attack(text_ls, true_label, predictor, perturb_ratio, stop_words_set, word2idx, idx2word, cos_sim,
                  sim_predictor=None, import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15,
                  synonym_num=50, batch_size=32):
    """
    随机选择一定比例的词进行扰动并尝试攻击：
    - 随机采样扰动位置；
    - 同义词候选检索与语义/词性过滤；
    返回与 `attack` 相同的五元组。
    """
    # 先检查原始文本的预测结果
    # Substep: original prediction check
    try:
        owner = getattr(predictor, "__self__", None)
        if owner is not None:
            owner.current_step = "Check original prediction"
            print("STEP_START Check original prediction total=1")
    except Exception:
        pass
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(text_ls)

        # randomly get perturbed words
        perturb_idxes = random.sample(range(len_text), int(len_text * perturb_ratio))
        words_perturb = [(idx, text_ls[idx]) for idx in perturb_idxes]

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = text_ls[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            # Substep: synonym evaluation at position
            try:
                owner = getattr(predictor, "__self__", None)
                if owner is not None:
                    owner.current_step = f"Synonym evaluation@{idx}"
                    print(f"STEP_START Synonym evaluation@{idx} total={len(new_texts)}")
            except Exception:
                pass
            new_probs = predictor(new_texts, batch_size=batch_size)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            ref = ' '.join(text_cache[text_range_min:text_range_max])
            cand = list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts))
            semantic_sims = sim_predictor.semantic_sim_cached([ref] * len(new_texts), cand)
            # Ensure semantic_sims length matches candidate count; fallback to non-cached if mismatch
            semantic_sims = np.asarray(semantic_sims).reshape(-1)
            if semantic_sims.shape[0] != len(new_texts):
                try:
                    semantic_sims = sim_predictor.semantic_sim([ref] * len(new_texts), cand)[0]
                    semantic_sims = np.asarray(semantic_sims).reshape(-1)
                except Exception:
                    if semantic_sims.shape[0] > len(new_texts):
                        semantic_sims = semantic_sims[:len(new_texts)]
                    else:
                        # Pad with zeros (conservative filter) if we cannot recompute
                        pad = len(new_texts) - semantic_sims.shape[0]
                        if pad > 0:
                            semantic_sims = np.concatenate([semantic_sims, np.zeros(pad, dtype=np.float32)], axis=0)

            num_queries += len(new_texts)
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy().astype(np.float32)
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        # Substep: final evaluation
        try:
            owner = getattr(predictor, "__self__", None)
            if owner is not None:
                owner.current_step = "Final evaluation"
                print("STEP_START Final evaluation total=1")
        except Exception:
            pass
        return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['wordLSTM', 'bert', 'wordCNN'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=True,
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        required=True,
                        help="Path to the USE encoder cache.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=15,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.7,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--auto_gpu",
                        action='store_true',
                        help="Enable auto GPU batch-size tuning to maximize memory usage")
    parser.add_argument("--gpu_fill_target",
                        default=0.9,
                        type=float,
                        help="Target fraction of free GPU memory to fill with batch size (0-1)")
    parser.add_argument("--num_workers",
                        default=2,
                        type=int,
                        help="Number of DataLoader workers for CPU tokenization/prep")
    parser.add_argument("--prefetch_factor",
                        default=2,
                        type=int,
                        help="Prefetch batches per worker (requires num_workers>0)")
    parser.add_argument("--fp16",
                        action='store_true',
                        help="Enable FP16 autocast inference to increase throughput (may allow larger batch)")
    parser.add_argument("--data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="max sequence length for BERT target model")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # Detect hardware and auto-tune params (only override defaults)
    def _get_cpu_name():
        try:
            if os.name == 'nt':
                out = subprocess.check_output(['wmic', 'cpu', 'get', 'Name'], text=True, stderr=subprocess.DEVNULL)
                lines = [l.strip() for l in out.splitlines() if l.strip() and 'Name' not in l]
                if lines:
                    return lines[0]
        except Exception:
            pass
        try:
            return platform.processor() or platform.uname().processor or 'Unknown CPU'
        except Exception:
            return 'Unknown CPU'

    cpu_logical = os.cpu_count() or 1
    cpu_name = _get_cpu_name()

    gpu_name = 'None'
    gpu_mem_gb = 0.0
    cc = 'N/A'
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_mem_gb = float(props.total_memory) / (1024**3)
            cc = f"{props.major}.{props.minor}"
        except Exception:
            pass

    # Recommended params based on hardware
    rec_workers = max(4, min(12, cpu_logical // 2))
    rec_prefetch = 3
    rec_fp16 = torch.cuda.is_available()
    # On Windows/WDDM, keeping more headroom avoids fallback to shared GPU memory
    if os.name == 'nt':
        rec_gft = 0.80
    else:
        rec_gft = 0.9 if gpu_mem_gb <= 4.5 else 0.95
    rec_msl = 256 if gpu_mem_gb >= 4.0 else args.max_seq_length

    # Override only if user kept defaults
    if getattr(args, 'num_workers', 2) == 2:
        args.num_workers = rec_workers
    if getattr(args, 'prefetch_factor', 2) == 2:
        args.prefetch_factor = rec_prefetch
    if not getattr(args, 'fp16', False) and rec_fp16:
        args.fp16 = True
    if getattr(args, 'gpu_fill_target', 0.9) == 0.9:
        args.gpu_fill_target = rec_gft
    if getattr(args, 'max_seq_length', 128) == 128:
        args.max_seq_length = rec_msl

    # Print for GUI parsing
    print(f"HW_CPU name={cpu_name} logical={cpu_logical}")
    print(f"HW_GPU name={gpu_name} mem_gb={gpu_mem_gb:.1f} cc={cc}")
    print(
        f"TUNING auto_gpu={args.auto_gpu} num_workers={args.num_workers} prefetch_factor={args.prefetch_factor} "
        f"fp16={args.fp16} max_seq_length={args.max_seq_length} gpu_fill_target={args.gpu_fill_target}"
    )

    # get data to attack
    texts, labels = dataloader.read_corpus(args.dataset_path)
    data = list(zip(texts, labels))
    data = data[:args.data_size] # choose how many samples for adversary
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'wordLSTM':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'wordCNN':
        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100, cnn=True).cuda()
        checkpoint = torch.load(args.target_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)
    elif args.target_model == 'bert':
        model = NLI_infer_BERT(args.target_model_path,
                               nclasses=args.nclasses,
                               max_seq_length=args.max_seq_length,
                               batch_size=args.batch_size,
                               auto_gpu=args.auto_gpu,
                               gpu_fill_target=args.gpu_fill_target,
                               num_workers=getattr(args, 'num_workers', 2),
                               prefetch_factor=getattr(args, 'prefetch_factor', 2),
                               fp16=getattr(args, 'fp16', False))
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    idx2word = {}
    word2idx = {}

    print("Building vocab...")
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        try:
            cos_sim = np.load(args.counter_fitting_cos_sim_path, mmap_mode='r')
        except Exception:
            cos_sim = np.load(args.counter_fitting_cos_sim_path)
    else:
        # calculate the cosine similarity matrix
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        product = np.dot(embeddings, embeddings.T)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        cos_sim = product / np.dot(norm, norm.T)
    print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    log_file = open(os.path.join(args.output_dir, 'results_log'), 'a')

    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')
    for idx, (text, true_label) in enumerate(data):
        if idx % 20 == 0:
            print('{} samples out of {} have been finished!'.format(idx, args.data_size))
        if args.perturb_ratio > 0.:
            new_text, num_changed, orig_label, \
            new_label, num_queries = random_attack(text, true_label, predictor, args.perturb_ratio, stop_words_set,
                                                    word2idx, idx2word, cos_sim, sim_predictor=use,
                                                    sim_score_threshold=args.sim_score_threshold,
                                                    import_score_threshold=args.import_score_threshold,
                                                    sim_score_window=args.sim_score_window,
                                                    synonym_num=args.synonym_num,
                                                    batch_size=args.batch_size)
        else:
            new_text, num_changed, orig_label, \
            new_label, num_queries = attack(text, true_label, predictor, stop_words_set,
                                            word2idx, idx2word, cos_sim, sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size)

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)
        if true_label != new_label:
            adv_failures += 1

        changed_rate = 1.0 * num_changed / len(text)

        if true_label == orig_label and true_label != new_label:
            changed_rates.append(changed_rate)
            orig_texts.append(' '.join(text))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)

    message = 'For target model {}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}\n'.format(args.target_model,
                                                                     (1-orig_failures/1000)*100,
                                                                     (1-adv_failures/1000)*100,
                                                                     np.mean(changed_rates)*100,
                                                                     np.mean(nums_queries))
    print(message)
    log_file.write(message)

    with open(os.path.join(args.output_dir, 'adversaries.txt'), 'w') as ofile:
        for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
            ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))

if __name__ == "__main__":
    main()