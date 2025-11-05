# TextFooler 项目使用说明

本项目基于 TextFooler 对抗攻击算法，用于对文本分类模型（如 BERT、LSTM、CNN 等）进行对抗样本生成与攻击实验。以下内容记录了实验环境配置、运行步骤以及常见问题处理。

---

## 📦 1. 安装依赖

进入项目根目录：

```bash
cd TextFooler-2025
pip install -r requirements.txt
```

### ⚠️ 依赖版本说明

原项目依赖存在版本冲突，如 TensorFlow 和 Pattern 等库无法正确安装。下面是修改后的可运行版本依赖（部分包做了更新，适配 Python 3.10）：

```bash
absl-py==2.1.0
astor==0.8.1
beautifulsoup4==4.9.1
boto3==1.14.7
botocore==1.17.7
certifi==2020.4.5.2
chardet==3.0.4
click==7.1.2
docutils==0.15.2
feedparser==6.0.10
gast==0.4.0
grpcio==1.51.1  
h5py==3.8.0
idna==2.9
importlib-metadata==1.6.1
jmespath==0.10.0
joblib==0.15.1
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.2
lxml==4.9.2
Markdown==3.3.7
nltk==3.5
numpy==1.23.5
protobuf==3.19.6
python-dateutil==2.8.1
python-docx==0.8.10
regex==2020.6.8
requests==2.24.0
s3transfer==0.3.3
six==1.15.0
soupsieve==2.0.1
tensorboard==2.10.1
tensorflow-gpu==2.10.1
tensorflow-hub==0.12.0
termcolor==2.1.1
torch==1.13.1
tqdm==4.46.1
urllib3==1.25.9
Werkzeug==2.2.3
zipp==3.1.0
python==3.10
transformers==4.33.3
```

详见 `requirements.txt`。

---

## 🔧 2. 安装 ESIM 包（用于 NLI 任务）

```bash
cd ESIM
python setup.py install
cd ..
```

---

## 📂 3. 准备数据和预训练资源

- 攻击使用的数据可以直接放置到 `data/` 目录下
-   若打算训练目标模型：
下载作者提供的完整且处理好的[数据集](https://drive.google.com/open?id=1N-FYUa5XN8qDs4SgttQQnrkeTXXAXjTv)（放在./TextFooler-master/traindata/xx），修改./TextFooler-master/BERT/run_classifier_XX.py中的data_dir，进入目录./TextFooler-master/BERT并且运行指令：
```bash
python run_classifier_XX.py
```

以 AG 新闻分类为例：

```bash
cd TextFooler/BERT
python run_classifier_AG.py
```

训练完成会在 `BERT/results/ag/` 下生成以下文件：

```
bert_config.json
eval_results.txt
pytorch_model.bin
vocab.txt
```

---

## 🔁 4. 预计算词向量相似度矩阵（可选）

若使用 `counter-fitted-vectors.txt`，可提前计算余弦相似度矩阵，节省攻击计算时间：

```bash
python comp_cos_sim_mat.py ./Embeddings/counter-fitted-vectors.txt
```

生成文件：`cos_sim_counter_fitting.npy`

---

## 💥 5. 运行攻击脚本

### 常用参数说明

| 参数 | 含义 |
|------|------|
| `--dataset_path` | 数据集路径 |
| `--target_model` | 目标模型，如 `bert`、`lstm` |
| `--target_model_path` | 模型权重路径，可以下载作者[训练过的BERT模型参数](https://drive.google.com/drive/folders/1wKjelHFcqsT3GgA7LzWmoaAHcUkP4c7B?usp=sharing)，[训练过的LSTM模型参数](https://drive.google.com/drive/folders/108myH_HHtBJX8MvhBQuvTGb-kGOce5M2?usp=sharing)，[训练过的CNN模型参数](https://drive.google.com/drive/folders/1Ifowzfers0m1Aw2vE8O7SMifHUhkTEjh?usp=sharing) |
| `--counter_fitting_embeddings_path` | 反拟合词向量路径 |
| `--counter_fitting_cos_sim_path` | 预计算相似度矩阵路径（可选） |
| `--USE_cache_path` | USE 模型缓存路径（为空则自动下载） |

### 示例：攻击 BERT 模型

```bash
python attack_classification.py \
  --dataset_path "./data" \
  --target_model bert \
  --target_model_path "./BERT/results/ag" \
  --counter_fitting_embeddings_path "./Embeddings/counter-fitted-vectors.txt" \
  --counter_fitting_cos_sim_path "./Embeddings/cos_sim_counter_fitting.npy" \
  --USE_cache_path "./USE_cache_path"
```


## 📁 项目结构

```
TextFooler/
│
├── attack_classification.py        # 单句分类攻击主脚本
├── attack_nli.py                   # NLI（句对）攻击主脚本
├── train_classifier.py             # 训练单句分类器（LSTM/CNN）
├── run_attack_classification.py    # 启动/示例脚本（分类攻击）
├── run_attack_nli.py               # 启动/示例脚本（NLI攻击）
├── comp_cos_sim_mat.py             # 生成 counter-fitted 相似度矩阵（或修改为 Top-K）
├── dataloader.py                   # 数据加载 / pad / batch 化
├── modules.py                      # 模型模块（Embedding、CNN、LSTM 等）
├── criteria.py                     # 语义/POS/时态约束工具
├── requirements.txt
├── .DS_Store
├── README.md
│
├── data/                           # 数据目录
│   ├── ag                         # AG 新闻分类数据集
│   ├── fake                      # fake news / fake reviews 数据集
│   ├── imdb                       # IMDB 影评（情感分类）数据集
│   ├── mnli                       # MNLI 原始/通用集（句对），常用于 NLI
│   ├── mnli_matched              # MNLI matched 验证集
│   └── mnli_mismatched            # MNLI mismatched 验证集
│   └──…
│
├── traindata/                        # 用于训练的数据目录
│   ├── ag                         # AG 新闻分类 数据集
│   │   ├── test.csv
│   │   ├── test_tok.csv
│   │   ├── train.csv
│   │   ├── train_tok.csv
│   │   ├── proc.py
│   │   └──…
│   └──…
│
├── Embedding/                      # 向量目录（预训练词向量等）
│   ├── glove.6B.300d.txt                  # GloVe 预训练词向量
│   ├── counter-fitted-vectors.txt       # Counter-fitted 同义词词向量
│   └── cos_sim_counter_fitting.npy  # 预计算的词向量相似度矩阵
│ 
├── BERT/                               # BERT 模块
│   ├── __init__.py                     # 包声明
│   ├── extract_features.py             # 提取 BERT 表征特征
│   ├── file_utils.py                   # 模型/缓存路径管理
│   ├── modeling.py                     # Transformer 模型与分类器结构
│   ├── optimization.py                 # 优化器 & warmup 策略
│   ├── tokenization.py                 # 分词器 & WordPiece 实现
│   ├── run_classifier.py               # 通用 Fine-tuning 脚本
│   ├── run_classifier_AG.py            # AG 新闻分类微调
│   ├── run_classifier_Fake.py          # 假新闻检测微调
│   ├── run_classifier_IMDB.py          # IMDB 情感分类微调
│   ├── run_classifier_mnli.py          # MNLI 自然语言推理任务微调
│   ├── run_classifier_MR.py            # MR（Movie Review）分类任务
│   ├── run_classifier_snli.py          # SNLI 推理任务微调
│   ├── run_classifier_Yelp.py          # Yelp 评论分类任务微调:
│   ├── pytorch_cache
│   └── results          # 攻击 / 训练输出目录
│          ├── ag   
│         │   ├── bert_config.json
│         │   ├── eval_results.txt
│         │   ├── pytorch_model.bin
│         │   └── vocab.txt
│          └──…         
│
├── ESIM/                           #  ESIM 模型目录（若使用 NLI ESIM）
│   ├── esim/
│   ├── scripts/
│   ├── .DS_Store
│   └── setup.py
│
└── tf_cache/                       # USE（Universal Sentence Encoder）缓存目录
```

---

## 🛠️ 常见问题记录

### 1. `pattern.en` 导入失败
将 `from pattern.en import ...` 修改为 `from pattern.text.en import ...`。

### 2. NLTK 资源缺失

```python
import nltk
nltk.download(['punkt', 'averaged_perceptron_tagger', 'universal_tagset', 'wordnet', 'omw-1.4'])
```

⚠️ 若 `wordnet` 下载失败，手动放置语料包到：
`C:\Users\username\AppData\Roaming\nltk_data\corpora`

### 3. CUDA 报错

`AssertionError: Torch not compiled with CUDA enabled`  
解决方式：安装 GPU 版本的 PyTorch 或在本地重新配置 CUDA + PyTorch。

### 4. TensorFlow 版本冲突

TensorFlow 1.x API 在 2.x 环境不兼容，需修改代码：

```python
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
```

并将所有 TF1 API 替换为 `tf.compat.v1.xxx`

---

## ✅ 致谢

本项目基于官方 TextFooler 代码修改优化，适配新版依赖与环境。欢迎在原项目基础上进行二次开发或复现实验结果。
 
 方法与结果参考以论文： Jin, Di, et al. "[Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment](https://arxiv.org/pdf/1907.11932.pdf)." arXiv preprint arXiv:1907.11932 (2019)

