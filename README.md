# Anonymous Repository for ARR Submission

This repository contains the code implementation of "Rethinking Co-citation: A Context-Level Graph Model for Citation Recommendation", submitted to ARR 2026.

This repository is anonymized for double-blind review. All identifying information will be released after acceptance.

## Overview

The overview of the proposed Compressed Graph Model (CGM) is shown below.

![Overview](main.png)

## Datasets

We adopt four widely used public datasets:
- FullPaperPeerRead
- ACL-200
- RefSeer
- ArXiv

## Evaluation Phases

Our model supports two evaluation phases:
- `prefetch`: Recommend citations from the entire corpus (no candidate list required).
- `rerank`: Re-rank a provided candidate list.

## 1. Environment

### Requirements

- Python >= 3.10
- transformers==4.57.1
- torch==2.9.0
- spacy==3.8.7
- scikit-learn==1.7.2

### Install required Python packages

```bash
pip install -r requirements.txt
```

## 2. Work flow
### Step 1: Data Preparation
Download the dataset files from the public source:
- [Google Drive](https://drive.google.com/drive/u/0/folders/11n4YVHgUPfzetJi-y5voFpmRIjiBM0lQ)


Set the dataset root directory:
```bash
export DATASET_ROOT=/path/to/dataset_root
```

### Step 2: Candidate Generation (for rerank phase)
To run the model in the rerank phase, you need a test file with candidate lists.

By default, we use BM25 to generate a candidate list of top 100 papers per test context. If the ground-truth cited paper is not in the top-100 list, we add it to the candidate list to ensure evaluation is valid.

```bash
python generate_candidates.py \
  --dataset_root $DATASET_ROOT \
  --dataset_name peerread \
  --model_type bm25 \
  --top_k 100

```
This generates test_candidates.json under the corresponding dataset folder.


### Step 3: Run Experiments

#### Option A: Hyper-parameter sweep
```bash
bash run.sh
```
#### Option B: Run a single experiment
Prefetch phase:
```bash
CUDA_VISIBLE_DEVICES=0 python cgm.py \
  --phase prefetch \
  --dataset_root $DATASET_ROOT \
  --dataset_name peerread \
  --top_k 1000 \
  --epochs 10 \
  --num_nodes 2500 \
  --mode hybrid \
  --compression_mode both \
  --merge_num 4

```

Rerank phase:
```bash
CUDA_VISIBLE_DEVICES=0 python cgm.py \
  --phase rerank \
  --dataset_root $DATASET_ROOT \
  --dataset_name peerread \
  --top_k 100 \
  --epochs 10 \
  --num_nodes 2500 \
  --mode hybrid \
  --compression_mode both \
  --merge_num 4
```

## 3. Ablation Studies
Control the parameters when running for ablation studies:
- Vanilla model: --mode vanilla
- w/o GCN: --no_GCN
- w/o structure: --compression_mode semantic
- w/o semantic: --compression_mode structure
- w/o AGG: (please specify the corresponding flag/setting in the implementation)
