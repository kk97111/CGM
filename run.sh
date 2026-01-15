#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
PYTHON="${PYTHON:-your_path_to_python_env}"
DATASET_ROOT="${DATASET_ROOT:-your_path_to_dataset}"

DATASETS=("peerread")
MODES=("hybrid")
NUM_NODES=(2500)
MERGE_NUM=(0 2 4 8 10 20 50 100)
COMPRESSION_MODE=("both" "structure" "semantic")

for dataset in "${DATASETS[@]}"; do
  for compression_mode in "${COMPRESSION_MODE[@]}"; do
    for mode in "${MODES[@]}"; do
      for num_nodes in "${NUM_NODES[@]}"; do
        for merge_num in "${MERGE_NUM[@]}"; do
          echo "=== Running: $dataset ($mode) Compression Mode:($compression_mode) num_nodes:($num_nodes) Merge Num:($merge_num) ==="
          "$PYTHON" ours_3.py \
            --dataset_root "$DATASET_ROOT" \
            --dataset_name "$dataset" \
            --mode "$mode" \
            --phase "prefetch" \
            --batch_size 16 \
            --num_nodes "$num_nodes" \
            --epochs 10 \
            --merge_num "$merge_num" \
            --compression_mode "$compression_mode"
          echo
        done
      done
    done
  done
done

echo "All done!"
