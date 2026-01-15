# #!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
# PYTHON=/home/yingpeng/miniconda3/envs/torchtune/bin/python
# DATASET_ROOT=/data/yingpeng/citationRec_dyp/dataset/raw

# DATASETS=("peerread" "acl")
# MODES=( "hybrid") #
# GCN=( True False )
# CONTEXT_ALIGNMENT=( True False )
# NUM_NODES=( 250 500 1000 2500 5000 10000 )

# for dataset in "${DATASETS[@]}"; do
#   for mode in "${MODES[@]}"; do
#     for num_nodes in "${NUM_NODES[@]}"; do
#       echo "=== Running: $dataset ($mode) ==="
#       $PYTHON ours_3.py \
#         --dataset_root "$DATASET_ROOT" \
#         --dataset_name "$dataset" \
#         --mode "$mode" \
#         --phase "prefetch" \
#         --batch_size 16 \
#         --num_nodes "$num_nodes" \
#         --epochs 10 \
#         --GCN True \
#         --context_alignment True \

#       echo
#     done
#   done
# done

# echo "All done!"


export CUDA_VISIBLE_DEVICES=1
PYTHON=/home/yingpeng/miniconda3/envs/torchtune/bin/python
DATASET_ROOT=/data/yingpeng/citationRec_dyp/dataset/raw

DATASETS=("peerread")
MODES=( "hybrid") #
GCN=( False )
CONTEXT_ALIGNMENT=( False )
NUM_NODES=( 9363 )
# NUM_NODES=( 5000 )


for dataset in "${DATASETS[@]}"; do
  for mode in "${MODES[@]}"; do
    for num_nodes in "${NUM_NODES[@]}"; do
      for gcn in "${GCN[@]}"; do
        for context_alignment in "${CONTEXT_ALIGNMENT[@]}"; do
          echo "=== Running: $dataset ($mode) GCN:($gcn) Context Alignment:($context_alignment) num_nodes:($num_nodes) ==="
          $PYTHON ours_3.py \
            --dataset_root "$DATASET_ROOT" \
            --dataset_name "$dataset" \
            --mode "$mode" \
            --phase "rerank" \
            --batch_size 16 \
            --num_nodes "$num_nodes" \
            --epochs 10 \
            --GCN "$gcn" \
            --context_alignment "$context_alignment" \

          echo
        done
      done
    done
  done
done

echo "All done!"
