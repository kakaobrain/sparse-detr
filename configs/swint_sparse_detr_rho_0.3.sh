#!/usr/bin/env bash

set -x

EXP_DIR=exps/swint_sparse_detr_0.3
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --backbone swin-t \
    --with_box_refine \
    --two_stage \
    --eff_query_init \
    --eff_specific_head \
    --rho 0.3 \
    --use_enc_aux_loss \
    ${PY_ARGS}
