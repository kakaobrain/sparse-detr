#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_sparse_detr_0.1
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --eff_query_init \
    --eff_specific_head \
    --rho 0.1 \
    --use_enc_aux_loss \
    ${PY_ARGS}
