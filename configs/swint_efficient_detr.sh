#!/usr/bin/env bash

set -x

EXP_DIR=exps/swint_efficient_detr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --backbone swin-t \
    --with_box_refine \
    --two_stage \
    --eff_query_init \
    --eff_specific_head \
    ${PY_ARGS}
