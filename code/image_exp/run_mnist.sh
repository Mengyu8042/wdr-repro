#!/usr/bin/env bash
# Launch an MNIST experiment.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

ess_threshold=0.5
P=64
R=3
class_batch_size=100
strategy="t-wdr"
device_id=0
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/mnist_P${P}_${strategy}_${timestamp}.log"

echo "[run_mnist.sh] Running $strategy on GPU $device_id -> $log_file"
CUDA_VISIBLE_DEVICES=$device_id python -u run_image.py \
    --model_and_diffusion_config image_confs/mnist_model_and_diffusion_conf.yml \
    --task_config                image_confs/task_class_cond_gen_mnist_conf.yml \
    --sampler_config             image_confs/base_sampler_conf.yml \
    --ess_threshold $ess_threshold \
    --P $P --R $R \
    --class_batch_size $class_batch_size \
    --classifier_guidance_scale 1 \
    --sampler "tds" \
    --resample_strategy "$strategy" \
    --t_truncate_percent 0.0 \
    --compute_fid True \
    --compute_lpips True \
    2>&1 | tee "$log_file"
echo "[run_mnist.sh] MNIST run finished. Results in outputs_*/class_cond_gen/, log in $log_file."
