"""Class-conditional image generation with TDS and the WDR / T-WDR resamplers."""

import os
import random
import time
import csv
import json
from collections import defaultdict
from functools import partial
from typing import Optional

import fire
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import yaml
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from image_diffusion import dist_util
from image_diffusion.image_util import imwrite, toU8
from image_diffusion.operators import ConditioningMethod
from image_diffusion.script_util import (  # noqa: E402
    NUM_CLASSES,
    create_model_and_diffusion,
    load_classifier,
    model_and_diffusion_defaults,
    sampler_defaults,
    select_args,
)
from smc_utils.feynman_kac_pf import smc_FK
from utils import yamlread

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL("libgcc_s.so.1")
except Exception:
    pass


BETWEEN_RUN_PAIR_COUNT = 5000


class TensorImageDataset(Dataset):
    """
    Wrap (N, C, H, W) tensors so torch-fidelity can consume them directly.
    """

    def __init__(self, imgs: th.Tensor):
        x = imgs.detach().cpu()

        if x.dtype != th.uint8:
            if x.min().item() < 0:
                x = (x + 1.0) / 2.0
            x = x.clamp(0, 1)
            x = (x * 255.0).round().to(th.uint8)

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        self.x = x

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx]


def build_real_pool_mnist(root="./data", per_class_count=2000):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root=root, train=True, download=True, transform=tfm)

    buckets = {i: [] for i in range(10)}
    for img, lab in ds:
        lab = int(lab)
        if len(buckets[lab]) < per_class_count:
            x = img
            if x.size(0) == 1:
                x = x.repeat(3, 1, 1)
            x = (x * 255.0).round().to(th.uint8)
            buckets[lab].append(x)
        if all(len(buckets[i]) >= per_class_count for i in range(10)):
            break

    return {lab: th.stack(buckets[lab], 0) for lab in range(10)}


def build_real_pool_cifar10(root="./data", per_class_count=2000):
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)

    buckets = {i: [] for i in range(10)}
    for img, lab in ds:
        lab = int(lab)
        if len(buckets[lab]) < per_class_count:
            x = (img * 255.0).round().to(th.uint8)
            buckets[lab].append(x)
        if all(len(buckets[i]) >= per_class_count for i in range(10)):
            break

    return {lab: th.stack(buckets[lab], 0) for lab in range(10)}


@th.no_grad()
def fid_torch_fidelity(real_imgs: th.Tensor, gen_imgs: th.Tensor, device="cuda") -> Optional[float]:
    """
    real_imgs, gen_imgs: (N, C, H, W), can be in [-1, 1] or [0, 1].
    Returns FID only.
    """

    if real_imgs.size(0) == 0 or gen_imgs.size(0) == 0:
        return None

    use_cuda = str(device) != "cpu" and th.cuda.is_available()
    metrics = calculate_metrics(
        input1=TensorImageDataset(real_imgs),
        input2=TensorImageDataset(gen_imgs),
        cuda=use_cuda,
        fid=True,
        verbose=False,
    )
    fid_val = metrics.get("frechet_inception_distance")
    return float(fid_val) if fid_val is not None else None


def _prepare_lpips(device):
    net = lpips.LPIPS(net="vgg").to(device).eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net


def _ensure_lpips_input(x: th.Tensor, device: th.device) -> th.Tensor:
    x = x.to(device)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


@th.no_grad()
def _lpips_pair_distances(
    left: th.Tensor,
    right: th.Tensor,
    lpips_fn,
    batch_size: int = 64,
) -> np.ndarray:
    if lpips_fn is None or left.size(0) == 0 or right.size(0) == 0:
        return np.asarray([], dtype=np.float32)
    if left.size(0) != right.size(0):
        raise ValueError("left and right must have the same batch size for LPIPS.")

    device = next(lpips_fn.parameters()).device
    left = _ensure_lpips_input(left, device)
    right = _ensure_lpips_input(right, device)

    vals = []
    for start in range(0, left.size(0), batch_size):
        end = min(start + batch_size, left.size(0))
        dist = lpips_fn(left[start:end], right[start:end])
        vals.append(dist.view(-1).cpu())

    if not vals:
        return np.asarray([], dtype=np.float32)
    return th.cat(vals, 0).numpy()


@th.no_grad()
def _lpips_stats(x: th.Tensor, lpips_fn) -> dict:
    """
    x: (N, 1/3, H, W) in [-1, 1]
    Returns mean pairwise LPIPS.
    """

    if lpips_fn is None or x.size(0) < 2:
        return {"mean": None}

    device = next(lpips_fn.parameters()).device
    x = _ensure_lpips_input(x, device)

    dists = []
    for i in range(x.size(0) - 1):
        dist = lpips_fn(x[i : i + 1], x[i + 1 :])
        dists.append(dist.view(-1).cpu())

    if not dists:
        return {"mean": None}

    values = th.cat(dists, 0).numpy()
    return {"mean": float(values.mean())}


def _append_row(csv_path, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _set_all_seeds(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    th.manual_seed(seed_value)
    th.cuda.manual_seed_all(seed_value)


def _build_class_cond_schedule(class_label: Optional[int], num_replications: int):
    labels = [int(class_label)] if class_label is not None else list(range(10))
    schedule = []
    for replication_id in range(num_replications):
        for label in labels:
            schedule.append((int(label), int(replication_id)))
    return schedule


def _stats_dict(values) -> dict:
    if values is None:
        return {"mean": None, "std": None}
    arr = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None}
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _per_rep_means(records, key):
    """For each replication, average `key` across all classes in that rep.
    Returns one value per rep; across-rep std on this list reflects rep-to-rep variation
    with the class effect averaged out."""
    by_rep = defaultdict(list)
    for r in records:
        v = r.get(key)
        if v is not None and np.isfinite(float(v)):
            by_rep[r["replication"]].append(float(v))
    return [float(np.mean(vs)) for vs in by_rep.values() if vs]


def _summarize_run_records(
    records,
    between_lpips_per_rep=None,
    between_ratio_per_rep=None,
) -> dict:
    """For per-iter metrics: collapse the class axis first (mean over classes within each rep),
    then report mean/std across the per-rep values."""
    n_runs = len(records)
    n_images = int(sum(record["n_particles"] for record in records))
    n_correct = int(sum(record["n_correct"] for record in records))

    summary = {
        "n_runs": n_runs,
        "n_images": n_images,
        "n_correct": n_correct,
        "acc_overall": float(n_correct / n_images) if n_images > 0 else None,
        "acc": _stats_dict(_per_rep_means(records, "acc")),
        "lpips_within_mean": _stats_dict(_per_rep_means(records, "lpips_within_mean")),
        "between_lpips_rep_count": len(between_lpips_per_rep) if between_lpips_per_rep is not None else 0,
        "between_lpips": _stats_dict(between_lpips_per_rep),
        "between_over_within_rep_count": len(between_ratio_per_rep) if between_ratio_per_rep is not None else 0,
        "between_over_within": _stats_dict(between_ratio_per_rep),
    }
    return summary


def _sample_between_run_values(run_entries, lpips_fn, seed: int):
    valid_entries = [
        entry
        for entry in run_entries
        if entry["samples"].size(0) > 0
        and entry["record"]["lpips_within_mean"] is not None
        and entry["record"]["lpips_within_mean"] > 0
    ]
    if lpips_fn is None or len(valid_entries) < 2:
        empty = np.asarray([], dtype=np.float32)
        empty_int = np.asarray([], dtype=np.int64)
        return empty, empty, empty_int, empty_int

    rng = np.random.default_rng(seed)
    left_images = []
    right_images = []
    denominators = []
    left_reps = []
    ratio_left_reps = []

    for _ in range(BETWEEN_RUN_PAIR_COUNT):
        first = int(rng.integers(0, len(valid_entries)))
        second = int(rng.integers(0, len(valid_entries) - 1))
        if second >= first:
            second += 1

        entry_a = valid_entries[first]
        entry_b = valid_entries[second]
        idx_a = int(rng.integers(0, entry_a["samples"].size(0)))
        idx_b = int(rng.integers(0, entry_b["samples"].size(0)))

        left_images.append(entry_a["samples"][idx_a])
        right_images.append(entry_b["samples"][idx_b])
        left_reps.append(int(entry_a["record"]["replication"]))
        denominators.append(
            0.5
            * (
                entry_a["record"]["lpips_within_mean"]
                + entry_b["record"]["lpips_within_mean"]
            )
        )

    distances = _lpips_pair_distances(th.stack(left_images, 0), th.stack(right_images, 0), lpips_fn)
    denom_arr = np.asarray(denominators, dtype=np.float64)
    left_reps_arr = np.asarray(left_reps, dtype=np.int64)
    valid_mask = np.isfinite(denom_arr) & (denom_arr > 0)
    ratios = distances[valid_mask] / denom_arr[valid_mask]
    ratio_left_reps_arr = left_reps_arr[valid_mask]
    return distances, ratios, left_reps_arr, ratio_left_reps_arr


def _calc_agg_fid(run_entries_by_label, real_pool, device, prefix=""):
    combined_agg = {"fid": None}

    label_order = sorted(run_entries_by_label.keys())
    all_gen_tensors = []

    for label in label_order:
        samples_list = [entry["samples"] for entry in run_entries_by_label[label]]
        if not samples_list:
            continue
        all_gen_tensors.append(th.cat(samples_list, dim=0))

    if all_gen_tensors:
        all_gen_tensor = th.cat(all_gen_tensors, dim=0)
        all_real_tensor = th.cat([real_pool[label] for label in label_order], dim=0)
        fid_comb = fid_torch_fidelity(all_real_tensor, all_gen_tensor, device=str(device))
        combined_agg = {"fid": fid_comb}
        print(f"  [{prefix}] Combined: FID={fid_comb:.3f}" if fid_comb is not None else f"  [{prefix}] Combined: FID=None")

    return combined_agg


class NoneDict(defaultdict):
    def __init__(self, **kwargs):
        super().__init__(self.return_none)
        self.update(**kwargs)

    @staticmethod
    def return_none():
        return None

    def __getattr__(self, attr):
        return self.get(attr)


def process_configs(
    model_and_diffusion_config,
    task_config,
    sampler_config,
    resample_strategy,
    sampler=None,
    P=None,
    T=None,
    var_type=None,
    ess_threshold=None,
    t_truncate_percent=None,
    classifier_guidance_scale=None,
    R=None,
):
    model_and_diffusion_config_defaults = model_and_diffusion_defaults()
    model_and_diffusion_config_defaults.update(yamlread(model_and_diffusion_config))
    model_and_diffusion_config = model_and_diffusion_config_defaults

    task_config = yamlread(task_config)

    sampler_config_defaults = sampler_defaults()
    sampler_config_defaults.update(yamlread(sampler_config))
    sampler_config = sampler_config_defaults

    if resample_strategy is not None:
        sampler_config.update({"resample_strategy": resample_strategy})
    if sampler is not None:
        sampler_config.update({"name": sampler})
    if T is not None:
        model_and_diffusion_config.update({"timestep_respacing": T})
    if P is not None:
        sampler_config.update(num_particles=P)
    if ess_threshold is not None:
        sampler_config.update(ess_threshold=ess_threshold)
    if t_truncate_percent is not None:
        sampler_config.update(t_truncate_percent=t_truncate_percent)
    if R is not None:
        sampler_config.update(R=R)

    if sampler_config["ess_threshold"] is None:
        sampler_config.update(ess_threshold=1.0)

    if sampler_config.get("R") is None:
        sampler_config.update(R=1)

    sampler_config["batch_p"] = sampler_config["num_particles"]

    if classifier_guidance_scale is not None:
        task_config["classifier_guidance_scale"] = classifier_guidance_scale

    if var_type is not None:
        task_config.update(pred_xstart_var_type=float(var_type))

    model_and_diffusion_config.update({"sampler": sampler_config["name"]})

    return model_and_diffusion_config, task_config, sampler_config


def main(
    model_and_diffusion_config,
    task_config,
    sampler_config,
    output_dir="./outputs",
    resample_strategy=None,
    sampler="tds",
    P=None,
    T=None,
    var_type=6,
    ess_threshold=None,
    t_truncate_percent=None,
    classifier_guidance_scale=1.0,
    class_batch_size=100,
    class_label=None,
    dataset_label=None,
    dataset_offset=0,
    dataset_size=None,
    seed=42,
    debug_plot=False,
    debug_statistics=False,
    save_output_i=100,
    debug_first_i=0,
    debug_plot_interval=10,
    debug_plot_start=-1,
    debug_plot_end=-1,
    R=1,
    compute_fid=False,
    compute_lpips=False,
):
    requested_class_label = class_label

    if sampler == "replacement":
        ess_threshold = 0
        sampler_is_smc_method = False
    else:
        sampler_is_smc_method = True

    model_and_diffusion_config, task_config, sampler_config = process_configs(
        model_and_diffusion_config,
        task_config,
        sampler_config,
        resample_strategy,
        sampler,
        P,
        T,
        var_type,
        ess_threshold,
        t_truncate_percent,
        classifier_guidance_scale,
        R=R,
    )

    if task_config.get("image_size") == 32:
        output_dir = "./outputs_cifar10"
    elif task_config.get("image_size") == 28:
        output_dir = "./outputs_mnist"

    num_replications = int(class_batch_size)

    sampler_name = sampler_config["name"]
    T = model_and_diffusion_config["timestep_respacing"]
    P = sampler_config["num_particles"]
    R_val = sampler_config.get("R", 1)
    classifier_guidance_scale = task_config["classifier_guidance_scale"]
    sampler_full_name = (
        f"{sampler_name}_T{T}_P{P}_R{R_val}_"
        f"essthreshold{sampler_config['ess_threshold']}_"
        f"truncate{sampler_config['t_truncate_percent']}"
        f"/guidancescale{classifier_guidance_scale}_"
        f"{sampler_config['resample_strategy']}"
    )

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(
        output_dir,
        f"{task_config['name']}",
        f"{sampler_full_name}_data{dataset_label}-{dataset_offset}-{dataset_size}_{timestr}",
    )

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving to {output_path}", flush=True)

    output_config_file = os.path.join(output_path, "config.yml")
    config_dict = {
        "seed": seed,
        "class_batch_size": int(class_batch_size),
        "num_replications_per_label": num_replications,
        "model_and_diffusion_config": model_and_diffusion_config,
        "task_config": task_config,
        "sampler_config": sampler_config,
        "dataset_label": dataset_label,
        "dataset_offset": dataset_offset,
        "dataset_size": dataset_size,
        "output_path": output_path,
    }
    with open(output_config_file, "w") as handle:
        yaml.dump(config_dict, handle, default_flow_style=False)

    metrics_csv = os.path.join(output_path, "metrics_details.csv")

    model_and_diffusion_config = NoneDict(**model_and_diffusion_config)
    task_config = NoneDict(**task_config)
    sampler_config = NoneDict(**sampler_config)

    _set_all_seeds(int(seed))

    sample_path = os.path.join(output_path, "samples")
    os.makedirs(sample_path, exist_ok=True)

    device = dist_util.dev(None)
    lpips_fn = _prepare_lpips(device) if compute_lpips else None

    img_size = model_and_diffusion_config.image_size
    if img_size == 28:
        print("Using MNIST dataset and VAE.")
        real_pool = build_real_pool_mnist(per_class_count=2000) if compute_fid else None
        vae_path = "./models/checkpoints/vae_mnist_z32.pt"
    elif img_size == 32:
        print("Using CIFAR-10 dataset and VAE.")
        real_pool = build_real_pool_cifar10(per_class_count=2000) if compute_fid else None
        vae_path = "./models/checkpoints/vae_cifar10_z32.pt"
    else:
        print(f"Warning: Unknown image size {img_size}, defaulting to MNIST.")
        real_pool = build_real_pool_mnist(per_class_count=2000) if compute_fid else None
        vae_path = "./models/checkpoints/vae_mnist_z32.pt"

    model, diffusion = create_model_and_diffusion(
        **select_args(model_and_diffusion_config, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(
            os.path.expanduser(model_and_diffusion_config.model_path),
            map_location="cpu",
        )
    )
    model.to(device)

    if model_and_diffusion_config.use_fp16:
        model.convert_to_fp16()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    def model_fn(x, t, y=None, **kwargs):
        return model(x, t, y if model_and_diffusion_config.class_cond else None, **kwargs)

    print(f"Using class_batch_size={num_replications} as replications per label.")
    classifier = load_classifier(task_config["classifier_path"], device=device)
    classifier_prob_base = ConditioningMethod(operator=None).classifier_prob
    classifier_prob_base = partial(
        classifier_prob_base,
        classifier=classifier,
        guidance_scale=classifier_guidance_scale,
    )

    diffusion.task = task_config.name
    diffusion.use_mean_pred = True
    diffusion.t_truncate = int(diffusion.T * sampler_config["t_truncate_percent"])

    run_source = _build_class_cond_schedule(requested_class_label, num_replications)

    if debug_plot or debug_statistics:
        debug_path = os.path.join(output_path, "debug")
        os.makedirs(debug_path, exist_ok=True)
        assert debug_plot_end <= debug_plot_start
        debug_info_base = {
            "debug_path": debug_path,
            "plot_interval": debug_plot_interval,
            "plot_start": debug_plot_start,
            "plot_end": debug_plot_end,
        }
    else:
        debug_info_base = {}

    final_records = []
    truncate_records = []
    final_run_entries_by_label = defaultdict(list)
    truncate_run_entries_by_label = defaultdict(list)

    all_gen_samples_list = []
    all_gen_labels_list = []
    all_gen_replications_list = []
    all_truncate_samples_list = []
    all_truncate_labels_list = []
    all_truncate_replications_list = []

    def _record_class_cond_metrics(record_type, samples, target_label, replication_id, iter_idx):
        with th.no_grad():
            logits_all = classifier(samples.to(device))
            probs_all = th.softmax(logits_all, dim=1)
            preds_all = th.argmax(probs_all, dim=1)

        correct_mask = preds_all.cpu() == target_label
        num_particles = int(samples.size(0))
        num_correct = int(correct_mask.sum().item())
        acc_iter = float(num_correct / max(1, num_particles))
        lpips_mean = _lpips_stats(samples, lpips_fn)["mean"] if compute_lpips else None

        record = {
            "type": record_type,
            "iter": int(iter_idx),
            "label": int(target_label),
            "replication": int(replication_id),
            "n_particles": num_particles,
            "n_correct": num_correct,
            "acc": acc_iter,
            "lpips_within_mean": lpips_mean,
        }

        _append_row(
            metrics_csv,
            {
                "type": record["type"],
                "iter": record["iter"],
                "label": record["label"],
                "replication": record["replication"],
                "n_particles": record["n_particles"],
                "n_correct": record["n_correct"],
                "acc": record["acc"],
                "LPIPS_within_mean": record["lpips_within_mean"],
            },
        )

        lpips_str = f"{lpips_mean:.3f}" if lpips_mean is not None else "n/a"
        print(
            f"[METRICS] type={record_type} iter={iter_idx} label={target_label} "
            f"rep={replication_id} acc={acc_iter:.3f} "
            f"LPIPS(within mean)={lpips_str}"
        )

        samples_cpu = samples.detach().cpu()
        entry = {"record": record, "samples": samples_cpu}

        labels_cpu = th.full((samples_cpu.size(0),), int(target_label), dtype=th.long)
        replications_cpu = th.full((samples_cpu.size(0),), int(replication_id), dtype=th.long)

        if record_type == "final":
            final_records.append(record)
            final_run_entries_by_label[target_label].append(entry)
            all_gen_samples_list.append(samples_cpu)
            all_gen_labels_list.append(labels_cpu)
            all_gen_replications_list.append(replications_cpu)
        else:
            truncate_records.append(record)
            truncate_run_entries_by_label[target_label].append(entry)
            all_truncate_samples_list.append(samples_cpu)
            all_truncate_labels_list.append(labels_cpu)
            all_truncate_replications_list.append(replications_cpu)

    for i, data_i in enumerate(run_source):
        target_label, replication_id = data_i
        target_label = int(target_label)
        replication_id = int(replication_id)
        run_seed = int(seed) + 100000 * target_label + replication_id
        ref_img_name = f"label{target_label}_rep{replication_id}_iter{i}"

        _set_all_seeds(run_seed)

        current_debug_plot = bool(debug_plot and i <= debug_first_i)
        current_debug_statistics = bool(debug_statistics and i <= debug_first_i)
        debug_info = dict(debug_info_base)
        debug_info.update({"img_name": ref_img_name})

        model_kwargs = {}
        classes = th.randint(low=0, high=NUM_CLASSES, size=(), device=device)
        model_kwargs["y"] = classes

        classifier_prob = partial(classifier_prob_base, measurement=target_label)
        diffusion.classifier_prob_fn = classifier_prob

        diffusion.clear_cache()

        M = partial(
            diffusion.M,
            model=model_fn,
            device=device,
            pred_xstart_var_type=task_config.pred_xstart_var_type,
        )
        G = partial(
            diffusion.G,
            model=model_fn,
            debug_plot=current_debug_plot,
            debug_statistics=current_debug_statistics,
            debug_info=debug_info,
            pred_xstart_var_type=task_config.pred_xstart_var_type,
        )

        final_sample, log_w, normalized_w, resample_indices_trace, ess_trace, log_w_trace, xt_trace = smc_FK(
            M=M,
            G=G,
            resample_strategy=sampler_config.resample_strategy,
            ess_threshold=sampler_config.ess_threshold,
            T=diffusion.T,
            P=sampler_config.num_particles,
            verbose=True,
            log_xt_trace=False,
            extra_vals={"model_kwargs": model_kwargs, "batch_p": sampler_config.batch_p},
            vae_path=vae_path,
            R=sampler_config.R,
        )

        truncate_sample = None
        truncate_log_w = None
        if diffusion.t_truncate > 1:
            truncate_sample = diffusion.xpred_at_t_truncate
            truncate_log_w = diffusion.log_w_x0_truncate

            if sampler_config.ess_threshold == 0:
                log_w_Ttotp1 = log_w_trace[-(diffusion.t_truncate + 2)]
                log_w_Ttotp1 = log_w_Ttotp1.to(truncate_log_w.device)
                assert log_w_Ttotp1.shape == truncate_log_w.shape, f"{log_w_Ttotp1.shape}, {truncate_log_w.shape}"
                truncate_log_w += log_w_Ttotp1

        nrow = int(np.ceil(np.sqrt(sampler_config.num_particles)))
        if i < save_output_i:
            print("\nSaving!", flush=True)

            gridded_image = make_grid(final_sample, nrow)
            imwrite(
                path=os.path.join(sample_path, f"sample_{ref_img_name}.png"),
                img=toU8(gridded_image),
            )

        _record_class_cond_metrics("final", final_sample, target_label, replication_id, i)

        if diffusion.t_truncate > 1:
            if i < save_output_i:
                truncate_grid = make_grid(truncate_sample, nrow)
                imwrite(
                    path=os.path.join(sample_path, f"sample_truncate_{ref_img_name}.png"),
                    img=toU8(truncate_grid),
                )

            _record_class_cond_metrics("truncate", truncate_sample, target_label, replication_id, i)

    print("\nCalculating Aggregated Metrics...")

    def _between_per_rep(run_entries_by_label, seed_offset):
        # Pool every per-class sample, tagged by left's rep, then average per rep.
        lpips_by_rep = defaultdict(list)
        ratio_by_rep = defaultdict(list)
        for label in sorted(run_entries_by_label.keys()):
            distances, ratios, left_reps, ratio_left_reps = _sample_between_run_values(
                run_entries_by_label[label],
                lpips_fn,
                seed=int(seed) + seed_offset + int(label),
            )
            for d, r in zip(distances, left_reps):
                lpips_by_rep[int(r)].append(float(d))
            for v, r in zip(ratios, ratio_left_reps):
                ratio_by_rep[int(r)].append(float(v))
        lpips_per_rep = [float(np.mean(v)) for v in lpips_by_rep.values() if v]
        ratio_per_rep = [float(np.mean(v)) for v in ratio_by_rep.values() if v]
        return lpips_per_rep, ratio_per_rep

    final_between_lpips_per_rep, final_between_ratio_per_rep = _between_per_rep(
        final_run_entries_by_label, seed_offset=1000000
    )

    if compute_fid:
        combined_agg_final = _calc_agg_fid(
            final_run_entries_by_label,
            real_pool,
            device,
            prefix="Final",
        )
    else:
        combined_agg_final = None

    combined_agg_truncate = None
    overall_truncate = None
    if diffusion.t_truncate > 1:
        truncate_between_lpips_per_rep, truncate_between_ratio_per_rep = _between_per_rep(
            truncate_run_entries_by_label, seed_offset=2000000
        )

        overall_truncate = _summarize_run_records(
            truncate_records,
            between_lpips_per_rep=truncate_between_lpips_per_rep,
            between_ratio_per_rep=truncate_between_ratio_per_rep,
        )
        if compute_fid:
            combined_agg_truncate = _calc_agg_fid(
                truncate_run_entries_by_label,
                real_pool,
                device,
                prefix="Truncate",
            )

    overall_final = _summarize_run_records(
        final_records,
        between_lpips_per_rep=final_between_lpips_per_rep,
        between_ratio_per_rep=final_between_ratio_per_rep,
    )

    summary = {
        "num_replications_per_label": num_replications,
        "between_pairs_per_label": BETWEEN_RUN_PAIR_COUNT,
        "overall_final": overall_final,
        "combined_aggregated_final": combined_agg_final,
        "overall_truncate": overall_truncate,
        "combined_aggregated_truncate": combined_agg_truncate,
    }

    summary_path = os.path.join(output_path, "metrics_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("[SUMMARY]", json.dumps(summary["overall_final"], indent=2))

    if all_gen_samples_list:
        all_samples_path = os.path.join(output_path, "all_samples.npz")
        print(f"Saving all generated samples to {all_samples_path}...")
        np.savez_compressed(
            all_samples_path,
            samples=th.cat(all_gen_samples_list, dim=0).numpy(),
            labels=th.cat(all_gen_labels_list, dim=0).numpy(),
            replications=th.cat(all_gen_replications_list, dim=0).numpy(),
        )

    if all_truncate_samples_list:
        all_truncate_samples_path = os.path.join(output_path, "all_truncate_samples.npz")
        print(f"Saving all truncate samples to {all_truncate_samples_path}...")
        np.savez_compressed(
            all_truncate_samples_path,
            samples=th.cat(all_truncate_samples_list, dim=0).numpy(),
            labels=th.cat(all_truncate_labels_list, dim=0).numpy(),
            replications=th.cat(all_truncate_replications_list, dim=0).numpy(),
        )


if __name__ == "__main__":
    fire.Fire(main)
