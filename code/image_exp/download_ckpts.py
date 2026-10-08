#!/usr/bin/env python3
"""Download and verify large checkpoints for the image experiments.

Files (placed in ./models/):
    uncond_model_mnist.pt        ~55 MB    MNIST unconditional diffusion model
    uncond_model_cifar10.pt      ~201 MB   CIFAR-10 unconditional diffusion model
    classifier_mnist.pth.tar     ~180 MB   MNIST classifier
    classifier_cifar10.pth.tar   ~180 MB   CIFAR-10 classifier

Usage:
    python download_ckpts.py
    python download_ckpts.py --verify    # only verify existing files, no download
    python download_ckpts.py --no-verify # skip SHA256 check after download
    HF_REPO_ID=other/repo python download_ckpts.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


DEFAULT_REPO_ID = "Mengyu8042/wdr-checkpoints"
DEFAULT_REVISION = "main"

# Each entry: (HF filename, local relative path, size in bytes, SHA256 hex digest).
CKPTS = [
    ("uncond_model_mnist.pt",      "models/uncond_model_mnist.pt",
     57883985,  "941455dab870142ea9eb3fb1eeddc1dec00e6355eca2adb124f658f96ac497ae"),
    ("uncond_model_cifar10.pt",    "models/uncond_model_cifar10.pt",
     210292511, "351ef8b22e54e0eb2f8ef70389e101ef67e313f8fda41b46139575b3ce042cec"),
    ("classifier_mnist.pth.tar",   "models/classifier_mnist.pth.tar",
     188537249, "d52735e768fa7ec2bc98575471468145f369ae869793048d3992b194d76a99b9"),
    ("classifier_cifar10.pth.tar", "models/classifier_cifar10.pth.tar",
     188546465, "38dfe2c536c640df3aee3a8f8b49973d0b34f4827d44ed0f1ca6f52029b1d9e2"),
]


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_already_good(target: Path, expected_size: int, expected_sha: str, do_sha: bool) -> bool:
    if not target.exists():
        return False
    if target.stat().st_size != expected_size:
        return False
    if not do_sha:
        return True
    return sha256_of_file(target) == expected_sha


def download_via_hf_hub(repo_id, revision, here, do_sha):
    from huggingface_hub import hf_hub_download
    for hf_name, local_rel, expected_size, expected_sha in CKPTS:
        target = here / local_rel
        if file_already_good(target, expected_size, expected_sha, do_sha=False):
            print(f"[skip] {target.name} already present ({target.stat().st_size / 1e6:.1f} MB)")
            if do_sha:
                actual = sha256_of_file(target)
                if actual != expected_sha:
                    raise RuntimeError(
                        f"SHA256 mismatch for {target}: got {actual}, expected {expected_sha}"
                    )
                print(f"       SHA256 verified")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {hf_name} -> {target}")
        cached = hf_hub_download(
            repo_id=repo_id, filename=hf_name, revision=revision, repo_type="dataset",
            local_dir=str(target.parent),
        )
        if Path(cached).resolve() != target.resolve():
            os.replace(cached, target)
        if do_sha:
            actual = sha256_of_file(target)
            if actual != expected_sha:
                raise RuntimeError(
                    f"SHA256 mismatch for {target}: got {actual}, expected {expected_sha}"
                )
            print(f"           SHA256 verified")


def download_via_requests(url_base, here, do_sha):
    import requests
    for hf_name, local_rel, expected_size, expected_sha in CKPTS:
        target = here / local_rel
        if file_already_good(target, expected_size, expected_sha, do_sha=False):
            print(f"[skip] {target.name}")
            if do_sha:
                actual = sha256_of_file(target)
                if actual != expected_sha:
                    raise RuntimeError(
                        f"SHA256 mismatch for {target}: got {actual}, expected {expected_sha}"
                    )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{url_base.rstrip('/')}/resolve/main/{hf_name}"
        print(f"[download] {url} -> {target}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        if do_sha:
            actual = sha256_of_file(target)
            if actual != expected_sha:
                raise RuntimeError(
                    f"SHA256 mismatch for {target}: got {actual}, expected {expected_sha}"
                )


def verify_only(here):
    """Check existing files against the manifest; do not download."""
    all_good = True
    for _hf_name, local_rel, expected_size, expected_sha in CKPTS:
        target = here / local_rel
        if not target.exists():
            print(f"[missing] {target}")
            all_good = False
            continue
        size = target.stat().st_size
        if size != expected_size:
            print(f"[size mismatch] {target}: got {size}, expected {expected_size}")
            all_good = False
            continue
        actual = sha256_of_file(target)
        if actual != expected_sha:
            print(f"[sha mismatch] {target}: got {actual}, expected {expected_sha}")
            all_good = False
            continue
        print(f"[ok] {target.name} ({size / 1e6:.1f} MB)")
    return all_good


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID))
    parser.add_argument("--revision", default=os.environ.get("HF_REVISION", DEFAULT_REVISION))
    parser.add_argument("--verify", action="store_true",
                        help="Verify already-downloaded files against the manifest and exit.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip SHA256 verification after download.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    if args.verify:
        ok = verify_only(here)
        sys.exit(0 if ok else 1)

    do_sha = not args.no_verify
    print(f"Downloading from HuggingFace dataset repo: {args.repo_id}@{args.revision}")
    print(f"Target directory: {here / 'models'}")
    print(f"SHA256 verification: {'on' if do_sha else 'off'}")

    try:
        download_via_hf_hub(args.repo_id, args.revision, here, do_sha)
    except ImportError:
        print("huggingface_hub not installed; falling back to direct HTTP.", file=sys.stderr)
        download_via_requests(
            f"https://huggingface.co/datasets/{args.repo_id}", here, do_sha
        )

    print("Done. Checkpoints ready in ./models/.")


if __name__ == "__main__":
    main()
