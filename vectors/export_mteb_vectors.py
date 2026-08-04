#!/usr/bin/env python3
"""Run one pinned MTEB task and save every query and document embedding."""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mteb-root", type=Path, required=True)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("mteb_config.json")
    )
    return parser.parse_args()


def read_config(path: Path, task: str, model_name: str) -> tuple[dict, dict]:
    """Load the pinned task and model settings."""
    config = json.loads(path.read_text(encoding="utf-8"))
    allowed_tasks = [*config["datasets"], *config.get("external_datasets", [])]
    if task not in allowed_tasks:
        raise ValueError(f"task is not part of the paper workload: {task}")
    matches = [model for model in config["models"] if model["name"] == model_name]
    if len(matches) != 1:
        raise ValueError(f"model is not part of the paper workload: {model_name}")
    return config, matches[0]


def model_slug(model_name: str) -> str:
    """Convert a model name to its vector-directory name."""
    return model_name.replace("/", "__")


def check_mteb_source(root: Path, expected_commit: str) -> None:
    """Require the configured upstream MTEB source revision."""
    if not (root / "pyproject.toml").exists():
        raise FileNotFoundError(f"MTEB source is missing: {root}")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    if completed.stdout.strip() != expected_commit:
        raise RuntimeError("MTEB source revision does not match mteb_config.json")


def configure_cache(cache_root: Path) -> None:
    """Set explicit cache locations without changing the user's home directory."""
    cache_root.mkdir(parents=True, exist_ok=True)
    huggingface = cache_root / "huggingface"
    values = {
        "HF_HOME": huggingface,
        "HF_HUB_CACHE": huggingface / "hub",
        "HF_DATASETS_CACHE": cache_root / "datasets",
        "HF_MODULES_CACHE": huggingface / "modules",
        "SENTENCE_TRANSFORMERS_HOME": cache_root / "sentence_transformers",
        "TORCH_HOME": cache_root / "torch",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "XDG_CACHE_HOME": cache_root,
    }
    for path in values.values():
        path.mkdir(parents=True, exist_ok=True)
    for key, path in values.items():
        os.environ[key] = str(path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    """Convert common scalar containers to JSON-compatible values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return str(value)


def as_numpy(vectors: Any) -> np.ndarray:
    """Move an embedding batch to a NumPy array."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(vectors, torch.Tensor):
        return vectors.detach().cpu().numpy()
    return np.asarray(vectors)


class EmbeddingWriter:
    """Wrap an MTEB encoder and save its aligned inputs and vectors."""

    def __init__(self, model: Any, run_root: Path, expected_dim: int) -> None:
        """Create a strict embedding writer for one model run."""
        if not hasattr(model, "encode"):
            raise TypeError("the selected MTEB model is not an encoder")
        self._model = model
        self.run_root = run_root
        self.expected_dim = expected_dim
        self.manifest = run_root / "manifest.jsonl"
        self.counters: dict[tuple[str, str, str, str], int] = {}

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped MTEB model."""
        return getattr(self._model, name)

    @property
    def mteb_model_meta(self) -> Any:
        """Return the wrapped model metadata."""
        return self._model.mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, value: Any) -> None:
        """Update the wrapped model metadata."""
        self._model.mteb_model_meta = value

    def similarity(self, embeddings1: Any, embeddings2: Any) -> Any:
        """Compute pairwise similarities with the wrapped model."""
        return self._model.similarity(embeddings1, embeddings2)

    def similarity_pairwise(self, embeddings1: Any, embeddings2: Any) -> Any:
        """Compute aligned similarities with the wrapped model."""
        return self._model.similarity_pairwise(embeddings1, embeddings2)

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Any:
        """Encode one MTEB input collection and save the aligned output."""
        vectors = self._model.encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            batch_size=batch_size,
            **kwargs,
        )
        self.write_batch(
            inputs, vectors, task_metadata, hf_split, hf_subset, prompt_type
        )
        return vectors

    def write_batch(
        self,
        inputs: Any,
        vectors: Any,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any,
    ) -> None:
        """Write one vector batch, its records, metadata, and hashes."""
        array = as_numpy(vectors)
        if array.ndim != 2 or array.shape[1] != self.expected_dim:
            raise RuntimeError(f"unexpected embedding shape: {array.shape}")
        dataset = getattr(inputs, "dataset", None)
        if dataset is None:
            raise RuntimeError("MTEB input records are unavailable")
        task_name = str(getattr(task_metadata, "name", "unknown"))
        prompt_name = str(prompt_type) if prompt_type is not None else "none"
        key = (task_name, str(hf_split), str(hf_subset), prompt_name)
        batch_index = self.counters.get(key, 0)
        self.counters[key] = batch_index + 1
        batch_root = (
            self.run_root
            / task_name
            / f"split={hf_split}"
            / f"subset={hf_subset}"
            / f"prompt_type={prompt_name}"
        )
        batch_root.mkdir(parents=True, exist_ok=True)
        stem = f"batch_{batch_index:05d}"
        vector_path = batch_root / f"{stem}.npy"
        record_path = batch_root / f"{stem}.jsonl"
        meta_path = batch_root / f"{stem}.meta.json"
        np.save(vector_path, array, allow_pickle=False)
        count = 0
        with record_path.open("w", encoding="utf-8") as handle:
            for row_index, item in enumerate(dataset):
                row = {"row_index": row_index}
                if isinstance(item, dict):
                    row.update(json_value(item))
                else:
                    row["item"] = json_value(item)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        if count != len(array):
            raise RuntimeError("MTEB record and vector counts do not match")
        entry = {
            "task_name": task_name,
            "hf_split": str(hf_split),
            "hf_subset": str(hf_subset),
            "prompt_type": prompt_name,
            "batch_index": batch_index,
            "num_vectors": len(array),
            "embedding_dim": int(array.shape[1]),
            "dtype": str(array.dtype),
            "record_count": count,
            "vectors_path": str(vector_path.relative_to(self.run_root)),
            "records_path": str(record_path.relative_to(self.run_root)),
            "vectors_sha256": sha256_file(vector_path),
            "records_sha256": sha256_file(record_path),
        }
        meta_path.write_text(
            json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with self.manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def enable_embedding_writer(run_root: Path, expected_dim: int) -> None:
    """Wrap every loaded MTEB encoder with the strict embedding writer."""
    from mteb.models.model_meta import ModelMeta

    original = ModelMeta.load_model

    @functools.wraps(original)
    def patched(model_meta: Any, *args: Any, **kwargs: Any) -> Any:
        """Load one model and attach the embedding writer."""
        return EmbeddingWriter(
            original(model_meta, *args, **kwargs), run_root, expected_dim
        )

    ModelMeta.load_model = patched


def enable_mistral_dtype_compat() -> None:
    """Map legacy SentenceTransformers dtype arguments to torch_dtype."""
    from sentence_transformers.models import Transformer

    original = Transformer._load_model

    @functools.wraps(original)
    def patched(
        instance: Any,
        model_name_or_path: str,
        config: Any,
        cache_dir: str,
        backend: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Load a transformer after renaming its legacy dtype argument."""
        dtype = kwargs.pop("dtype", None)
        if dtype is not None:
            kwargs.setdefault("torch_dtype", dtype)
        return original(
            instance, model_name_or_path, config, cache_dir, backend, *args, **kwargs
        )

    Transformer._load_model = patched


def set_codefuse_attention(implementation: str) -> None:
    """Use the recorded attention implementation for F2LLM-v2-8B."""
    from mteb.models.model_implementations import codefuse_models

    for name in dir(codefuse_models):
        meta = getattr(codefuse_models, name)
        if getattr(meta, "name", None) != "codefuse-ai/F2LLM-v2-8B":
            continue
        model_kwargs = meta.loader_kwargs.setdefault("model_kwargs", {})
        model_kwargs["attn_implementation"] = implementation
        return
    raise RuntimeError("F2LLM-v2-8B metadata is missing from the pinned MTEB source")


def enable_geoembedding_compat() -> None:
    """Restore the Mistral interfaces required by GeoEmbedding."""
    from transformers.models.mistral import modeling_mistral

    if not hasattr(modeling_mistral.MistralModel, "_update_causal_mask"):

        def update_causal_mask(
            instance: Any,
            attention_mask: Any,
            input_embeds: Any,
            cache_position: Any,
            past_key_values: Any,
            output_attentions: bool = False,
        ) -> Any:
            """Build the causal mask expected by the GeoEmbedding model code."""
            del output_attentions
            position_ids = (
                cache_position.unsqueeze(0) if cache_position is not None else None
            )
            mask_function = (
                modeling_mistral.create_causal_mask
                if instance.config.sliding_window is None
                else modeling_mistral.create_sliding_window_causal_mask
            )
            return mask_function(
                config=instance.config,
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        modeling_mistral.MistralModel._update_causal_mask = update_causal_mask
    original = modeling_mistral.apply_rotary_pos_emb

    @functools.wraps(original)
    def apply_rotary(
        query: Any,
        key: Any,
        cosine: Any,
        sine: Any,
        position_ids: Any = None,
        unsqueeze_dim: int = 1,
    ) -> Any:
        """Retry GeoEmbedding rotary positions with its expected tensor axis."""
        try:
            return original(
                query,
                key,
                cosine,
                sine,
                position_ids=position_ids,
                unsqueeze_dim=unsqueeze_dim,
            )
        except RuntimeError as error:
            if "must match the size of tensor b" not in str(error):
                raise
            cosine = cosine[..., : query.shape[-1]]
            sine = sine[..., : query.shape[-1]]
            return original(
                query,
                key,
                cosine,
                sine,
                position_ids=position_ids,
                unsqueeze_dim=2,
            )

    modeling_mistral.apply_rotary_pos_emb = apply_rotary


def pinned_snapshot(component: dict[str, str]) -> Path:
    """Download and verify one Hugging Face model snapshot by commit."""
    from huggingface_hub import snapshot_download

    revision = str(component["revision"])
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("model component revisions must be full commit hashes")
    path = Path(
        snapshot_download(
            repo_id=str(component["repo_id"]),
            revision=revision,
            cache_dir=os.environ["HF_HUB_CACHE"],
        )
    ).resolve()
    if path.name != revision:
        raise RuntimeError(f"snapshot revision mismatch for {component['repo_id']}")
    return path


def prepare_llm2vec_components(
    model: dict, cache_root: Path
) -> tuple[Path, Path, dict]:
    """Prepare a fully local LLM2Vec adapter chain from pinned snapshots."""
    components = model.get("components")
    if not isinstance(components, dict) or set(components) != {
        "backbone",
        "mntp",
        "supervised",
    }:
        raise ValueError("LLM2Vec requires backbone, MNTP, and supervised components")
    if (
        components["supervised"]["repo_id"] != model["name"]
        or components["supervised"]["revision"] != model["revision"]
    ):
        raise ValueError("the supervised LLM2Vec component must match the model lock")
    snapshots = {
        name: pinned_snapshot(component) for name, component in components.items()
    }
    identity = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    prepared_root = cache_root / "llm2vec" / identity
    prepared_mntp = prepared_root / "mntp"
    metadata_path = prepared_root / "components.json"
    adapter_path = prepared_mntp / "adapter_config.json"
    expected_metadata = {"components": components, "backbone_path": str(snapshots["backbone"])}
    valid = False
    if metadata_path.is_file() and adapter_path.is_file():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        valid = (
            recorded == expected_metadata
            and adapter.get("base_model_name_or_path") == str(snapshots["backbone"])
        )
    if not valid:
        if prepared_root.exists():
            shutil.rmtree(prepared_root)
        prepared_root.mkdir(parents=True)
        shutil.copytree(snapshots["mntp"], prepared_mntp)
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        adapter["base_model_name_or_path"] = str(snapshots["backbone"])
        adapter_path.write_text(
            json.dumps(adapter, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return prepared_mntp, snapshots["supervised"], components


def enable_llm2vec_compat(model: dict, cache_root: Path) -> dict:
    """Replace the MTEB loader with the pinned local LLM2Vec chain."""
    import torch
    from mteb.models.model_implementations import llm2vec_models

    prepared_mntp, supervised, components = prepare_llm2vec_components(
        model, cache_root
    )
    for attribute in dir(llm2vec_models):
        meta = getattr(llm2vec_models, attribute)
        if getattr(meta, "name", None) != model["name"]:
            continue
        original = meta.loader

        @functools.wraps(original)
        def patched_loader(*args: Any, **kwargs: Any) -> Any:
            """Load the pinned backbone and both adapters from local snapshots."""
            if args:
                raise TypeError("the pinned LLM2Vec loader accepts keyword arguments")
            kwargs.pop("revision", None)
            kwargs.setdefault("device_map", "auto")
            kwargs.setdefault("torch_dtype", torch.bfloat16)
            return llm2vec_models.LLM2VecModel(
                base_model_name_or_path=str(prepared_mntp),
                peft_model_name_or_path=str(supervised),
                **kwargs,
            )

        meta.loader = patched_loader
        return components
    raise RuntimeError("LLM2Vec metadata is missing from the pinned MTEB source")


def package_versions(runtime: str) -> dict[str, str]:
    """Return versions of the packages that affect vector generation."""
    packages = [
        "mteb",
        "torch",
        "sentence-transformers",
        "transformers",
        "datasets",
        "accelerate",
        "huggingface-hub",
    ]
    if runtime == "llm2vec":
        packages.append("llm2vec")
    return {name: metadata.version(name) for name in packages}


def main() -> None:
    """Validate the runtime, run MTEB, and save one vector manifest."""
    args = parse_args()
    config, model = read_config(args.config, args.task, args.model)
    args.mteb_root = args.mteb_root.expanduser().resolve()
    args.vector_root = args.vector_root.expanduser().resolve()
    args.cache_root = args.cache_root.expanduser().resolve()
    check_mteb_source(args.mteb_root, config["mteb_commit"])
    configure_cache(args.cache_root)
    sys.path.insert(0, str(args.mteb_root))
    versions = package_versions(str(model["runtime"]))
    if versions["mteb"] != config["mteb_version"]:
        raise RuntimeError("installed MTEB version does not match mteb_config.json")
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = (
        args.vector_root
        / "datasets"
        / args.task
        / "embeddings"
        / model_slug(args.model)
        / model["revision"]
        / run_id
    )
    if run_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"vector run already exists: {run_root}")
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True)
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = model.get("batch_size_by_task", {}).get(
            args.task, model.get("batch_size")
        )
    if args.model in {
        "Salesforce/SFR-Embedding-Mistral",
        "intfloat/e5-mistral-7b-instruct",
    }:
        enable_mistral_dtype_compat()
    if model.get("attention_implementation"):
        set_codefuse_attention(model["attention_implementation"])
    if args.model == "GeoGPT-Research-Project/GeoEmbedding":
        enable_geoembedding_compat()
    model_components = None
    if model["runtime"] == "llm2vec":
        model_components = enable_llm2vec_compat(model, args.cache_root)
    enable_embedding_writer(run_root, int(model["embedding_dim"]))
    run_meta = {
        "task": args.task,
        "model": args.model,
        "model_revision": model["revision"],
        "mteb_commit": config["mteb_commit"],
        "batch_size": batch_size,
        "packages": versions,
    }
    if model_components is not None:
        run_meta["model_components"] = model_components
    (run_root / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result_root = args.vector_root / "datasets" / args.task / "mteb_results"
    command = [
        "mteb",
        "run",
        "-m",
        args.model,
        "-t",
        args.task,
        "--model-revision",
        model["revision"],
        "--output-folder",
        str(result_root),
        "--overwrite-strategy",
        "always",
        "--device",
        args.device,
    ]
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    from mteb.cli import main as mteb_main

    sys.argv = command
    mteb_main()
    if not (run_root / "manifest.jsonl").exists():
        raise RuntimeError("MTEB completed without exporting vectors")
    print(run_root)


if __name__ == "__main__":
    main()
