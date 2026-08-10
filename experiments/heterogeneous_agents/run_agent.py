#!/usr/bin/env python3
"""Model-agnostic, resumable runner for one heterogeneous agent.

Unlike ``run_inference.py``, this runner never emits raw Qwen ChatML.  It uses
the selected checkpoint's own chat template, which is required for a fair
Gemma/Llama comparison.  Choice tasks are teacher-forced continuation scores;
generation tasks retain every sample verbatim for later parsing.
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from experiments.heterogeneous_agents.core import (
    ContractError, load_agent_config, read_jsonl, sha256, softmax,
    validate_task_response, write_jsonl_atomic,
)


def _agent(config: Mapping[str, Any], agent_id: str) -> dict:
    matches = [agent for agent in config["agents"] if agent["id"] == agent_id]
    if len(matches) != 1:
        raise ContractError(f"unknown agent id {agent_id!r}")
    return matches[0]


def validate_tasks(tasks: List[dict], agent_id: str) -> Dict[str, dict]:
    by_id = {}
    for index, task in enumerate(tasks):
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ContractError(f"task {index}: invalid task_id")
        if task_id in by_id:
            raise ContractError(f"duplicate task_id: {task_id}")
        if task.get("agent_id") != agent_id:
            raise ContractError(f"{task_id}: belongs to {task.get('agent_id')}, not {agent_id}")
        if task.get("mode") not in {
                "generate", "choice", "constant", "representation"}:
            raise ContractError(f"{task_id}: invalid mode")
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise ContractError(f"{task_id}: empty prompt")
        if task["mode"] == "generate":
            if not isinstance(task.get("n_samples"), int) or task["n_samples"] < 1:
                raise ContractError(f"{task_id}: invalid n_samples")
        elif task["mode"] in {"choice", "constant"}:
            choices = task.get("choices")
            if (not isinstance(choices, list) or not choices
                    or any(not isinstance(choice, str) or not choice for choice in choices)):
                raise ContractError(f"{task_id}: invalid choices")
            if (task["mode"] == "constant"
                    and task.get("constant_choice") not in choices):
                raise ContractError(f"{task_id}: invalid constant_choice")
            if task["mode"] == "choice":
                codes = task.get("choice_codes")
                if (not isinstance(codes, dict) or set(codes) != set(choices)
                        or any(not isinstance(code, str) or not code for code in codes.values())
                        or len(set(codes.values())) != len(choices)):
                    raise ContractError(f"{task_id}: invalid choice_codes")
                variants = task.get("choice_variants")
                if (not isinstance(variants, list)
                        or not variants
                        or len(variants) % len(choices) != 0):
                    raise ContractError(f"{task_id}: invalid choice_variants")
                seen_by_choice = {choice: set() for choice in choices}
                counts_by_choice = {
                    choice: {} for choice in choices}
                for variant in variants:
                    variant_codes = variant.get("choice_codes") if isinstance(variant, dict) else None
                    if (not isinstance(variant, dict)
                            or not isinstance(variant.get("prompt"), str)
                            or not variant["prompt"].strip()
                            or not isinstance(variant_codes, dict)
                            or set(variant_codes) != set(choices)
                            or len(set(variant_codes.values())) != len(choices)):
                        raise ContractError(f"{task_id}: malformed choice variant")
                    for choice, code in variant_codes.items():
                        seen_by_choice[choice].add(code)
                        counts = counts_by_choice[choice]
                        counts[code] = counts.get(code, 0) + 1
                expected_codes = set(codes.values())
                if any(values != expected_codes for values in seen_by_choice.values()):
                    raise ContractError(f"{task_id}: choice variants are not code-balanced")
                expected_count = len(variants) // len(choices)
                if any(
                        set(counts) != expected_codes
                        or any(value != expected_count for value in counts.values())
                        for counts in counts_by_choice.values()):
                    raise ContractError(
                        f"{task_id}: choice variant code counts are not balanced")
        else:
            fractions = task.get("representation_layer_fractions")
            dimension = task.get("representation_projection_dim")
            namespace = task.get("representation_projection_namespace")
            if (
                not isinstance(fractions, list)
                or not fractions
                or any(
                    not isinstance(value, (int, float))
                    or not 0.0 < float(value) <= 1.0
                    for value in fractions)
                or sorted(map(float, fractions)) != list(map(float, fractions))
                or len(set(map(float, fractions))) != len(fractions)
                or not isinstance(dimension, int)
                or dimension < 1
                or dimension > 256
                or not isinstance(namespace, str)
                or not namespace
            ):
                raise ContractError(
                    f"{task_id}: invalid representation extraction contract")
        by_id[task_id] = task
    return by_id


def _resolve_revision(model_name: str, configured: Optional[str],
                      override: Optional[str]) -> str:
    requested = override or configured
    if requested:
        return requested
    # Import only after CLI validation. This uses the existing fail-closed
    # local-cache/HF resolution logic and returns an immutable commit SHA.
    from run_inference import resolve_effective_revision
    return resolve_effective_revision(model_name, None)


def _assert_text_logits_equivalent(before, after, *, required_token_ids,
                                   agent_id: str) -> dict:
    """Validate repeat text forwards without subtracting signed infinities."""
    import torch

    if before.shape != after.shape:
        raise ContractError(
            f"{agent_id}: text-logit shape changed after vision stripping: "
            f"{tuple(before.shape)} != {tuple(after.shape)}")
    before_nan, after_nan = torch.isnan(before), torch.isnan(after)
    before_posinf, after_posinf = torch.isposinf(before), torch.isposinf(after)
    before_neginf, after_neginf = torch.isneginf(before), torch.isneginf(after)
    counts = {
        "before_nan": int(before_nan.sum().item()),
        "after_nan": int(after_nan.sum().item()),
        "before_posinf": int(before_posinf.sum().item()),
        "after_posinf": int(after_posinf.sum().item()),
        "before_neginf": int(before_neginf.sum().item()),
        "after_neginf": int(after_neginf.sum().item()),
    }
    # A NaN anywhere in the vocabulary can poison generation softmax. Do not
    # excuse it merely because the same NaN appears in both forwards.
    if counts["before_nan"] or counts["after_nan"]:
        raise ContractError(
            f"{agent_id}: unsafe NaN text logits during vision-strip check; {counts}")
    if (not torch.equal(before_posinf, after_posinf)
            or not torch.equal(before_neginf, after_neginf)):
        raise ContractError(
            f"{agent_id}: signed-infinity logit mask changed after vision stripping; "
            f"{counts}")
    for token_id in required_token_ids:
        if token_id < 0 or token_id >= before.shape[-1]:
            raise ContractError(
                f"{agent_id}: required decision token id {token_id} outside logits")
        if (not torch.isfinite(before[..., token_id]).all()
                or not torch.isfinite(after[..., token_id]).all()):
            raise ContractError(
                f"{agent_id}: non-finite decision-code logit for token id {token_id}")
    finite = torch.isfinite(before) & torch.isfinite(after)
    if not finite.any():
        raise ContractError(f"{agent_id}: no finite text logits to compare")
    finite_before, finite_after = before[finite], after[finite]
    max_abs_difference = float((finite_before - finite_after).abs().max().item())
    reference_scale = float(torch.maximum(
        finite_before.abs().max(), finite_after.abs().max()).item())
    allowed = 1e-5 + 1e-4 * reference_scale
    if max_abs_difference > allowed:
        raise ContractError(
            f"{agent_id}: finite text logits changed after vision stripping; "
            f"max_abs_difference={max_abs_difference:.9g}, "
            f"allowed={allowed:.9g}, reference_scale={reference_scale:.9g}, "
            f"nonfinite_counts={counts}")
    return {
        **counts,
        "finite_logits": int(finite.sum().item()),
        "max_abs_difference": max_abs_difference,
        "allowed_difference": allowed,
    }


def _gemma3_four_gpu_device_map(model_config, visible_gpus: int) -> dict:
    """Place Gemma deterministically after Accelerate overestimates fp32+4bit RAM.

    The 1.007B tied embedding and the soon-to-be-removed vision stack remain on
    GPU 0. The 48 decoder layers are split contiguously over GPUs 1--3. This
    map is based on module boundaries and keeps tied input/output embeddings on
    one device; it does not rely on unquantized pre-load memory estimation.
    """
    if visible_gpus != 4:
        raise ContractError(
            f"gemma3_four_gpu requires exactly 4 visible GPUs, got {visible_gpus}")
    text_config = getattr(model_config, "text_config", None)
    layer_count = int(getattr(text_config, "num_hidden_layers", 0))
    if layer_count <= 0 or layer_count % 3 != 0:
        raise ContractError(
            f"unexpected Gemma decoder layer count for 3-way split: {layer_count}")
    layers_per_device = layer_count // 3
    device_map = {
        "model.vision_tower": 0,
        "model.multi_modal_projector": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 3,
        "model.language_model.rotary_emb": 3,
        # lm_head.weight is tied to embed_tokens.weight and must co-locate.
        "lm_head": 0,
    }
    for layer_index in range(layer_count):
        device = 1 + min(layer_index // layers_per_device, 2)
        device_map[f"model.language_model.layers.{layer_index}"] = device
    return device_map


def _gemma3_two_or_four_gpu_device_map(model_config, visible_gpus: int) -> dict:
    """Place one Gemma replica on either a two- or four-GPU worker group.

    Two-GPU placement keeps the large tied embedding/head on local GPU 0 and
    the complete decoder on local GPU 1.  This uses only two inter-device
    boundaries per decode step and permits two independent replicas on four
    physical GPUs.  Four-GPU placement preserves the audited legacy map.
    """
    if visible_gpus == 4:
        return _gemma3_four_gpu_device_map(model_config, visible_gpus)
    if visible_gpus != 2:
        raise ContractError(
            "gemma3_two_or_four_gpu requires exactly 2 or 4 visible GPUs, "
            f"got {visible_gpus}")
    text_config = getattr(model_config, "text_config", None)
    layer_count = int(getattr(text_config, "num_hidden_layers", 0))
    if layer_count <= 0:
        raise ContractError(f"unexpected Gemma decoder layer count: {layer_count}")
    device_map = {
        "model.vision_tower": 0,
        "model.multi_modal_projector": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 1,
        "model.language_model.rotary_emb": 1,
        "lm_head": 0,
    }
    for layer_index in range(layer_count):
        device_map[f"model.language_model.layers.{layer_index}"] = 1
    return device_map


def _qwen35_four_gpu_device_map(model_config, visible_gpus: int) -> dict:
    """Evenly split the 32-layer Qwen text stack over four visible GPUs."""
    if visible_gpus != 4:
        raise ContractError(
            f"qwen35_four_gpu requires exactly 4 visible GPUs, got {visible_gpus}")
    text_config = getattr(model_config, "text_config", None)
    layer_count = int(getattr(text_config, "num_hidden_layers", 0))
    if layer_count <= 0 or layer_count % visible_gpus != 0:
        raise ContractError(f"unexpected Qwen decoder layer count: {layer_count}")
    layers_per_device = layer_count // visible_gpus
    device_map = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 3,
        "model.language_model.rotary_emb": 3,
        "lm_head": 3,
    }
    for layer_index in range(layer_count):
        device_map[f"model.language_model.layers.{layer_index}"] = min(
            layer_index // layers_per_device, visible_gpus - 1)
    return device_map


def _llama_four_gpu_device_map(model_config, visible_gpus: int) -> dict:
    """Evenly split the 32-layer Llama stack over four visible GPUs."""
    if visible_gpus != 4:
        raise ContractError(
            f"llama_four_gpu requires exactly 4 visible GPUs, got {visible_gpus}")
    layer_count = int(getattr(model_config, "num_hidden_layers", 0))
    if layer_count <= 0 or layer_count % visible_gpus != 0:
        raise ContractError(f"unexpected Llama decoder layer count: {layer_count}")
    layers_per_device = layer_count // visible_gpus
    device_map = {"model.embed_tokens": 0, "model.norm": 3,
                  "model.rotary_emb": 3, "lm_head": 3}
    for layer_index in range(layer_count):
        device_map[f"model.layers.{layer_index}"] = min(
            layer_index // layers_per_device, visible_gpus - 1)
    return device_map


def _requested_device_map(agent: Mapping[str, Any], model_config,
                          visible_gpus: int):
    """Use explicit safe sharding for one worker, local auto for one-GPU workers."""
    strategy = agent["device_map_strategy"]
    if strategy == "gemma3_four_gpu":
        return _gemma3_four_gpu_device_map(model_config, visible_gpus)
    if strategy == "gemma3_two_or_four_gpu":
        return _gemma3_two_or_four_gpu_device_map(model_config, visible_gpus)
    if strategy == "gemma3_text_single_gpu":
        if visible_gpus != 1:
            raise ContractError(
                "gemma3_text_single_gpu requires one isolated visible GPU per worker")
        return "auto"
    if visible_gpus == 4 and strategy == "qwen35_four_gpu_or_single":
        return _qwen35_four_gpu_device_map(model_config, visible_gpus)
    if visible_gpus == 4 and strategy == "llama_four_gpu_or_single":
        return _llama_four_gpu_device_map(model_config, visible_gpus)
    return "auto"


def _actual_tensor_devices(model) -> set[str]:
    """Return model-state devices without relying on Accelerate metadata."""
    devices = {
        str(tensor.device)
        for iterator in (model.parameters(), model.buffers())
        for tensor in iterator
    }
    if not devices:
        raise ContractError("loaded model exposes no parameters or buffers")
    return devices


def _tensor_names_on_device(model, device: str, limit: int = 12) -> list[str]:
    """Diagnostic names for fail-closed placement errors."""
    names = []
    for kind, getter_name in (("parameter", "named_parameters"),
                              ("buffer", "named_buffers")):
        getter = getattr(model, getter_name, None)
        if getter is None:
            continue
        for name, tensor in getter():
            if str(tensor.device) == device:
                names.append(f"{kind}:{name}")
                if len(names) >= limit:
                    return names
    return names


def _validate_model_placement(model, agent: Mapping[str, Any], *,
                              visible_gpus: int) -> dict:
    """Fail closed on offload/meta tensors and prove local CUDA placement.

    Transformers may omit ``hf_device_map`` when ``device_map='auto'`` resolves
    to a single device, so actual parameter/buffer devices are authoritative.
    """
    recorded = getattr(model, "hf_device_map", None)
    recorded = recorded if isinstance(recorded, dict) and recorded else None
    if recorded is not None:
        placements = {str(value) for value in recorded.values()}
        if "disk" in placements:
            raise ContractError(f"{agent['id']}: disk offload is not permitted")
        if not agent["allow_fp32_cpu_offload"] and "cpu" in placements:
            raise ContractError(
                f"{agent['id']}: unexpected CPU offload in GPU-only mode")

    actual = _actual_tensor_devices(model)
    if "meta" in actual:
        raise ContractError(f"{agent['id']}: model retains meta tensors: {actual}")
    if not agent["allow_fp32_cpu_offload"] and "cpu" in actual:
        raise ContractError(
            f"{agent['id']}: model has CPU tensors in GPU-only mode: {actual}; "
            f"examples={_tensor_names_on_device(model, 'cpu')}")
    cuda_indices = {
        int(device.split(":", 1)[1])
        for device in actual
        if device.startswith("cuda:") and device.split(":", 1)[1].isdigit()
    }
    non_cuda = {device for device in actual if not device.startswith("cuda:")}
    if non_cuda and not agent["allow_fp32_cpu_offload"]:
        raise ContractError(
            f"{agent['id']}: unexpected non-CUDA model tensors: {non_cuda}")
    expected_indices = set(range(visible_gpus))
    if not cuda_indices or not cuda_indices.issubset(expected_indices):
        raise ContractError(
            f"{agent['id']}: CUDA tensor placement {cuda_indices} is outside "
            f"the {visible_gpus} visible device(s); actual={actual}")
    manual_multi_gpu = agent["device_map_strategy"] in {
        "gemma3_four_gpu", "gemma3_two_or_four_gpu",
        "qwen35_four_gpu_or_single", "llama_four_gpu_or_single",
    } and visible_gpus == 4
    if manual_multi_gpu:
        if cuda_indices != expected_indices:
            raise ContractError(
                f"{agent['id']}: manual Gemma map did not use all four GPUs: "
                f"actual={cuda_indices}, expected={expected_indices}")
        if recorded is None:
            raise ContractError(
                f"{agent['id']}: manual multi-GPU map was not recorded")
    return {
        "recorded_device_map": recorded,
        "actual_tensor_devices": sorted(actual),
    }


def _load_model(agent: Mapping[str, Any], revision: str, precision: str,
                *, expected_visible_gpus: int):
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM,
                              AutoModelForImageTextToText, AutoTokenizer,
                              BitsAndBytesConfig, Gemma3ForCausalLM)

    tokenizer_kwargs = {"revision": revision}
    if agent.get("fix_mistral_regex", False):
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        agent["model"], **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    visible_gpus = torch.cuda.device_count()
    if visible_gpus != expected_visible_gpus:
        raise ContractError(
            f"{agent['id']}: CUDA isolation failure: expected "
            f"{expected_visible_gpus} visible GPU(s), found {visible_gpus}; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    runtime_dtype = getattr(torch, agent["runtime_dtype"])
    quant_compute_dtype_name = agent.get(
        "quant_compute_dtype", agent["runtime_dtype"])
    quant_compute_dtype = getattr(torch, quant_compute_dtype_name)
    model_config = AutoConfig.from_pretrained(agent["model"], revision=revision)
    requested_device_map = _requested_device_map(
        agent, model_config, visible_gpus)
    model_kwargs: Dict[str, Any] = {
        "revision": revision,
        "dtype": runtime_dtype,
        "device_map": requested_device_map,
    }
    text_only_runtime = bool(agent.get("text_only_runtime", False))
    if text_only_runtime:
        # The official multimodal checkpoint stores its decoder below the
        # ``language_model.`` prefix.  Loading that exact pinned state into the
        # official text class avoids ever instantiating the unused vision
        # tower, while full-checkpoint parameters remain counted legally.
        model_kwargs["config"] = model_config.text_config
        model_kwargs["key_mapping"] = {r"^language_model\.": ""}
        model_kwargs["output_loading_info"] = True
    if torch.cuda.is_available() and not isinstance(requested_device_map, dict):
        model_kwargs["max_memory"] = {
            index: int(torch.cuda.get_device_properties(index).total_memory * 0.88)
            for index in range(torch.cuda.device_count())
        }
        if agent["allow_fp32_cpu_offload"]:
            model_kwargs["max_memory"]["cpu"] = "64GiB"
    if precision == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=quant_compute_dtype,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=bool(
                agent["allow_fp32_cpu_offload"]))
    loader = (Gemma3ForCausalLM if text_only_runtime else
              (AutoModelForImageTextToText
               if agent["model_class"] == "multimodal" else AutoModelForCausalLM))
    loaded = loader.from_pretrained(agent["model"], **model_kwargs)
    if text_only_runtime:
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise ContractError("Gemma text-only load did not return loading diagnostics")
        model, loading_info = loaded
        missing = set(loading_info.get("missing_keys", [])) - {"lm_head.weight"}
        unexpected = set(loading_info.get("unexpected_keys", []))
        unexpected = {
            key for key in unexpected
            if not (key.startswith("vision_tower.")
                    or key.startswith("multi_modal_projector."))
        }
        mismatched = loading_info.get("mismatched_keys", [])
        errors = loading_info.get("error_msgs", [])
        if missing or unexpected or mismatched or errors:
            raise ContractError(
                "Gemma text-only checkpoint remap was incomplete: "
                f"missing={sorted(missing)[:8]}, "
                f"unexpected={sorted(unexpected)[:8]}, "
                f"mismatched={mismatched[:4]}, errors={errors[:4]}")
    else:
        model = loaded
    model.eval()
    if agent.get("fp32_residual_stream", False):
        # Store the 1B-entry tied embedding/head in FP16, but promote the
        # embedding output before the decoder. BitsAndBytes still performs its
        # 4-bit linear matmuls with FP16 compute and casts results back to the
        # FP32 input dtype. Residual additions and attention state therefore
        # retain the safe dtype without requiring a second GPU.
        embedding = model.get_input_embeddings()

        def _promote_embedding_output(_module, _inputs, output):
            return output.float()

        embedding.register_forward_hook(_promote_embedding_output)
    head_scale = float(agent.get("fp16_lm_head_input_scale", 1.0))
    if head_scale > 1.0:
        # Gemma's 262k-way tied FP16 head overflowed on Turing. Scaling its
        # input before the FP16 matmul and restoring scale in FP32 is
        # algebraically equivalent apart from expected FP16 rounding, and
        # prevents accumulation overflow without a 4GB FP32 head.
        class _ScaledFP16Head(torch.nn.Module):
            def __init__(self, base, scale: float):
                super().__init__()
                self.base = base
                self.scale = scale

            @property
            def weight(self):
                return self.base.weight

            def forward(self, hidden_states):
                projected = torch.nn.functional.linear(
                    (hidden_states / self.scale).to(self.base.weight.dtype),
                    self.base.weight,
                    getattr(self.base, "bias", None))
                return projected.float() * self.scale

        model.lm_head = _ScaledFP16Head(model.lm_head, head_scale)
    placement = _validate_model_placement(
        model, agent, visible_gpus=visible_gpus)
    print(f"{agent['id']}: CUDA isolation passed: visible_gpus={visible_gpus}; "
          f"runtime_dtype={agent['runtime_dtype']}; "
          f"quant_compute_dtype={quant_compute_dtype_name}; "
          f"resolved_device_map={placement['recorded_device_map']}; "
          f"actual_tensor_devices={placement['actual_tensor_devices']}", flush=True)
    if agent.get("strip_unused_vision") and not text_only_runtime:
        required_token_ids = []
        for code in ("A", "B", "C", "D"):
            tokens = tokenizer(code, add_special_tokens=False)["input_ids"]
            if len(tokens) != 1:
                raise ContractError(
                    f"{agent['id']}: decision code {code!r} is not one token")
            required_token_ids.append(tokens[0])
        probe = tokenizer(
            "Text-only vision-module equivalence check.", return_tensors="pt")
        probe = {key: value.to(model.get_input_embeddings().weight.device)
                 for key, value in probe.items()}
        with torch.inference_mode():
            before_strip = model(
                **probe, logits_to_keep=1, use_cache=False).logits.detach().float().cpu()
        for module_path in agent["unused_text_only_parameter_prefixes"]:
            parent = model
            parts = module_path.split(".")
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    raise ContractError(
                        f"{agent['id']}: cannot strip missing module path {module_path}")
                parent = getattr(parent, part)
            leaf = parts[-1]
            module = getattr(parent, leaf, None)
            if module is None:
                raise ContractError(
                    f"{agent['id']}: cannot strip missing module path {module_path}")
            setattr(parent, leaf, None)
            del module
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        with torch.inference_mode():
            after_strip = model(
                **probe, logits_to_keep=1, use_cache=False).logits.detach().float().cpu()
        diagnostics = _assert_text_logits_equivalent(
            before_strip, after_strip, required_token_ids=required_token_ids,
            agent_id=agent["id"])
        print(f"{agent['id']}: vision stripping text-logit check passed: "
              f"max_abs_difference={diagnostics['max_abs_difference']:.9g}, "
              f"signed_infinities="
              f"{diagnostics['after_posinf'] + diagnostics['after_neginf']}")
        del before_strip, after_strip, probe
    elif text_only_runtime:
        probe = tokenizer(
            "Text-only finite-logit execution check.", return_tensors="pt")
        probe = {key: value.to(model.get_input_embeddings().weight.device)
                 for key, value in probe.items()}
        with torch.inference_mode():
            logits = model(
                **probe, logits_to_keep=1, use_cache=False).logits.detach().float().cpu()
        diagnostics = _assert_text_logits_equivalent(
            logits, logits.clone(), required_token_ids=[
                tokenizer(code, add_special_tokens=False)["input_ids"][0]
                for code in ("A", "B", "C", "D")],
            agent_id=agent["id"])
        print(f"{agent['id']}: text-only fp16 finite-logit check passed: "
              f"head_input_scale={head_scale:g}; "
              f"fp32_residual_stream="
              f"{bool(agent.get('fp32_residual_stream', False))}; "
              f"signed_infinities="
              f"{diagnostics['after_posinf'] + diagnostics['after_neginf']}")
        del logits, probe
    return model, tokenizer


def _prompt_text(tokenizer, prompt: str) -> str:
    # A single user turn works across Qwen, Gemma and Llama and avoids Gemma
    # templates that reject a separate system role.
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def _input_device(model):
    return model.get_input_embeddings().weight.device


def _generate(model, tokenizer, prompt: str, *, n_samples: int,
              temperature: float, max_new_tokens: int, seed: int,
              sample_batch_size: int = 1) -> List[str]:
    return _generate_prompt_batch(
        model, tokenizer, [prompt], n_samples=n_samples,
        temperature=temperature, max_new_tokens=max_new_tokens, seed=seed,
        sample_batch_size=sample_batch_size)[0]


def _generate_prompt_batch(model, tokenizer, prompts: List[str], *, n_samples: int,
                           temperature: float, max_new_tokens: int, seed: int,
                           sample_batch_size: int = 1) -> List[List[str]]:
    import torch

    if not prompts:
        raise ContractError("prompt batch must not be empty")
    rendered = [_prompt_text(tokenizer, prompt) for prompt in prompts]
    encoded = tokenizer(
        rendered, padding=len(rendered) > 1, return_tensors="pt")
    encoded = {key: value.to(_input_device(model)) for key, value in encoded.items()}
    if sample_batch_size < 1:
        raise ContractError("sample_batch_size must be at least one")
    results: List[List[str]] = [[] for _ in prompts]
    # Generate multiple continuations in one decode pass. This reuses prompt
    # prefill and parallelizes token steps across the sample batch. Chunking
    # keeps the KV-cache multiplier bounded on 11GB cards. Batch size one
    # preserves the historical per-sample seed schedule exactly.
    for sample_offset in range(0, n_samples, sample_batch_size):
        batch_count = min(sample_batch_size, n_samples - sample_offset)
        return_count = batch_count if temperature > 0 else 1
        batch_seed = (seed + sample_offset * 104729) % (2**63 - 1)
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)
        kwargs = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "num_return_sequences": return_count,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            kwargs.update({"temperature": temperature, "top_p": 0.95})
        with torch.inference_mode():
            output = model.generate(**kwargs)
        expected_rows = len(prompts) * return_count
        if output.ndim != 2 or output.shape[0] != expected_rows:
            raise ContractError(
                f"generate returned shape {tuple(output.shape)}, expected "
                f"({expected_rows}, sequence_length)")
        for prompt_index in range(len(prompts)):
            start = prompt_index * return_count
            sequences = list(output[start:start + return_count])
            if temperature <= 0 and batch_count > 1:
                sequences *= batch_count
            for sequence in sequences:
                continuation = sequence[encoded["input_ids"].shape[1]:]
                results[prompt_index].append(
                    tokenizer.decode(
                        continuation, skip_special_tokens=True).strip())
    if any(len(values) != n_samples for values in results):
        raise ContractError(
            "batched generation did not return the requested samples per prompt: "
            f"{[len(values) for values in results]} != {n_samples}")
    return results


CHOICE_FORWARD_TOKEN_CELLS = 1024


def _choice_scores_batch(model, tokenizer, tasks: List[Mapping[str, Any]]
                         ) -> List[tuple[Dict[str, dict], Dict[str, float], str]]:
    """Score choice variants with bounded activation-memory microbatches.

    Choice-code balancing creates several prompt variants per logical task.
    ``task_batch_size=1`` previously forwarded all variants together, so one
    long list-valued prompt could still OOM an 11GB Gemma worker.  Bound the
    padded token cells in each forward while retaining every frozen variant.
    """
    import torch
    import torch.nn.functional as functional

    if not tasks or any(task.get("mode") != "choice" for task in tasks):
        raise ContractError("choice batch must contain at least one choice task")
    rendered, offsets = [], []
    for task in tasks:
        start = len(rendered)
        rendered.extend(_prompt_text(tokenizer, variant["prompt"])
                        for variant in task["choice_variants"])
        offsets.append((start, len(rendered)))
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    code_ids: Dict[str, int] = {}
    for code in sorted({code for task in tasks for variant in task["choice_variants"]
                        for code in variant["choice_codes"].values()}):
        tokens = tokenizer(code, add_special_tokens=False)["input_ids"]
        if len(tokens) != 1:
            raise ContractError(
                f"choice code {code!r} must be exactly one token; got {len(tokens)}")
        code_ids[code] = tokens[0]
    width = int(encoded["input_ids"].shape[1])
    forward_batch_size = max(
        1,
        min(len(rendered), CHOICE_FORWARD_TOKEN_CELLS // max(1, width)),
    )
    log_prob_chunks = []
    device = _input_device(model)
    for batch_start in range(0, len(rendered), forward_batch_size):
        batch_stop = min(
            len(rendered), batch_start + forward_batch_size)
        chunk = {
            key: value[batch_start:batch_stop].to(device)
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            logits = model(
                **chunk, logits_to_keep=1).logits[:, -1, :]
        log_prob_chunks.append(
            functional.log_softmax(logits.float(), dim=-1).cpu())
        del chunk, logits
    log_probs = torch.cat(log_prob_chunks, dim=0)
    results = []
    for task, (start, stop) in zip(tasks, offsets):
        choices, variants = task["choices"], task["choice_variants"]
        scores: Dict[str, dict] = {}
        for choice in choices:
            variant_values = [
                float(log_probs[start + variant_index,
                                code_ids[variant["choice_codes"][choice]]].item())
                for variant_index, variant in enumerate(variants)]
            mean = sum(variant_values) / len(variant_values)
            scores[choice] = {
                "codes": [variant["choice_codes"][choice] for variant in variants],
                "variant_logprobs": variant_values,
                "sum_logprob": mean,
                "mean_logprob": mean,
                "token_count": 1,
                "variant_count": len(variant_values),
            }
        probabilities = softmax({choice: value["mean_logprob"]
                                 for choice, value in scores.items()})
        winner = max(choices, key=lambda choice: (
            probabilities[choice], -choices.index(choice)))
        results.append((scores, probabilities, winner))
        if stop - start != len(variants):
            raise AssertionError("choice batch offset drift")
    return results


def _choice_scores(model, tokenizer, choices: List[str],
                   choice_variants: List[Mapping[str, Any]]
                   ) -> tuple[Dict[str, dict], Dict[str, float], str]:
    task = {"mode": "choice", "choices": choices,
            "choice_variants": choice_variants}
    return _choice_scores_batch(model, tokenizer, [task])[0]


_REPRESENTATION_PROJECTION_CACHE: Dict[tuple, Any] = {}


def _projection_matrix(hidden_size: int, projection_dim: int,
                       namespace: str, layer_index: int):
    """Deterministic label-free Rademacher projection on CPU."""
    import hashlib
    import math
    import torch

    key = (hidden_size, projection_dim, namespace, layer_index)
    if key not in _REPRESENTATION_PROJECTION_CACHE:
        seed = int.from_bytes(hashlib.sha256(
            f"representation-projection-v1\x1f{namespace}\x1f"
            f"{hidden_size}\x1f{projection_dim}\x1f{layer_index}".encode()
        ).digest()[:8], "big") % (2**63 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        matrix = torch.randint(
            0, 2, (hidden_size, projection_dim),
            generator=generator, dtype=torch.int8)
        matrix = (
            matrix.to(torch.float32).mul_(2.0).sub_(1.0)
            / math.sqrt(hidden_size))
        _REPRESENTATION_PROJECTION_CACHE[key] = matrix
    return _REPRESENTATION_PROJECTION_CACHE[key]


def _project_hidden_states(hidden_states, tasks: List[Mapping[str, Any]]
                           ) -> List[dict]:
    """Project the last prompt-token state without retaining raw activations."""
    import math
    import torch

    if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) < 2:
        raise ContractError("model did not return decoder hidden states")
    layer_count = len(hidden_states) - 1
    outputs = []
    for batch_index, task in enumerate(tasks):
        fractions = [
            float(value)
            for value in task["representation_layer_fractions"]]
        projection_dim = int(task["representation_projection_dim"])
        namespace = str(task["representation_projection_namespace"])
        layers = {}
        for fraction in fractions:
            layer_index = max(
                1, min(layer_count, int(round(fraction * layer_count))))
            vector = hidden_states[layer_index][batch_index, -1, :]
            vector = vector.detach().float().cpu()
            if not bool(torch.isfinite(vector).all()):
                raise ContractError(
                    f"{task['task_id']}: non-finite hidden representation")
            rms = float(torch.sqrt(torch.mean(vector * vector)).item())
            if not math.isfinite(rms) or rms <= 0.0:
                raise ContractError(
                    f"{task['task_id']}: invalid hidden-state RMS {rms}")
            normalized = vector / rms
            matrix = _projection_matrix(
                int(vector.numel()), projection_dim, namespace, layer_index)
            projected = normalized @ matrix
            layers[f"{fraction:.6g}"] = {
                "layer_index": layer_index,
                "hidden_size": int(vector.numel()),
                "rms": rms,
                "projection": [
                    round(float(value), 8) for value in projected.tolist()],
            }
        outputs.append({
            "schema": "frozen-hidden-representation-v1",
            "projection_namespace": namespace,
            "projection_dim": projection_dim,
            "decoder_layer_count": layer_count,
            "layers": layers,
        })
    return outputs


def _representation_batch(model, tokenizer,
                          tasks: List[Mapping[str, Any]]) -> List[dict]:
    """Extract compact representations for several independent prompts."""
    import torch

    rendered = [
        _prompt_text(tokenizer, str(task["prompt"])) for task in tasks]
    encoded = tokenizer(
        rendered, add_special_tokens=False, padding=True,
        return_tensors="pt")
    encoded = {
        key: value.to(_input_device(model)) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(
            **encoded, output_hidden_states=True,
            use_cache=False, logits_to_keep=1)
    hidden_states = getattr(output, "hidden_states", None)
    representations = _project_hidden_states(hidden_states, tasks)
    del output, hidden_states
    return representations


def _text_token_count(tokenizer, text: str) -> int:
    values = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ContractError("single-text token count unexpectedly returned a batch")
        values = values[0]
    return len(values)


def _synchronize_cuda() -> None:
    """Make wall-clock generation telemetry include all sharded CUDA work."""
    try:
        import torch
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)
    except (RuntimeError, AssertionError):
        # CPU unit-test doubles and a driver disappearing during teardown do
        # not change task correctness; the generation call itself still fails
        # normally if CUDA becomes unusable.
        return


def _attach_generation_telemetry(result: dict, tokenizer, prompt: str,
                                 generations: List[str], *,
                                 batch_wall_seconds: float,
                                 batch_task_count: int) -> None:
    output_counts = [_text_token_count(tokenizer, text) for text in generations]
    result["generation_telemetry"] = {
        "prompt_tokens": _text_token_count(tokenizer, prompt),
        "output_tokens": output_counts,
        "total_output_tokens": sum(output_counts),
        "batch_wall_seconds": batch_wall_seconds,
        "batch_task_count": batch_task_count,
    }


def _run_task(model, tokenizer, task: Mapping[str, Any], base_seed: int,
              generation_batch_size: int = 1) -> dict:
    seed = int.from_bytes(
        __import__("hashlib").sha256(
            f"{base_seed}\x1f{task['task_id']}".encode()).digest()[:8], "big")
    result = {field: task[field] for field in
              ("task_id", "agent_id", "subject", "relation", "phase", "mode")}
    if task["mode"] == "generate":
        _synchronize_cuda()
        started = time.perf_counter()
        result["generations"] = _generate(
            model, tokenizer, task["prompt"], n_samples=task["n_samples"],
            temperature=float(task.get("temperature", 0.8)),
            max_new_tokens=int(task.get("max_new_tokens", 192)), seed=seed,
            sample_batch_size=generation_batch_size)
        _synchronize_cuda()
        elapsed = time.perf_counter() - started
        result["generation_batch_size"] = min(
            generation_batch_size, int(task["n_samples"]))
        _attach_generation_telemetry(
            result, tokenizer, str(task["prompt"]), result["generations"],
            batch_wall_seconds=elapsed, batch_task_count=1)
    elif task["mode"] == "choice":
        scores, probabilities, winner = _choice_scores(
            model, tokenizer, task["choices"], task["choice_variants"])
        result.update({"choice_scores": scores,
                       "choice_probabilities": probabilities,
                       "selected_choice": winner})
    elif task["mode"] == "representation":
        result["representation"] = _representation_batch(
            model, tokenizer, [task])[0]
    else:
        choice = task["constant_choice"]
        result.update({"choice_scores": {choice: {"sum_logprob": 0.0,
                                                   "mean_logprob": 0.0,
                                                   "token_count": 0}},
                       "choice_probabilities": {choice: 1.0},
                       "selected_choice": choice})
    for field in ("candidate_key", "candidate_item", "excluded_proposer_agents"):
        if field in task:
            result[field] = task[field]
    return result


def _generation_signature(task: Mapping[str, Any]) -> tuple:
    if task["mode"] != "generate":
        return (task["mode"],)
    return (
        "generate", int(task["n_samples"]),
        float(task.get("temperature", 0.8)),
        int(task.get("max_new_tokens", 192)),
    )


def _fixed_task_groups(tasks: List[dict], task_batch_size: int) -> List[List[dict]]:
    """Create deterministic consecutive groups without crossing generation settings."""
    if task_batch_size < 1:
        raise ContractError("task_batch_size must be at least one")
    groups: List[List[dict]] = []
    current: List[dict] = []
    for task in tasks:
        if (current and (len(current) >= task_batch_size
                         or _generation_signature(current[0]) !=
                         _generation_signature(task))):
            groups.append(current)
            current = []
        current.append(task)
    if current:
        groups.append(current)
    if [task["task_id"] for group in groups for task in group] != [
            task["task_id"] for task in tasks]:
        raise ContractError("fixed task grouping changed task order or coverage")
    return groups


def _task_group_id(tasks: List[Mapping[str, Any]], base_seed: int) -> str:
    import hashlib
    payload = "\x1f".join(
        [str(base_seed), "generation-task-batch-v1"]
        + [str(task["task_id"]) for task in tasks])
    return hashlib.sha256(payload.encode()).hexdigest()


def _run_task_group(model, tokenizer, tasks: List[Mapping[str, Any]],
                    base_seed: int, generation_batch_size: int) -> List[dict]:
    if not tasks:
        raise ContractError("cannot run an empty task group")
    if len(tasks) == 1:
        results = [_run_task(
            model, tokenizer, tasks[0], base_seed, generation_batch_size)]
    elif tasks[0]["mode"] == "generate":
        signature = _generation_signature(tasks[0])
        if any(_generation_signature(task) != signature for task in tasks):
            raise ContractError("incompatible tasks entered one generation batch")
        group_id = _task_group_id(tasks, base_seed)
        seed = int(group_id[:16], 16) % (2**63 - 1)
        _synchronize_cuda()
        started = time.perf_counter()
        generated = _generate_prompt_batch(
            model, tokenizer, [str(task["prompt"]) for task in tasks],
            n_samples=int(tasks[0]["n_samples"]),
            temperature=float(tasks[0].get("temperature", 0.8)),
            max_new_tokens=int(tasks[0].get("max_new_tokens", 192)),
            seed=seed, sample_batch_size=generation_batch_size)
        _synchronize_cuda()
        elapsed = time.perf_counter() - started
        results = []
        for task, generations in zip(tasks, generated):
            result = {field: task[field] for field in
                      ("task_id", "agent_id", "subject", "relation", "phase", "mode")}
            result["generations"] = generations
            result["generation_batch_size"] = min(
                generation_batch_size, int(task["n_samples"]))
            _attach_generation_telemetry(
                result, tokenizer, str(task["prompt"]), generations,
                batch_wall_seconds=elapsed, batch_task_count=len(tasks))
            for field in ("candidate_key", "candidate_item", "excluded_proposer_agents"):
                if field in task:
                    result[field] = task[field]
            results.append(result)
    elif tasks[0]["mode"] == "choice":
        if any(task["mode"] != "choice" for task in tasks):
            raise ContractError("incompatible tasks entered one choice batch")
        results = []
        for task, (scores, probabilities, winner) in zip(
                tasks, _choice_scores_batch(model, tokenizer, list(tasks))):
            result = {field: task[field] for field in
                      ("task_id", "agent_id", "subject", "relation", "phase", "mode")}
            result.update({"choice_scores": scores,
                           "choice_probabilities": probabilities,
                           "selected_choice": winner})
            for field in ("candidate_key", "candidate_item", "excluded_proposer_agents"):
                if field in task:
                    result[field] = task[field]
            results.append(result)
    elif tasks[0]["mode"] == "representation":
        if any(task["mode"] != "representation" for task in tasks):
            raise ContractError(
                "incompatible tasks entered one representation batch")
        representations = _representation_batch(
            model, tokenizer, list(tasks))
        results = []
        for task, representation in zip(tasks, representations):
            result = {field: task[field] for field in
                      ("task_id", "agent_id", "subject", "relation",
                       "phase", "mode")}
            result["representation"] = representation
            for field in (
                    "candidate_key", "candidate_item",
                    "excluded_proposer_agents"):
                if field in task:
                    result[field] = task[field]
            results.append(result)
    else:
        if any(task["mode"] != "constant" for task in tasks):
            raise ContractError("incompatible tasks entered one constant batch")
        results = [_run_task(
            model, tokenizer, task, base_seed, generation_batch_size)
                   for task in tasks]
    group_id = _task_group_id(tasks, base_seed)
    for result in results:
        if result["mode"] == "generate":
            result["generation_task_batch_size"] = len(tasks)
            result["generation_task_batch_id"] = group_id
        elif result["mode"] == "choice":
            result["choice_task_batch_size"] = len(tasks)
            result["choice_task_batch_id"] = group_id
        elif result["mode"] == "representation":
            result["representation_task_batch_size"] = len(tasks)
            result["representation_task_batch_id"] = group_id
    return results


def _worker(worker_index: int, gpu_ids: List[int], agent: dict, revision: str,
            precision: str, base_seed: int, generation_batch_size: int,
            task_queue: "mp.Queue",
            result_queue: "mp.Queue") -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in gpu_ids)
    try:
        model, tokenizer = _load_model(
            agent, revision, precision, expected_visible_gpus=len(gpu_ids))
    except Exception as exc:
        result_queue.put((None, None,
                          f"worker {worker_index} model load failed: {type(exc).__name__}: "
                          f"{exc}\n{traceback.format_exc()}"))
        return
    while True:
        unit = task_queue.get()
        if unit is None:
            break
        try:
            tasks, needed_ids = unit
            responses = _run_task_group(
                model, tokenizer, tasks, base_seed, generation_batch_size)
            for response in responses:
                if response["task_id"] in needed_ids:
                    result_queue.put((response["task_id"], response, None))
        except Exception as exc:
            task_id = tasks[0]["task_id"] if tasks else None
            result_queue.put((task_id, None,
                              f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--agents", default=str(Path(__file__).with_name("agents.json")))
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--precision", choices=["4bit", "fp16"], default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--generation-batch-size", type=int, default=1,
        help="number of N-sample continuations decoded together (1 preserves legacy)")
    parser.add_argument(
        "--task-batch-size", type=int, default=1,
        help="number of compatible subject prompts decoded together")
    parser.add_argument("--smoke-limit", type=int, default=0)
    parser.add_argument("--smoke-offset", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.generation_batch_size < 1 or args.task_batch_size < 1
            or args.smoke_offset < 0):
        raise SystemExit("generation and task batch sizes must be at least one")

    config_path = Path(args.agents).resolve()
    config = load_agent_config(config_path)
    agent = _agent(config, args.agent_id)
    task_path = Path(args.tasks).resolve()
    output_path = Path(args.output).resolve()
    tasks = read_jsonl(task_path)
    by_id = validate_tasks(tasks, args.agent_id)
    if args.smoke_limit:
        tasks = tasks[args.smoke_offset:args.smoke_offset + args.smoke_limit]
        by_id = validate_tasks(tasks, args.agent_id)
    elif args.smoke_offset:
        raise SystemExit("--smoke-offset requires --smoke-limit")
    # Constants are materialized in the parent process and must not fragment
    # otherwise compatible neural batches (the commitment file intentionally
    # interleaves constant and choice phases for some relations).
    neural_tasks = [task for task in tasks if task["mode"] != "constant"]
    fixed_groups = _fixed_task_groups(neural_tasks, args.task_batch_size)
    expected_group = {
        task["task_id"]: (len(group), _task_group_id(group, args.seed))
        for group in fixed_groups for task in group
    }
    precision = args.precision or agent["precision"]
    revision = _resolve_revision(agent["model"], agent.get("revision"), args.model_revision)

    completed: Dict[str, dict] = {}
    if output_path.is_file():
        for response in read_jsonl(output_path):
            task_id = response.get("task_id")
            if task_id not in by_id:
                raise ContractError(f"stale/foreign response task: {task_id}")
            validate_task_response(by_id[task_id], response)
            if (by_id[task_id]["mode"] == "generate"
                    and int(response.get("generation_batch_size", 1)) != min(
                        args.generation_batch_size,
                        int(by_id[task_id]["n_samples"]))):
                raise ContractError(
                    f"{task_id}: response generation batch size is incompatible "
                    "with this resume command")
            if by_id[task_id]["mode"] == "generate":
                expected_size, expected_id = expected_group[task_id]
                actual_size = int(response.get("generation_task_batch_size", 1))
                actual_id = response.get("generation_task_batch_id")
                if (actual_size != expected_size
                        or (actual_id is not None and actual_id != expected_id)
                        or (actual_id is None and expected_size != 1)):
                    raise ContractError(
                        f"{task_id}: response task batch is incompatible with "
                        "this deterministic resume plan")
            elif by_id[task_id]["mode"] == "choice":
                expected_size, expected_id = expected_group[task_id]
                if (int(response.get("choice_task_batch_size", 1)) != expected_size
                        or response.get("choice_task_batch_id") != expected_id):
                    raise ContractError(
                        f"{task_id}: response choice batch is incompatible with "
                        "this deterministic resume plan")
            elif by_id[task_id]["mode"] == "representation":
                expected_size, expected_id = expected_group[task_id]
                if (
                    int(response.get(
                        "representation_task_batch_size", 1)) != expected_size
                    or response.get(
                        "representation_task_batch_id") != expected_id
                ):
                    raise ContractError(
                        f"{task_id}: response representation batch is "
                        "incompatible with this deterministic resume plan")
            completed[task_id] = response
    pending = [task for task in tasks if task["task_id"] not in completed]
    print(f"agent={args.agent_id} model={agent['model']} revision={revision} "
          f"complete={len(completed)} pending={len(pending)} precision={precision} "
          f"runtime_dtype={agent['runtime_dtype']} "
          f"quant_compute_dtype={agent.get('quant_compute_dtype', agent['runtime_dtype'])}")
    if args.dry_run:
        return 0

    # Schema-guaranteed constants require no checkpoint. Materialize them in
    # the parent process so only genuine neural scoring enters the GPU queue.
    constant_tasks = [task for task in pending if task["mode"] == "constant"]
    for task in constant_tasks:
        completed[task["task_id"]] = _run_task(
            None, None, task, args.seed, args.generation_batch_size)
    if constant_tasks:
        write_jsonl_atomic(output_path, [completed[task["task_id"]] for task in tasks
                                         if task["task_id"] in completed])
        pending = [task for task in pending if task["mode"] != "constant"]
        print(f"materialized {len(constant_tasks)} constant tasks without model load; "
              f"GPU pending={len(pending)}")

    if not pending:
        # Empty review shards and fully resumed agents must not load a 8--12B
        # checkpoint merely to discover that no work remains.
        ordered = [completed[task["task_id"]] for task in tasks]
        write_jsonl_atomic(output_path, ordered)
        manifest = {
            "schema": "heterogeneous-agent-responses-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent_id": args.agent_id, "model": agent["model"], "revision": revision,
            "precision": precision, "runtime_dtype": agent["runtime_dtype"],
            "quant_compute_dtype": agent.get(
                "quant_compute_dtype", agent["runtime_dtype"]),
            "fp32_cpu_offload": agent["allow_fp32_cpu_offload"],
            "device_map_strategy": agent["device_map_strategy"],
            "seed": args.seed, "tasks": len(tasks),
            "generation_batch_size": args.generation_batch_size,
            "task_batch_size": args.task_batch_size,
            "task_path": str(task_path), "task_sha256": sha256(task_path),
            "agent_config": str(config_path), "agent_config_sha256": sha256(config_path),
            "output": str(output_path), "output_sha256": sha256(output_path),
            "declared_portfolio_parameter_upper_bound": config["declared_parameter_total"],
            "verified_checkpoint_parameter_count": agent["verified_parameter_count"],
            "active_text_inference_parameter_count":
                agent["active_text_inference_parameter_count"],
            "stripped_unused_vision": bool(agent["strip_unused_vision"]),
        }
        output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"complete without model load: {output_path}")
        return 0

    import torch
    total_gpus = torch.cuda.device_count()
    if total_gpus < 1:
        raise SystemExit("No CUDA GPUs visible; refusing silent CPU model inference")
    if args.num_workers < 1 or args.num_workers > total_gpus:
        raise SystemExit(f"--num-workers must be in [1,{total_gpus}]")
    if (agent["device_map_strategy"] == "gemma3_two_or_four_gpu"
            and args.num_workers == 2):
        if total_gpus != 4:
            raise SystemExit(
                "two-replica Gemma diagnostic requires exactly four visible GPUs")
        gpu_groups = [[0, 1], [2, 3]]
    else:
        gpu_groups = [[index] for index in range(args.num_workers)]
    if args.num_workers == 1:
        gpu_groups = [list(range(total_gpus))]

    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    pending_ids = {task["task_id"] for task in pending}
    work_units = []
    for group in fixed_groups:
        needed = {task["task_id"] for task in group} & pending_ids
        if needed:
            work_units.append((group, needed))
            task_queue.put((group, needed))
    for _ in gpu_groups:
        task_queue.put(None)
    workers = []
    for worker_index, gpu_ids in enumerate(gpu_groups):
        process = context.Process(
            target=_worker,
            args=(worker_index, gpu_ids, agent, revision, precision, args.seed,
                  args.generation_batch_size, task_queue, result_queue))
        process.start()
        workers.append(process)
        print(f"worker {worker_index}: GPU(s) {gpu_ids}")

    failures = []
    received = 0
    started = time.monotonic()
    while received < len(pending):
        try:
            task_id, response, error = result_queue.get(timeout=900)
        except queue.Empty:
            if not any(process.is_alive() for process in workers):
                failures.append("all workers exited before completing pending tasks")
                break
            continue
        if task_id is None:
            failures.append(error)
            break
        received += 1
        if error:
            failures.append(f"{task_id}: {error}")
        else:
            validate_task_response(by_id[task_id], response)
            completed[task_id] = response
        if received % max(1, args.checkpoint_every) == 0 or received == len(pending):
            ordered = [completed[task["task_id"]] for task in tasks
                       if task["task_id"] in completed]
            write_jsonl_atomic(output_path, ordered)
            elapsed = max(time.monotonic() - started, 1e-6)
            print(f"{received}/{len(pending)} new tasks; {received / elapsed:.3f} tasks/s")
        if failures:
            break

    for process in workers:
        if failures and process.is_alive():
            process.terminate()
        process.join(timeout=30)
    if failures:
        raise RuntimeError("agent run failed:\n" + "\n".join(failures[:5]))
    if len(completed) != len(tasks):
        raise RuntimeError(f"incomplete artifact: {len(completed)}/{len(tasks)} tasks")

    manifest = {
        "schema": "heterogeneous-agent-responses-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "agent_id": args.agent_id, "model": agent["model"], "revision": revision,
        "precision": precision, "runtime_dtype": agent["runtime_dtype"],
        "quant_compute_dtype": agent.get(
            "quant_compute_dtype", agent["runtime_dtype"]),
        "fp32_cpu_offload": agent["allow_fp32_cpu_offload"],
        "device_map_strategy": agent["device_map_strategy"],
        "seed": args.seed, "tasks": len(tasks),
        "generation_batch_size": args.generation_batch_size,
        "task_batch_size": args.task_batch_size,
        "task_path": str(task_path), "task_sha256": sha256(task_path),
        "agent_config": str(config_path), "agent_config_sha256": sha256(config_path),
        "output": str(output_path), "output_sha256": sha256(output_path),
        "declared_portfolio_parameter_upper_bound": config["declared_parameter_total"],
        "verified_checkpoint_parameter_count": agent["verified_parameter_count"],
        "active_text_inference_parameter_count":
            agent["active_text_inference_parameter_count"],
        "stripped_unused_vision": bool(agent["strip_unused_vision"]),
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"complete: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
