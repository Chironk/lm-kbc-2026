#!/usr/bin/env python3
"""Cheap access/compatibility preflight; never downloads model weight shards."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

from lm_kbc.core import load_agent_config
from lm_kbc.run_agent import _requested_device_map


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", default=str(Path(__file__).with_name("agents.json")))
    parser.add_argument("--online-access", action="store_true",
                        help="Resolve immutable Hugging Face revisions (metadata alone does not prove gated artifact access)")
    parser.add_argument("--tokenizer-check", action="store_true",
                        help="Also load tokenizers and render one chat turn (small downloads possible)")
    parser.add_argument("--minimum-free-gib", type=float, default=80.0)
    args = parser.parse_args()

    config_path = Path(args.agents).resolve()
    config = load_agent_config(config_path)
    import torch
    import transformers

    free = shutil.disk_usage(Path.home()).free / (1024 ** 3)
    failures = []
    report = {
        "agent_config": str(config_path),
        "declared_parameter_total": config["declared_parameter_total"],
        "verified_parameter_total": config["verified_parameter_total"],
        "active_text_inference_parameter_total":
            config["active_text_inference_parameter_total"],
        "parameter_cap": config["parameter_cap"],
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "free_home_gib": free,
        "agents": {},
    }
    if free < args.minimum_free_gib:
        failures.append(
            f"only {free:.1f} GiB free under home; require {args.minimum_free_gib:.1f} GiB")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        failures.append("no CUDA GPU visible")

    if args.online_access:
        from huggingface_hub import HfApi
        api = HfApi()
    for agent in config["agents"]:
        row = {
            "model": agent["model"], "configured_revision": agent.get("revision"),
            "parameter_upper_bound": agent["parameter_upper_bound"],
            "verified_parameter_count": agent["verified_parameter_count"],
            "runtime_dtype": agent["runtime_dtype"],
            "allow_fp32_cpu_offload": agent["allow_fp32_cpu_offload"],
            "device_map_strategy": agent["device_map_strategy"],
            "gated": bool(agent.get("gated", False)),
        }
        if args.online_access:
            try:
                info = api.model_info(agent["model"], revision=agent.get("revision"))
                row["resolved_revision"] = info.sha
                row["metadata_lookup"] = "ok"
            except Exception as exc:
                row["metadata_lookup"] = f"FAILED: {type(exc).__name__}: {exc}"
                failures.append(f"{agent['id']}: Hugging Face metadata lookup failed")
        # Count the actual pinned architecture on the meta device. This loads
        # no weight shard and allocates no full parameter tensors. Tied weights
        # are explicitly tied before counting unique Parameter objects.
        try:
            from accelerate import init_empty_weights
            from transformers import (AutoConfig, AutoModelForCausalLM,
                                      AutoModelForImageTextToText)
            model_config = AutoConfig.from_pretrained(
                agent["model"], revision=agent.get("revision"))
            auto_class = (AutoModelForImageTextToText
                          if agent["model_class"] == "multimodal"
                          else AutoModelForCausalLM)
            with init_empty_weights():
                empty_model = auto_class.from_config(model_config)
            empty_model.tie_weights()
            counted = sum(parameter.numel() for parameter in empty_model.parameters())
            row["architecture_parameter_count"] = counted
            row["architecture_parameter_count_status"] = "ok"
            if counted != agent["verified_parameter_count"]:
                failures.append(
                    f"{agent['id']}: pinned architecture count {counted:,} != "
                    f"verified config count {agent['verified_parameter_count']:,}")
                row["architecture_parameter_count_status"] = "MISMATCH"
            if counted > agent["parameter_upper_bound"]:
                failures.append(
                    f"{agent['id']}: architecture count {counted:,} exceeds "
                    f"upper bound {agent['parameter_upper_bound']:,}")
            prefixes = tuple(agent["unused_text_only_parameter_prefixes"])
            unused_counted = sum(
                parameter.numel() for name, parameter in empty_model.named_parameters()
                if prefixes and name.startswith(prefixes))
            active_counted = counted - unused_counted
            row["unused_text_only_parameter_count"] = unused_counted
            row["active_text_inference_parameter_count"] = active_counted
            if unused_counted != agent["unused_text_only_parameter_count"]:
                failures.append(
                    f"{agent['id']}: removable vision/projector count "
                    f"{unused_counted:,} != configured "
                    f"{agent['unused_text_only_parameter_count']:,}")
            if active_counted != agent["active_text_inference_parameter_count"]:
                failures.append(
                    f"{agent['id']}: active text count {active_counted:,} != configured "
                    f"{agent['active_text_inference_parameter_count']:,}")
            # Only strategies with an explicit four-GPU module map can be
            # audited this way.  ``gemma3_text_single_gpu`` intentionally
            # resolves through Accelerate ``auto`` after CUDA isolation and
            # must not be called with four visible devices here.
            if agent["device_map_strategy"] in {
                    "gemma3_four_gpu", "gemma3_two_or_four_gpu",
                    "qwen35_four_gpu_or_single", "llama_four_gpu_or_single"}:
                manual_map = _requested_device_map(agent, model_config, 4)
                if not isinstance(manual_map, dict):
                    raise RuntimeError("manual four-GPU strategy did not return a map")
                named_state = list(empty_model.named_parameters()) + list(
                    empty_model.named_buffers())
                unmapped = [
                    name for name, _ in named_state
                    if not any(name == prefix or name.startswith(prefix + ".")
                               for prefix in manual_map)]
                row["manual_device_map_entries"] = len(manual_map)
                row["manual_device_map_parameter_coverage"] = (
                    "ok" if not unmapped else f"UNMAPPED: {unmapped[:5]}")
                if unmapped:
                    failures.append(
                        f"{agent['id']}: manual device map leaves model state unmapped")
            del empty_model
            gc.collect()
        except Exception as exc:
            row["architecture_parameter_count_status"] = (
                f"FAILED: {type(exc).__name__}: {exc}")
            failures.append(f"{agent['id']}: architecture parameter count failed")
        if args.tokenizer_check:
            try:
                from transformers import (AutoConfig, AutoModelForCausalLM,
                                          AutoModelForImageTextToText, AutoTokenizer)
                tokenizer_kwargs = {"revision": agent.get("revision")}
                if agent.get("fix_mistral_regex", False):
                    tokenizer_kwargs["fix_mistral_regex"] = True
                tokenizer = AutoTokenizer.from_pretrained(
                    agent["model"], **tokenizer_kwargs)
                model_config = AutoConfig.from_pretrained(
                    agent["model"], revision=agent.get("revision"))
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "Reply with ANSWER: test"}],
                    tokenize=False, add_generation_prompt=True)
                row["tokenizer_chat_template"] = "ok" if rendered else "empty"
                code_lengths = {
                    code: len(tokenizer(code, add_special_tokens=False)["input_ids"])
                    for code in ("A", "B", "C", "D")}
                row["choice_code_token_lengths"] = code_lengths
                if any(length != 1 for length in code_lengths.values()):
                    raise RuntimeError(
                        f"choice codes must be one token each, got {code_lengths}")
                row["authenticated_artifact_access"] = "ok"
                auto_class = (AutoModelForImageTextToText
                              if agent["model_class"] == "multimodal"
                              else AutoModelForCausalLM)
                # Resolve the lazy Transformers mapping without constructing
                # billions of parameters or downloading any weight shard.
                auto_class._model_mapping[type(model_config)]
                row["transformers_model_mapping"] = "ok"
            except Exception as exc:
                row["authenticated_artifact_access"] = "FAILED"
                row["tokenizer_chat_template"] = (
                    f"FAILED: {type(exc).__name__}: {exc}")
                failures.append(f"{agent['id']}: tokenizer/chat-template check failed")
        report["agents"][agent["id"]] = row

    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        print("PREFLIGHT FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 2
    print("PREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
