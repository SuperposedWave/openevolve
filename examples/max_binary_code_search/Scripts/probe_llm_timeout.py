#!/usr/bin/env python3
"""Probe chat-completions timeout behavior using an OpenEvolve config."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    llm_config_path = config.get("llm_config_path")
    if llm_config_path:
        external_path = Path(str(llm_config_path)).expanduser()
        if not external_path.is_absolute():
            external_path = path.parent / external_path
        with external_path.open("r", encoding="utf-8") as handle:
            external = yaml.safe_load(handle) or {}
        external_llm = external.get("llm", external)
        inline_llm = config.get("llm") or {}
        merged_llm = dict(external_llm)
        merged_llm.update(inline_llm)
        config["llm"] = merged_llm

    return config


def _post_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    chat_template_kwargs: dict | None = None,
) -> dict:
    url = api_base.rstrip("/") + "/chat/completions"
    request_body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if chat_template_kwargs is not None:
        request_body["chat_template_kwargs"] = chat_template_kwargs
    body = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            elapsed = time.monotonic() - started
            text = payload.decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {"raw_response_prefix": text[:500]}
            message = (
                parsed.get("choices", [{}])[0].get("message", {})
                if isinstance(parsed, dict)
                else {}
            )
            content = message.get("content") or message.get("reasoning") or ""
            return {
                "ok": True,
                "status": getattr(response, "status", None),
                "elapsed_sec": round(elapsed, 2),
                "response_bytes": len(payload),
                "content_chars": len(content),
            }
    except urllib.error.HTTPError as error:
        elapsed = time.monotonic() - started
        error_body = error.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": error.code,
            "elapsed_sec": round(elapsed, 2),
            "error_type": "HTTPError",
            "error_message": error.reason,
            "error_body_prefix": error_body[:500],
        }
    except Exception as error:
        elapsed = time.monotonic() - started
        return {
            "ok": False,
            "status": None,
            "elapsed_sec": round(elapsed, 2),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }


def _run_case(name: str, calls: list[dict], workers: int) -> dict:
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_post_chat_completion, **call) for call in calls]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    elapsed = time.monotonic() - started
    ok_count = sum(1 for item in results if item["ok"])
    return {
        "case": name,
        "calls": len(calls),
        "workers": workers,
        "ok_count": ok_count,
        "fail_count": len(results) - ok_count,
        "elapsed_sec": round(elapsed, 2),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/max_binary_code_search/Configs/config.yaml",
        help="OpenEvolve config path.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=240.0,
        help="Per HTTP request timeout for this probe.",
    )
    parser.add_argument(
        "--probe-max-tokens",
        type=int,
        default=512,
        help="Max tokens for non-heavy probe cases.",
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Also test the config's full llm.max_tokens value.",
    )
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    llm = config["llm"]
    prompt = config["prompt"]
    evaluator = config["evaluator"]

    api_key = os.environ.get("OPENAI_API_KEY") or llm.get("api_key")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set and llm.api_key is missing.")

    api_base = llm["api_base"]
    model = llm["models"][0]["name"]
    temperature = float(llm.get("temperature", 0.7))
    config_max_tokens = int(llm.get("max_tokens", 1024))
    config_timeout = float(llm.get("timeout", args.request_timeout))
    chat_template_kwargs = llm.get("chat_template_kwargs")
    parallel_evaluations = int(evaluator.get("parallel_evaluations", 1))

    base_call = {
        "api_base": api_base,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "timeout": min(args.request_timeout, config_timeout),
        "chat_template_kwargs": chat_template_kwargs,
    }
    tiny_messages = [
        {"role": "system", "content": "Reply with exactly: ok"},
        {"role": "user", "content": "Health check."},
    ]
    config_messages = [
        {"role": "system", "content": prompt["system_message"]},
        {
            "role": "user",
            "content": (
                "Return a tiny Python priority function with the required signature. "
                "Keep the answer under 80 lines."
            ),
        },
    ]

    print(
        json.dumps(
            {
                "config": args.config,
                "api_base": api_base,
                "model": model,
                "temperature": temperature,
                "config_max_tokens": config_max_tokens,
                "config_timeout": config_timeout,
                "probe_request_timeout": min(args.request_timeout, config_timeout),
                "parallel_evaluations": parallel_evaluations,
            },
            indent=2,
        )
    )

    cases = [
        (
            "tiny_single",
            [
                {
                    **base_call,
                    "messages": tiny_messages,
                    "max_tokens": min(args.probe_max_tokens, 64),
                }
            ],
            1,
        ),
        (
            "config_prompt_single",
            [
                {
                    **base_call,
                    "messages": config_messages,
                    "max_tokens": args.probe_max_tokens,
                }
            ],
            1,
        ),
        (
            "config_prompt_parallel",
            [
                {
                    **base_call,
                    "messages": config_messages,
                    "max_tokens": args.probe_max_tokens,
                }
                for _ in range(parallel_evaluations)
            ],
            parallel_evaluations,
        ),
    ]
    if args.include_heavy:
        cases.append(
            (
                "config_prompt_full_max_tokens",
                [
                    {
                        **base_call,
                        "messages": config_messages,
                        "max_tokens": config_max_tokens,
                    }
                ],
                1,
            )
        )

    for name, calls, workers in cases:
        print(json.dumps(_run_case(name, calls, workers), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
