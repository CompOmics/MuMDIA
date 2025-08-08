"""
Centralized config/argument loader for MuMDIA.

Precedence (highest to lowest):
1) Explicit CLI args
2) Environment variables with prefix MUMDIA_
3) Existing JSON config values
4) Argparse defaults

This keeps run.py simple while allowing flexible overrides without
duplicating parsing/merging logic across the codebase.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from utilities.logger import log_info


_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f"}


def _coerce_value(raw: str, target_type: Any) -> Any:
    """Coerce string from env into target type.

    Supports bool/int/float; falls back to raw string.
    """
    if target_type is bool or isinstance(target_type, bool):
        val = raw.strip().lower()
        if val in _TRUE:
            return True
        if val in _FALSE:
            return False
        # Fallback: non-empty string considered True
        return bool(val)

    if target_type is int or isinstance(target_type, int):
        try:
            return int(raw)
        except Exception:
            return raw

    if target_type is float or isinstance(target_type, float):
        try:
            return float(raw)
        except Exception:
            return raw

    return raw


def _defaults_from_parser(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    return {
        action.dest: action.default
        for action in parser._actions
        if getattr(action, "dest", None) is not None and action.default is not None
    }


def _explicit_cli_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Dict[str, Any]:
    explicit: Dict[str, Any] = {}
    argv = set(sys.argv)
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest:
            continue
        # If any of the option strings was present on CLI, treat as explicit
        if any(opt in argv for opt in getattr(action, "option_strings", [])):
            explicit[dest] = getattr(args, dest)
    return explicit


def _env_overrides(
    parser: argparse.ArgumentParser, args: argparse.Namespace, prefix: str = "MUMDIA_"
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if not dest:
            continue
        env_key = f"{prefix}{dest.upper()}"
        if env_key in os.environ:
            # Determine a target type using current value (arg or default)
            current = getattr(args, dest, None)
            # Fall back to the action.default if args doesn't contain it
            if current is None:
                current = getattr(action, "default", None)
            overrides[dest] = _coerce_value(
                os.environ[env_key], type(current) if current is not None else str
            )
    return overrides


def merge_config_from_sources(
    existing_config: Dict[str, Any] | None,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Merge configuration from defaults, existing JSON, env and CLI.

    Returns a full config dict with keys:
    - "mumdia": merged CLI-related settings
    - "sage_basic" and "sage": ensured to contain mzml_paths and database.fasta
    """
    config: Dict[str, Any] = (
        existing_config.copy() if isinstance(existing_config, dict) else {}
    )

    # Ensure sections exist
    config.setdefault("mumdia", {})
    config.setdefault("sage_basic", {})
    config.setdefault("sage", {})

    # 1) argparse defaults
    merged: Dict[str, Any] = _defaults_from_parser(parser)

    # 2) existing JSON values
    merged.update(config["mumdia"])

    # 3) env overrides
    merged.update(_env_overrides(parser, args))

    # 4) explicit CLI overrides
    merged.update(_explicit_cli_args(parser, args))

    # Persist back under mumdia
    config["mumdia"] = merged

    # Ensure mzML/FASTA are synchronized into sage sections
    mzml_file = merged.get("mzml_file")
    fasta_file = merged.get("fasta_file")

    if mzml_file:
        config["sage_basic"].setdefault("mzml_paths", [mzml_file])
        config["sage"].setdefault("mzml_paths", [mzml_file])
    if fasta_file:
        db_basic = config["sage_basic"].get("database", {})
        db_sage = config["sage"].get("database", {})
        db_basic.setdefault("fasta", fasta_file)
        db_sage.setdefault("fasta", fasta_file)
        config["sage_basic"]["database"] = db_basic
        config["sage"]["database"] = db_sage

    return config


def write_updated_config(config: Dict[str, Any], result_dir: str) -> str:
    new_config_path = os.path.join(result_dir, "updated_config.json")
    with open(new_config_path, "w") as f:
        json.dump(config, f, indent=4)
    log_info(f"Configuration updated and saved to {new_config_path}")
    return new_config_path
