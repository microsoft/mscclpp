# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import json
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_KNOWN_GPU_SKUS = ("GB300", "MI300X", "H100", "A100")


@dataclass(frozen=True)
class HardwareProfile:
    sku: str | None = None
    scale: int | None = None


@dataclass(frozen=True)
class TunedConfig:
    algorithm: str
    nblocks: int | None = None
    nthreads: int | None = None
    symmetric_memory: bool = False
    time_us: float | None = None


@dataclass(order=True, frozen=True)
class TunedConfigBySize:
    message_size: int
    config: TunedConfig


class TunedConfigStore:
    def __init__(self, profiles: dict[HardwareProfile, dict[str, list[TunedConfigBySize]]]) -> None:
        self._profiles = profiles

    @classmethod
    def empty(cls) -> "TunedConfigStore":
        return cls({})

    @classmethod
    def load_path(cls, path: str | Path) -> "TunedConfigStore":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_payload(json.load(handle))

    @classmethod
    def from_payload(cls, payload: Any) -> "TunedConfigStore":
        if not isinstance(payload, dict):
            raise ValueError("MSCCL++ tuned config must be a JSON object")
        raw_profiles = payload.get("profiles")
        if not isinstance(raw_profiles, list):
            raise ValueError("MSCCL++ tuned config must contain a 'profiles' list")
        profiles: dict[HardwareProfile, dict[str, list[TunedConfigBySize]]] = {}
        for raw_profile in raw_profiles:
            profile = _profile_from_payload(raw_profile)
            profiles[profile] = _configs_by_collective_from_payload(raw_profile.get("collectives", {}))
        return cls(profiles)

    def select(self, profile: HardwareProfile, collective: str, message_size: int) -> TunedConfig | None:
        for _, configs_by_collective in _matching_profiles(self._profiles, profile):
            config = _select_config(configs_by_collective, collective, message_size)
            if config is not None:
                return config
        return None

    def upsert(self, profile: HardwareProfile, collective: str, message_size: int, config: TunedConfig) -> None:
        configs = self._profiles.setdefault(profile, {}).setdefault(collective, [])
        for index, existing in enumerate(configs):
            if existing.message_size == message_size:
                configs[index] = TunedConfigBySize(message_size, config)
                break
        else:
            configs.append(TunedConfigBySize(message_size, config))
        configs.sort(key=lambda item: item.message_size)

    def write_path(self, path: str | Path) -> None:
        profiles_payload: list[dict[str, Any]] = []
        for profile, configs_by_collective in sorted(
            self._profiles.items(),
            key=lambda item: (item[0].sku is None, item[0].sku or "", item[0].scale is None, item[0].scale or 0),
        ):
            collectives: dict[str, list[dict[str, Any]]] = {}
            for collective, configs in sorted(configs_by_collective.items()):
                collectives[collective] = [_config_entry_payload(item) for item in sorted(configs)]
            profile_payload: dict[str, Any] = {}
            if profile.sku is not None:
                profile_payload["sku"] = profile.sku
            if profile.scale is not None:
                profile_payload["scale"] = profile.scale
            profile_payload["collectives"] = collectives
            profiles_payload.append(profile_payload)

        with Path(path).open("w", encoding="utf-8") as handle:
            handle.write(_format_tuned_config_json({"version": 1, "profiles": profiles_payload}))


def normalize_sku(raw_sku: str) -> str:
    upper_sku = raw_sku.upper()
    for known_sku in _KNOWN_GPU_SKUS:
        if known_sku in upper_sku:
            return known_sku
    normalized = re.sub(r"[^A-Z0-9]+", "_", upper_sku).strip("_")
    return normalized or "UNKNOWN"


def _profile_from_payload(raw_profile: Any) -> HardwareProfile:
    if not isinstance(raw_profile, dict):
        raise ValueError(f"Invalid tuned config profile: {raw_profile!r}")
    raw_sku = raw_profile.get("sku")
    return HardwareProfile(
        sku=None if raw_sku is None else normalize_sku(str(raw_sku)),
        scale=_optional_positive_int(raw_profile.get("scale"), "scale"),
    )


def _matching_profiles(
    profiles: dict[HardwareProfile, dict[str, list[TunedConfigBySize]]],
    runtime_profile: HardwareProfile,
) -> list[tuple[int, dict[str, list[TunedConfigBySize]]]]:
    matches: list[tuple[int, dict[str, list[TunedConfigBySize]]]] = []
    for profile, configs_by_collective in profiles.items():
        specificity = _profile_match_specificity(profile, runtime_profile)
        if specificity is not None:
            matches.append((specificity, configs_by_collective))
    return sorted(matches, key=lambda item: item[0], reverse=True)


def _profile_match_specificity(profile: HardwareProfile, runtime_profile: HardwareProfile) -> int | None:
    specificity = 0
    if profile.sku is not None:
        if profile.sku != runtime_profile.sku:
            return None
        specificity += 1
    if profile.scale is not None:
        if profile.scale != runtime_profile.scale:
            return None
        specificity += 1
    return specificity


def _select_config(
    configs_by_collective: dict[str, list[TunedConfigBySize]], collective: str, message_size: int
) -> TunedConfig | None:
    configs = configs_by_collective.get(collective, [])
    if not configs:
        return None
    sizes = [item.message_size for item in configs]
    index = bisect_left(sizes, message_size)
    if index == len(sizes):
        return configs[-1].config
    if sizes[index] == message_size or index == 0:
        return configs[index].config
    return configs[index - 1].config


def _configs_by_collective_from_payload(payload: Any) -> dict[str, list[TunedConfigBySize]]:
    if not isinstance(payload, dict):
        raise ValueError("MSCCL++ tuned config collectives must be an object")

    result: dict[str, list[TunedConfigBySize]] = {}
    for collective, raw_entries in payload.items():
        if isinstance(raw_entries, dict):
            raw_entries = raw_entries.get("configs", [])
        if not isinstance(raw_entries, list):
            continue
        configs = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Invalid tuned config entry for {collective}: {raw_entry!r}")
            configs.append(
                TunedConfigBySize(
                    message_size=_parse_positive_int(raw_entry.get("message_size"), "message_size"),
                    config=TunedConfig(
                        algorithm=str(raw_entry["algorithm"]),
                        nblocks=_optional_int(raw_entry.get("nblocks")),
                        nthreads=_optional_int(raw_entry.get("nthreads")),
                        symmetric_memory=_optional_bool(raw_entry.get("symmetric_memory", False)),
                        time_us=_optional_float(raw_entry.get("time_us")),
                    ),
                )
            )
        result[str(collective)] = sorted(configs)
    return result


def _config_entry_payload(item: TunedConfigBySize) -> dict[str, Any]:
    payload: dict[str, Any] = {"message_size": item.message_size, "algorithm": item.config.algorithm}
    if item.config.nblocks is not None:
        payload["nblocks"] = item.config.nblocks
    if item.config.nthreads is not None:
        payload["nthreads"] = item.config.nthreads
    if item.config.symmetric_memory:
        payload["symmetric_memory"] = item.config.symmetric_memory
    if item.config.time_us is not None:
        payload["time_us"] = item.config.time_us
    return payload


def _format_tuned_config_json(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, indent=2)
    pattern = re.compile(
        r"(?m)^(?P<indent> +)\{\n"
        r'(?P<body>(?P=indent)  "message_size": [^\n]+,?\n(?:(?P=indent)  "[^"]+": [^\n]+,?\n)*)'
        r"(?P=indent)\}(?P<comma>,?)$"
    )

    def compact(match: re.Match[str]) -> str:
        body = " ".join(line.strip() for line in match.group("body").splitlines())
        return f"{match.group('indent')}{{{body}}}{match.group('comma')}"

    return pattern.sub(compact, text) + "\n"


def _optional_int(value: Any | None) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any | None) -> float | None:
    return None if value is None else float(value)


def _optional_positive_int(value: Any | None, name: str) -> int | None:
    return None if value is None else _parse_positive_int(value, name)


def _optional_bool(value: Any | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"Expected boolean value, got {value!r}")


def _parse_positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed
