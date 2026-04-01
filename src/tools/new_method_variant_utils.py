#!/usr/bin/env python3
"""Shared helpers for MultiCF, BoundaryCF, and the SpanCF family run variants."""

from __future__ import annotations

import re
from dataclasses import dataclass


MULTICF_RUN_RE = re.compile(
    r"_k(?P<k>[^_]+)_ag(?P<agg>[^_]+)_w(?P<weight>[^_]+)_t(?P<temp>[^_]+)"
    r"_(?:dOn|dOff)_(?:aOn|aOff)(?:_seed\d+)?$"
)
BOUNDARY_RUN_RE = re.compile(
    r"_lr(?P<local>[^_]+)_bm(?P<margin>[^_]+)"
    r"_(?:dOn|dOff)_(?:aOn|aOff)(?:_seed\d+)?$"
)
SPAN_TAIL_RE = re.compile(r"_(?:dOn|dOff)_(?:aOn|aOff)(?:_seed\d+)?$")
SPAN_OLD_BODY_RE = re.compile(r"m(?P<mode>[^_]+)_sw(?P<shared>[^_]+)_uw(?P<unique>[^_]+)")
SPAN_NEW_BODY_RE = re.compile(
    r"m(?P<mode>[^_]+)_asw(?P<alt_shared>[^_]+)_auw(?P<alt_unique>[^_]+)"
    r"_osw(?P<orig_shared>[^_]+)_ouw(?P<orig_unique>[^_]+)"
    r"(?:_dlt(?P<delta>[^_]+))?"
    r"(?:_lr(?P<local>[^_]+)_bm(?P<margin>[^_]+))?"
    r"(?:_sr(?P<sam_rho>[^_]+)_sad(?P<sam_adaptive>[^_]+))?"
    r"(?:_pct(?P<proj_threshold>[^_]+))?"
)

MULTICF_KEY_RE = re.compile(
    r"^multicf_(?:(?P<tag>m\d+)|k(?P<k>[^_]+)_ag(?P<agg>[^_]+)_w(?P<weight>[^_]+)_t(?P<temp>[^_]+))$"
)
BOUNDARY_KEY_RE = re.compile(
    r"^boundary_cf_(?:(?P<tag>b\d+)|lr(?P<local>[^_]+)_bm(?P<margin>[^_]+))$"
)

MULTICF_TAGS = {
    ("2", "wm", "rr", "0p7"): "m1",
    ("2", "wm", "rr", "0p5"): "m2",
    ("2", "wm", "rr", "1p0"): "m3",
    ("3", "wm", "rr", "0p7"): "m4",
    ("3", "wm", "rr", "0p5"): "m5",
    ("2", "m", "uni", "1p0"): "m6",
}
BOUNDARY_TAGS = {
    ("0p75", "0p5"): "b1",
    ("1p0", "0p5"): "b2",
    ("0p5", "0p5"): "b3",
    ("0p5", "1p0"): "b4",
    ("0p75", "1p0"): "b5",
    ("1p0", "1p0"): "b6",
}
SPAN_TAGS = {
    ("lc", "0p10", "1p25"): "s1",
    ("lc", "0p10", "1p0"): "s2",
    ("lc", "0p25", "1p25"): "s3",
    ("lc", "0p0", "1p0"): "s4",
    ("lc", "0p25", "1p0"): "s5",
    ("so", "0p10", "1p25"): "s6",
}

MULTICF_TOKENS_BY_TAG = {tag: spec for spec, tag in MULTICF_TAGS.items()}
BOUNDARY_TOKENS_BY_TAG = {tag: spec for spec, tag in BOUNDARY_TAGS.items()}
SPAN_TOKENS_BY_TAG = {tag: spec for spec, tag in SPAN_TAGS.items()}

MULTICF_AGG_DISPLAY = {
    "wm": "weighted_mean",
    "m": "mean",
    "t1": "top1",
}
MULTICF_WEIGHT_DISPLAY = {
    "rr": "rerank",
    "uni": "uniform",
}
SPAN_MODE_DISPLAY = {
    "lc": "lcs",
    "so": "set_overlap",
}
BOOL_TOKEN_DISPLAY = {
    "t": "true",
    "f": "false",
}
SPAN_VARIANT_FAMILIES = {
    "span_cf": "SpanCF",
    "span_cf_samnpo": "SpanCF-SAMNPO",
    "span_cf_simnpo": "SpanCF-SimNPO",
    "span_cf_local_retain": "SpanCF-LocalRetain",
    "span_cf_simnpo_local_retain": "SpanCF-SimNPO-LocalRetain",
    "span_cf_simnpo_sam": "SpanCF-SimNPO-SAM",
    "span_cf_simnpo_projected": "SpanCF-SimNPO-Projected",
}

VARIANT_ALGORITHM_ORDER = {
    "multicf": 0,
    "boundary_cf": 1,
    "span_cf": 2,
    "span_cf_samnpo": 3,
    "span_cf_simnpo": 4,
    "span_cf_local_retain": 5,
    "span_cf_simnpo_local_retain": 6,
    "span_cf_simnpo_sam": 7,
    "span_cf_simnpo_projected": 8,
}


@dataclass(frozen=True)
class MethodVariantInfo:
    method_key: str
    display_name: str
    algorithm: str
    order_index: int


def format_numeric_token(token: str) -> str:
    return token.replace("p", ".")


def base_variant_algorithm(method_key: str) -> str | None:
    for prefix in sorted(SPAN_VARIANT_FAMILIES, key=len, reverse=True):
        if method_key == prefix or method_key.startswith(f"{prefix}_"):
            return prefix
    if method_key.startswith("multicf_"):
        return "multicf"
    if method_key.startswith("boundary_cf_"):
        return "boundary_cf"
    return None


def _multicf_info(*, tag: str | None, k: str, agg: str, weight: str, temp: str) -> MethodVariantInfo:
    method_key = f"multicf_{tag}" if tag is not None else f"multicf_k{k}_ag{agg}_w{weight}_t{temp}"
    display_prefix = f"MultiCF {tag.upper()}" if tag is not None else "MultiCF"
    display_name = (
        f"{display_prefix} (k={k} agg={MULTICF_AGG_DISPLAY.get(agg, agg)} "
        f"weight={MULTICF_WEIGHT_DISPLAY.get(weight, weight)} temp={format_numeric_token(temp)})"
    )
    order_index = int(tag[1:]) if tag is not None and tag[1:].isdigit() else 999
    return MethodVariantInfo(
        method_key=method_key,
        display_name=display_name,
        algorithm="multicf",
        order_index=order_index,
    )


def _boundary_info(*, tag: str | None, local: str, margin: str) -> MethodVariantInfo:
    method_key = f"boundary_cf_{tag}" if tag is not None else f"boundary_cf_lr{local}_bm{margin}"
    display_prefix = f"BoundaryCF {tag.upper()}" if tag is not None else "BoundaryCF"
    display_name = (
        f"{display_prefix} (local_retain={format_numeric_token(local)} "
        f"margin={format_numeric_token(margin)})"
    )
    order_index = int(tag[1:]) if tag is not None and tag[1:].isdigit() else 999
    return MethodVariantInfo(
        method_key=method_key,
        display_name=display_name,
        algorithm="boundary_cf",
        order_index=order_index,
    )


def _span_info(
    *,
    method_name: str,
    tag: str | None,
    mode: str,
    alt_shared: str,
    alt_unique: str,
    orig_shared: str,
    orig_unique: str,
    delta: str | None = None,
    local: str | None = None,
    margin: str | None = None,
    sam_rho: str | None = None,
    sam_adaptive: str | None = None,
    proj_threshold: str | None = None,
) -> MethodVariantInfo:
    prefix = SPAN_VARIANT_FAMILIES.get(method_name, method_name)
    order_index = int(tag[1:]) if tag is not None and tag[1:].isdigit() else 999

    if tag is not None:
        method_key = f"{method_name}_{tag}"
        display_name = f"{prefix} {tag.upper()}"
        return MethodVariantInfo(
            method_key=method_key,
            display_name=display_name,
            algorithm=method_name,
            order_index=order_index,
        )

    parts = [
        f"m{mode}",
        f"asw{alt_shared}",
        f"auw{alt_unique}",
        f"osw{orig_shared}",
        f"ouw{orig_unique}",
    ]
    display_parts = [
        f"mode={SPAN_MODE_DISPLAY.get(mode, mode)}",
        f"alt_shared={format_numeric_token(alt_shared)}",
        f"alt_unique={format_numeric_token(alt_unique)}",
        f"orig_shared={format_numeric_token(orig_shared)}",
        f"orig_unique={format_numeric_token(orig_unique)}",
    ]
    if delta is not None:
        parts.append(f"dlt{delta}")
        display_parts.append(f"delta={format_numeric_token(delta)}")
    if local is not None and margin is not None:
        parts.extend((f"lr{local}", f"bm{margin}"))
        display_parts.append(f"local_retain={format_numeric_token(local)}")
        display_parts.append(f"margin={format_numeric_token(margin)}")
    if sam_rho is not None and sam_adaptive is not None:
        parts.extend((f"sr{sam_rho}", f"sad{sam_adaptive}"))
        display_parts.append(f"sam_rho={format_numeric_token(sam_rho)}")
        display_parts.append(
            f"sam_adaptive={BOOL_TOKEN_DISPLAY.get(sam_adaptive, sam_adaptive)}"
        )
    if proj_threshold is not None:
        parts.append(f"pct{proj_threshold}")
        display_parts.append(f"proj_threshold={format_numeric_token(proj_threshold)}")

    return MethodVariantInfo(
        method_key=f"{method_name}_{'_'.join(parts)}",
        display_name=f"{prefix} ({' '.join(display_parts)})",
        algorithm=method_name,
        order_index=order_index,
    )


def _parse_span_tokens_from_text(text: str) -> dict[str, str] | None:
    legacy_match = SPAN_OLD_BODY_RE.search(text)
    if legacy_match is not None:
        mode = legacy_match.group("mode")
        shared = legacy_match.group("shared")
        unique = legacy_match.group("unique")
        return {
            "mode": mode,
            "alt_shared": shared,
            "alt_unique": unique,
            "orig_shared": shared,
            "orig_unique": unique,
        }

    modern_match = SPAN_NEW_BODY_RE.search(text)
    if modern_match is not None:
        payload = {
            "mode": modern_match.group("mode"),
            "alt_shared": modern_match.group("alt_shared"),
            "alt_unique": modern_match.group("alt_unique"),
            "orig_shared": modern_match.group("orig_shared"),
            "orig_unique": modern_match.group("orig_unique"),
        }
        for key in ("delta", "local", "margin", "sam_rho", "sam_adaptive", "proj_threshold"):
            value = modern_match.groupdict().get(key)
            if value is not None:
                payload[key] = value
        return payload
    return None


def extract_new_method_variant(run_name: str, method_name: str) -> MethodVariantInfo | None:
    if method_name == "multicf":
        match = MULTICF_RUN_RE.search(run_name)
        if match is None:
            return None
        tokens = (
            match.group("k"),
            match.group("agg"),
            match.group("weight"),
            match.group("temp"),
        )
        return _multicf_info(
            tag=MULTICF_TAGS.get(tokens),
            k=tokens[0],
            agg=tokens[1],
            weight=tokens[2],
            temp=tokens[3],
        )

    if method_name == "boundary_cf":
        match = BOUNDARY_RUN_RE.search(run_name)
        if match is None:
            return None
        tokens = (match.group("local"), match.group("margin"))
        return _boundary_info(
            tag=BOUNDARY_TAGS.get(tokens),
            local=tokens[0],
            margin=tokens[1],
        )

    if method_name in SPAN_VARIANT_FAMILIES:
        if SPAN_TAIL_RE.search(run_name) is None:
            return None
        tokens = _parse_span_tokens_from_text(run_name)
        if tokens is None:
            return None

        tag = None
        if method_name == "span_cf" and (
            tokens["alt_shared"] == tokens["orig_shared"]
            and tokens["alt_unique"] == tokens["orig_unique"]
        ):
            tag = SPAN_TAGS.get(
                (tokens["mode"], tokens["alt_shared"], tokens["alt_unique"])
            )
        return _span_info(method_name=method_name, tag=tag, **tokens)

    return None


def _span_info_from_method_key(method_key: str) -> MethodVariantInfo | None:
    method_name = base_variant_algorithm(method_key)
    if method_name not in SPAN_VARIANT_FAMILIES:
        return None
    if method_key == method_name:
        return None

    remainder = method_key[len(method_name) + 1 :]
    if method_name == "span_cf" and remainder in SPAN_TOKENS_BY_TAG:
        mode, shared, unique = SPAN_TOKENS_BY_TAG[remainder]
        return _span_info(
            method_name=method_name,
            tag=remainder,
            mode=mode,
            alt_shared=shared,
            alt_unique=unique,
            orig_shared=shared,
            orig_unique=unique,
        )

    tokens = _parse_span_tokens_from_text(remainder)
    if tokens is None:
        return None
    return _span_info(method_name=method_name, tag=None, **tokens)


def variant_info_from_method_key(method_key: str) -> MethodVariantInfo | None:
    match = MULTICF_KEY_RE.fullmatch(method_key)
    if match is not None:
        tag = match.group("tag")
        if tag is not None:
            tokens = MULTICF_TOKENS_BY_TAG.get(tag)
            if tokens is not None:
                return _multicf_info(
                    tag=tag,
                    k=tokens[0],
                    agg=tokens[1],
                    weight=tokens[2],
                    temp=tokens[3],
                )
            return MethodVariantInfo(
                method_key=method_key,
                display_name=f"MultiCF {tag.upper()}",
                algorithm="multicf",
                order_index=999,
            )
        return _multicf_info(
            tag=None,
            k=match.group("k"),
            agg=match.group("agg"),
            weight=match.group("weight"),
            temp=match.group("temp"),
        )

    match = BOUNDARY_KEY_RE.fullmatch(method_key)
    if match is not None:
        tag = match.group("tag")
        if tag is not None:
            tokens = BOUNDARY_TOKENS_BY_TAG.get(tag)
            if tokens is not None:
                return _boundary_info(tag=tag, local=tokens[0], margin=tokens[1])
            return MethodVariantInfo(
                method_key=method_key,
                display_name=f"BoundaryCF {tag.upper()}",
                algorithm="boundary_cf",
                order_index=999,
            )
        return _boundary_info(
            tag=None,
            local=match.group("local"),
            margin=match.group("margin"),
        )

    return _span_info_from_method_key(method_key)


def variant_sort_key(method_key: str) -> tuple[int, int, str] | None:
    info = variant_info_from_method_key(method_key)
    if info is None:
        return None
    return (VARIANT_ALGORITHM_ORDER[info.algorithm], info.order_index, info.method_key)
