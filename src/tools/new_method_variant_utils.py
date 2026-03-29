#!/usr/bin/env python3
"""Shared helpers for MultiCF / BoundaryCF / SpanCF run variants."""

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
SPAN_RUN_RE = re.compile(
    r"_m(?P<mode>[^_]+)_sw(?P<shared>[^_]+)_uw(?P<unique>[^_]+)"
    r"_(?:dOn|dOff)_(?:aOn|aOff)(?:_seed\d+)?$"
)

MULTICF_KEY_RE = re.compile(
    r"^multicf_(?:(?P<tag>m\d+)|k(?P<k>[^_]+)_ag(?P<agg>[^_]+)_w(?P<weight>[^_]+)_t(?P<temp>[^_]+))$"
)
BOUNDARY_KEY_RE = re.compile(
    r"^boundary_cf_(?:(?P<tag>b\d+)|lr(?P<local>[^_]+)_bm(?P<margin>[^_]+))$"
)
SPAN_KEY_RE = re.compile(
    r"^span_cf_(?:(?P<tag>s\d+)|m(?P<mode>[^_]+)_sw(?P<shared>[^_]+)_uw(?P<unique>[^_]+))$"
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

VARIANT_ALGORITHM_ORDER = {
    "multicf": 0,
    "boundary_cf": 1,
    "span_cf": 2,
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
    if method_key.startswith("multicf_"):
        return "multicf"
    if method_key.startswith("boundary_cf_"):
        return "boundary_cf"
    if method_key.startswith("span_cf_"):
        return "span_cf"
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


def _span_info(*, tag: str | None, mode: str, shared: str, unique: str) -> MethodVariantInfo:
    method_key = f"span_cf_{tag}" if tag is not None else f"span_cf_m{mode}_sw{shared}_uw{unique}"
    display_prefix = f"SpanCF {tag.upper()}" if tag is not None else "SpanCF"
    display_name = (
        f"{display_prefix} (mode={SPAN_MODE_DISPLAY.get(mode, mode)} "
        f"shared={format_numeric_token(shared)} unique={format_numeric_token(unique)})"
    )
    order_index = int(tag[1:]) if tag is not None and tag[1:].isdigit() else 999
    return MethodVariantInfo(
        method_key=method_key,
        display_name=display_name,
        algorithm="span_cf",
        order_index=order_index,
    )


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
        return _multicf_info(tag=MULTICF_TAGS.get(tokens), k=tokens[0], agg=tokens[1], weight=tokens[2], temp=tokens[3])

    if method_name == "boundary_cf":
        match = BOUNDARY_RUN_RE.search(run_name)
        if match is None:
            return None
        tokens = (match.group("local"), match.group("margin"))
        return _boundary_info(tag=BOUNDARY_TAGS.get(tokens), local=tokens[0], margin=tokens[1])

    if method_name == "span_cf":
        match = SPAN_RUN_RE.search(run_name)
        if match is None:
            return None
        tokens = (match.group("mode"), match.group("shared"), match.group("unique"))
        return _span_info(tag=SPAN_TAGS.get(tokens), mode=tokens[0], shared=tokens[1], unique=tokens[2])

    return None


def variant_info_from_method_key(method_key: str) -> MethodVariantInfo | None:
    match = MULTICF_KEY_RE.fullmatch(method_key)
    if match is not None:
        tag = match.group("tag")
        if tag is not None:
            tokens = MULTICF_TOKENS_BY_TAG.get(tag)
            if tokens is not None:
                return _multicf_info(tag=tag, k=tokens[0], agg=tokens[1], weight=tokens[2], temp=tokens[3])
            return MethodVariantInfo(method_key=method_key, display_name=f"MultiCF {tag.upper()}", algorithm="multicf", order_index=999)
        return _multicf_info(tag=None, k=match.group("k"), agg=match.group("agg"), weight=match.group("weight"), temp=match.group("temp"))

    match = BOUNDARY_KEY_RE.fullmatch(method_key)
    if match is not None:
        tag = match.group("tag")
        if tag is not None:
            tokens = BOUNDARY_TOKENS_BY_TAG.get(tag)
            if tokens is not None:
                return _boundary_info(tag=tag, local=tokens[0], margin=tokens[1])
            return MethodVariantInfo(method_key=method_key, display_name=f"BoundaryCF {tag.upper()}", algorithm="boundary_cf", order_index=999)
        return _boundary_info(tag=None, local=match.group("local"), margin=match.group("margin"))

    match = SPAN_KEY_RE.fullmatch(method_key)
    if match is not None:
        tag = match.group("tag")
        if tag is not None:
            tokens = SPAN_TOKENS_BY_TAG.get(tag)
            if tokens is not None:
                return _span_info(tag=tag, mode=tokens[0], shared=tokens[1], unique=tokens[2])
            return MethodVariantInfo(method_key=method_key, display_name=f"SpanCF {tag.upper()}", algorithm="span_cf", order_index=999)
        return _span_info(tag=None, mode=match.group("mode"), shared=match.group("shared"), unique=match.group("unique"))

    return None


def variant_sort_key(method_key: str) -> tuple[int, int, str] | None:
    info = variant_info_from_method_key(method_key)
    if info is None:
        return None
    return (VARIANT_ALGORITHM_ORDER[info.algorithm], info.order_index, info.method_key)
