#!/usr/bin/env python3
"""Build combined LaTeX tables from structured-saves trees."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from new_method_variant_utils import (
    base_variant_algorithm,
    variant_info_from_method_key,
    variant_sort_key,
)

SPLITS = ["duet_rare", "duet_popular", "duet_merged", "rwku"]
LRS = ["1e-4", "5e-5"]
EPOCH_SPECS = [("2.0", "2"), ("5.0", "5")]
FIXED_METRICS = [
    ("forget_qa_rouge", "F"),
    ("holdout_qa_rouge", "H"),
    ("forget_wrong_gen_rate", "FW"),
    ("holdout_wrong_gen_rate", "HW"),
    ("forget_qa_cos_sim", "FC"),
    ("holdout_qa_cos_sim", "HC"),
    ("utility_avg", "U"),
]
METRIC_DIRECTION = {
    "forget_qa_rouge": r"$\downarrow$",
    "holdout_qa_rouge": r"$\uparrow$",
    "forget_wrong_gen_rate": r"$\downarrow$",
    "holdout_wrong_gen_rate": r"$\downarrow$",
    "forget_qa_cos_sim": r"$\downarrow$",
    "holdout_qa_cos_sim": r"$\uparrow$",
    "utility_avg": r"$\uparrow$",
}
WRONG_GENERATION_METRIC_NAMES = {"forget_wrong_gen_rate", "holdout_wrong_gen_rate"}
WRONG_GENERATION_METRIC_MAP = {
    "forget_qa_rouge": "forget_wrong_gen_rate",
    "holdout_qa_rouge": "holdout_wrong_gen_rate",
}
WRONG_GENERATION_METHOD_MAP = {
    "dual_cf_full": "full",
    "dual_cf_d_only": "d_only",
    "dual_cf_a_only": "a_only",
    "dpo_cf": "dpo",
    "multicf": "multicf",
    "boundary_cf": "boundary_cf",
    "span_cf": "span_cf",
    "span_cf_samnpo": "span_cf_samnpo",
    "span_cf_simnpo": "span_cf_simnpo",
    "span_cf_local_retain": "span_cf_local_retain",
    "span_cf_simnpo_local_retain": "span_cf_simnpo_local_retain",
    "span_cf_simnpo_sam": "span_cf_simnpo_sam",
    "span_cf_simnpo_projected": "span_cf_simnpo_projected",
    "ga": "ga",
    "ada_pop": "ada_pop",
    "npo": "npo",
    "npo_sam": "npo_sam",
    "loku": "loku",
    "simnpo": "simnpo",
}
METRIC_LEGEND_LABELS = {
    "F": "forget ROUGE",
    "H": "holdout ROUGE",
    "FW": "forget wrong-generation rate",
    "HW": "holdout wrong-generation rate",
    "FC": "forget cosine similarity",
    "HC": "holdout cosine similarity",
    "U": "utility_avg",
    "M": "MMLU-Pro",
    "T": "TruthfulQA",
    "W": "Winogrande",
    "A": "ARC",
}
UTILITY_METRIC_RE = re.compile(r"^(mmlu_pro|truthfulqa_bin|winogrande|arc)_\d+_acc$")
UTILITY_METRIC_ORDER = {
    "mmlu_pro": 0,
    "truthfulqa_bin": 1,
    "winogrande": 2,
    "arc": 3,
}
UTILITY_METRIC_ABBREV = {
    "mmlu_pro": "M",
    "truthfulqa_bin": "T",
    "winogrande": "W",
    "arc": "A",
}
COMBINED_ROW_SPECS = [
    ("old", "full", "Full-old", "blue!10"),
    ("old", "d_only", "d-only-old", "orange!10"),
    ("old", "a_only", "a-only-old", "green!10"),
    ("old", "dpo", "DPO-old", "gray!8"),
    ("old", "ga", "GA", "gray!8"),
    ("old", "ada_pop", "AdaPop", "gray!8"),
    ("old", "npo", "NPO", "gray!8"),
    ("old", "npo_sam", "NPO-SAM", "gray!8"),
    ("old", "loku", "LoKU", "gray!8"),
    ("new", "full", "Full-new", "blue!20"),
    ("new", "d_only", "d-only-new", "orange!20"),
    ("new", "a_only", "a-only-new", "green!20"),
    ("new", "dpo", "DPO-new", "gray!15"),
    ("new", "multicf", "MultiCF", "teal!12"),
    ("new", "boundary_cf", "BoundaryCF", "cyan!12"),
    ("new", "span_cf", "SpanCF", "yellow!12"),
    ("new", "span_cf_samnpo", "SpanCF-SAMNPO", "orange!10"),
    ("new", "span_cf_simnpo", "SpanCF-SimNPO", "yellow!18"),
    ("new", "span_cf_local_retain", "SpanCF-LocalRetain", "lime!12"),
    ("new", "span_cf_simnpo_local_retain", "SpanCF-SimNPO-LocalRetain", "lime!18"),
    ("new", "span_cf_simnpo_sam", "SpanCF-SimNPO-SAM", "orange!12"),
    ("new", "span_cf_simnpo_projected", "SpanCF-SimNPO-Projected", "red!12"),
]
SIMNPO_ROW_SPEC = ("simnpo", "simnpo", "SimNPO", "red!12")
COMBINED_SIMPLECE_METHODS = [
    "simple_ce_cf1_ret1_gamma0",
    "simple_ce_cf0p5_ret1_gamma0",
]
SPLIT_LABELS = {
    "duet_rare": "DUET Rare",
    "duet_popular": "DUET Popular",
    "duet_merged": "DUET Merged",
    "rwku": "RWKU",
}
SIMPLE_CE_RE = re.compile(
    r"^simple_ce(?:_cf(?P<cf>[^_]+))?(?:_ret(?P<ret>[^_]+))?(?:_gamma(?P<gamma>[^_]+))?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-root",
        type=Path,
        help="Path to the old structured-saves directory (for example metrics-ep5-all-v2/structured-saves).",
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        help="Path to the new structured-saves directory (for example metrics-ep5-dualfc-new_cf/structured-saves).",
    )
    parser.add_argument(
        "--variant-root",
        type=Path,
        action="append",
        help=(
            "Optional structured-saves directory for a variant-only table build. "
            "Can be passed multiple times to combine methods from multiple structured-saves roots "
            "(for example metrics-new/ep5-dualfc-v2_5/structured-saves-avg)."
        ),
    )
    parser.add_argument(
        "--variant-algorithm",
        action="append",
        help=(
            "Optional variant algorithm family filter. Can be repeated. "
            "Examples: span_cf_samnpo, span_cf_simnpo_sam."
        ),
    )
    parser.add_argument(
        "--variant-method-key",
        action="append",
        help=(
            "Optional exact variant method key filter. Can be repeated. "
            "Examples: span_cf_s2, span_cf_s4."
        ),
    )
    parser.add_argument(
        "--variant-display",
        choices=("full", "compact"),
        default="full",
        help=(
            "How to render variant row labels in variant-only tables. "
            "`full` keeps parameter details, `compact` keeps only the method family/tag label."
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to the output .txt file that will contain all combined LaTeX tables.",
    )
    parser.add_argument(
        "--output-slides-tex",
        type=Path,
        help="Optional path to a Beamer .tex file with one slide per split/LR/epoch table.",
    )
    parser.add_argument(
        "--wrong-generations-root",
        type=Path,
        help=(
            "Optional path to analyze_wrong_generations.py outputs that contain "
            "method_stage_summary.tsv for wrong-generation rates."
        ),
    )
    parser.add_argument(
        "--simnpo-root",
        type=Path,
        help="Optional path to a structured-saves directory that contains simnpo rows.",
    )
    parser.add_argument(
        "--simplece-new-root",
        type=Path,
        help=(
            "Optional path to a structured-saves directory that contains the newer simple_ce rows. "
            "Defaults to --simnpo-root when omitted."
        ),
    )
    parser.add_argument(
        "--simplece-old-root",
        type=Path,
        help="Optional path to a structured-saves directory that contains the older simple_ce rows to compare against --simnpo-root.",
    )
    parser.add_argument(
        "--output-simplece-file",
        type=Path,
        help="Optional path to a .txt file that will contain SimpleCE-only LaTeX tables.",
    )
    parser.add_argument(
        "--output-simplece-slides-tex",
        type=Path,
        help="Optional path to a Beamer .tex file with one slide per SimpleCE-only table.",
    )
    return parser.parse_args()


def load_metric_rows(table_path: Path, *, missing_ok: bool = False) -> dict[str, dict[str, str]]:
    if not table_path.exists():
        if missing_ok:
            return {}
        raise FileNotFoundError(f"Missing metric table: {table_path}")

    rows: dict[str, dict[str, str]] = {}
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            method = row.get("method")
            if not method:
                continue
            rows[method] = row
    return rows


def utility_metric_sort_key(metric_name: str) -> tuple[int, str]:
    match = UTILITY_METRIC_RE.fullmatch(metric_name)
    if match is None:
        return (len(UTILITY_METRIC_ORDER), metric_name)
    return (UTILITY_METRIC_ORDER[match.group(1)], metric_name)


def discover_metrics(
    roots: list[Path],
    *,
    include_wrong_generation_metrics: bool,
) -> list[tuple[str, str]]:
    utility_metric_names: set[str] = set()
    for root in roots:
        for split in SPLITS:
            for lr in LRS:
                split_lr_dir = root / split / lr
                if not split_lr_dir.exists():
                    continue
                for table_path in split_lr_dir.glob("*_acc.tsv"):
                    metric_name = table_path.stem
                    if UTILITY_METRIC_RE.fullmatch(metric_name):
                        utility_metric_names.add(metric_name)

    metrics = [
        metric_spec
        for metric_spec in FIXED_METRICS
        if include_wrong_generation_metrics or metric_spec[0] not in WRONG_GENERATION_METRIC_NAMES
    ]
    for metric_name in sorted(utility_metric_names, key=utility_metric_sort_key):
        match = UTILITY_METRIC_RE.fullmatch(metric_name)
        assert match is not None
        metrics.append((metric_name, UTILITY_METRIC_ABBREV[match.group(1)]))
    return metrics


def normalize_epoch_column(raw_value: str | None) -> str | None:
    if raw_value in {None, ""}:
        return None
    return f"{float(raw_value):.1f}"


def infer_wrong_generation_split(benchmark: str, forget_split: str) -> str | None:
    if benchmark == "rwku":
        return "rwku"
    if forget_split == "city_forget_rare_5":
        return "duet_rare"
    if forget_split == "city_forget_popular_5":
        return "duet_popular"
    if forget_split in {"city_forget_5", "city_forget_rare_5+city_forget_popular_5"}:
        return "duet_merged"
    return None


def load_wrong_generation_index(
    wrong_generations_root: Path,
) -> tuple[dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, str]]]]]], set[str]]:
    summary_path = wrong_generations_root / "method_stage_summary.tsv"

    index: dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, str]]]]]] = {}
    labels: set[str] = set()
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            metric_name = WRONG_GENERATION_METRIC_MAP.get(str(row.get("metric_name") or ""))
            if metric_name is None:
                continue
            split = infer_wrong_generation_split(
                str(row.get("benchmark") or ""),
                str(row.get("forget_split") or ""),
            )
            epoch_column = normalize_epoch_column(row.get("epoch"))
            wrong_pct = row.get("wrong_pct")
            if split is None or epoch_column is None or wrong_pct in {None, ""}:
                continue

            label = str(row.get("input_root_label") or "")
            if not label:
                continue
            labels.add(label)
            split_map = index.setdefault(label, {}).setdefault(split, {}).setdefault(
                str(row.get("lr") or ""),
                {},
            )
            method_key = str(row.get("method_key") or "")
            method_map = split_map.setdefault(epoch_column, {}).setdefault(method_key, {})
            method_map[metric_name] = format(float(wrong_pct) / 100.0, ".12g")

    return index, labels


def resolve_wrong_generation_label(root: Path, available_labels: set[str]) -> str | None:
    for candidate in [root.name, *(parent.name for parent in root.parents)]:
        if candidate and candidate in available_labels:
            return candidate
    return None


def resolve_wrong_generation_methods(
    method_key: str,
    source_methods: set[str],
) -> list[str]:
    mapped_method = WRONG_GENERATION_METHOD_MAP.get(method_key)
    if mapped_method is not None:
        return [mapped_method] if mapped_method in source_methods else []

    if method_key in source_methods:
        return [method_key]

    if method_key != "simple_ce":
        return []

    simple_ce_methods = sorted(
        (method_name for method_name in source_methods if method_name.startswith("simple_ce")),
        key=simple_ce_sort_key,
    )
    if len(simple_ce_methods) == 1:
        return simple_ce_methods
    return []


def load_wrong_generation_rows(
    wrong_generation_index: dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, str]]]]]],
    wrong_generation_label: str | None,
    split: str,
    lr: str,
    metric_name: str,
    source_methods: set[str],
) -> dict[str, dict[str, str]]:
    if wrong_generation_label is None:
        return {}

    rows: dict[str, dict[str, str]] = {}
    by_epoch = (
        wrong_generation_index.get(wrong_generation_label, {})
        .get(split, {})
        .get(lr, {})
    )
    for epoch_column, method_entries in by_epoch.items():
        for method_key, metric_values in method_entries.items():
            raw_value = metric_values.get(metric_name)
            if raw_value is None:
                continue
            for resolved_method in resolve_wrong_generation_methods(method_key, source_methods):
                row = rows.setdefault(resolved_method, {"method": resolved_method})
                row[epoch_column] = raw_value
    return rows


def load_table_bundle(
    root: Path,
    split: str,
    lr: str,
    metrics: list[tuple[str, str]],
    *,
    wrong_generation_index: dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, str]]]]]]
    | None = None,
    wrong_generation_label: str | None = None,
) -> dict[str, dict[str, dict[str, str]]]:
    bundle: dict[str, dict[str, dict[str, str]]] = {}
    fixed_metric_names = {
        metric_name
        for metric_name, _metric_abbrev in FIXED_METRICS
        if metric_name not in WRONG_GENERATION_METRIC_NAMES
    }
    for metric_name, _metric_abbrev in metrics:
        if metric_name in WRONG_GENERATION_METRIC_NAMES:
            continue
        bundle[metric_name] = load_metric_rows(
            root / split / lr / f"{metric_name}.tsv",
            missing_ok=metric_name not in fixed_metric_names,
        )

    source_methods = set(bundle.get("forget_qa_rouge", {}))
    for metric_name, _metric_abbrev in metrics:
        if metric_name not in WRONG_GENERATION_METRIC_NAMES:
            continue
        direct_rows = load_metric_rows(
            root / split / lr / f"{metric_name}.tsv",
            missing_ok=True,
        )
        if direct_rows:
            bundle[metric_name] = direct_rows
            continue
        bundle[metric_name] = load_wrong_generation_rows(
            wrong_generation_index or {},
            wrong_generation_label,
            split,
            lr,
            metric_name,
            source_methods,
        )
    return bundle


def format_percent(raw_value: str | None) -> str:
    if raw_value in {None, ""}:
        return "--"
    return f"{float(raw_value) * 100.0:.1f}"


def escape_latex(text: str) -> str:
    return text.replace("_", r"\_")


def build_header_cells(metrics: list[tuple[str, str]]) -> list[str]:
    cells = ["Method"]
    for metric_name, metric_abbrev in metrics:
        direction = METRIC_DIRECTION.get(metric_name, r"$\uparrow$")
        cells.append(f"{metric_abbrev}{direction}")
    return cells


def build_direction_text(metrics: list[tuple[str, str]]) -> str:
    lower = [metric_abbrev for metric_name, metric_abbrev in metrics if METRIC_DIRECTION.get(metric_name) == r"$\downarrow$"]
    higher = [
        metric_abbrev
        for metric_name, metric_abbrev in metrics
        if METRIC_DIRECTION.get(metric_name, r"$\uparrow$") == r"$\uparrow$"
    ]
    if lower and higher:
        return (
            rf"{{\tiny Values are percentages. {' / '.join(lower)} are lower-is-better; "
            rf"{' / '.join(higher)} are higher-is-better.}}"
        )
    if lower:
        return rf"{{\tiny Values are percentages. {' / '.join(lower)} are lower-is-better.}}"
    if higher:
        return rf"{{\tiny Values are percentages. {' / '.join(higher)} are higher-is-better.}}"
    return r"{\tiny Values are percentages.}"


def build_metric_legend(metrics: list[tuple[str, str]]) -> str:
    legend_parts: list[str] = []
    seen_abbrevs: set[str] = set()
    for _metric_name, metric_abbrev in metrics:
        if metric_abbrev in seen_abbrevs:
            continue
        label = METRIC_LEGEND_LABELS.get(metric_abbrev)
        if label is None:
            continue
        legend_parts.append(f"{metric_abbrev} = {label.replace('_', r'\_')}")
        seen_abbrevs.add(metric_abbrev)
    return ",\\ ".join(legend_parts)


def build_row_cells(
    epoch_column: str,
    bundles: dict[str, dict[str, dict[str, dict[str, str]]]],
    row_specs: list[tuple[str, str, str, str]],
    metrics: list[tuple[str, str]],
) -> list[str]:
    lines: list[str] = []
    for source_name, source_method, display_name, color in row_specs:
        source_bundle = bundles[source_name]
        value_cells = [display_name]
        for metric_name, _metric_abbrev in metrics:
            method_row = source_bundle[metric_name].get(source_method)
            raw_value = None if method_row is None else method_row.get(epoch_column)
            value_cells.append(format_percent(raw_value))

        if color:
            lines.append(rf"\rowcolor{{{color}}}")
        lines.append(" & ".join(escape_latex(cell) for cell in value_cells) + r" \\")
    return lines


def build_table(
    *,
    caption: str,
    label: str,
    epoch_column: str,
    bundles: dict[str, dict[str, dict[str, dict[str, str]]]],
    row_specs: list[tuple[str, str, str, str]],
    metrics: list[tuple[str, str]],
) -> str:
    header_cells = build_header_cells(metrics)
    row_lines = build_row_cells(epoch_column, bundles, row_specs, metrics)
    metric_count = len(metrics)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{l*{{{metric_count}}}{{r}}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
        *row_lines,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def build_slide_frame(
    *,
    frame_title: str,
    epoch_column: str,
    bundles: dict[str, dict[str, dict[str, dict[str, str]]]],
    row_specs: list[tuple[str, str, str, str]],
    metrics: list[tuple[str, str]],
    compact: bool = False,
) -> str:
    header_cells = build_header_cells(metrics)
    row_lines = build_row_cells(epoch_column, bundles, row_specs, metrics)
    metric_count = len(metrics)

    table_font = r"\scriptsize"
    tabcolsep = "3.6pt"
    arraystretch = "1.05"
    max_totalheight = "0.78\\textheight"
    top_vspace = "0.35em"
    if compact:
        table_font = r"\fontsize{6.7}{7.4}\selectfont"
        tabcolsep = "3.3pt"
        arraystretch = "1.00"
        max_totalheight = "0.75\\textheight"
        top_vspace = "0.20em"

    lines = [
        r"\begin{frame}[t]",
        rf"\frametitle{{{frame_title}}}",
        build_direction_text(metrics),
        r"",
        rf"{{\tiny {build_metric_legend(metrics)}.}}",
        r"",
        rf"\vspace{{{top_vspace}}}",
        r"\centering",
        table_font,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}",
        rf"\begin{{adjustbox}}{{max width=\textwidth,max totalheight={max_totalheight},center}}",
        rf"\begin{{tabular}}{{l*{{{metric_count}}}{{r}}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
        *row_lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{frame}",
    ]
    return "\n".join(lines)


def build_slides_tex(
    *,
    title: str,
    subtitle: str,
    frames: list[str],
    neutral_theme: bool = False,
) -> str:
    lines = [
        r"\PassOptionsToPackage{table}{xcolor}",
        r"\documentclass[aspectratio=169,11pt]{beamer}",
        r"",
        r"\usetheme{Madrid}",
        r"\useinnertheme{rounded}",
        r"\setbeamertemplate{blocks}[rounded][shadow=false]",
        r"\setbeamertemplate{navigation symbols}{}",
        r"\setbeamertemplate{footline}[frame number]",
        r"\setbeamersize{text margin left=6mm, text margin right=6mm}",
        r"",
        r"\usepackage{iftex}",
        r"\ifPDFTeX",
        r"  \usepackage[utf8]{inputenc}",
        r"  \usepackage[T2A]{fontenc}",
        r"  \usepackage[english]{babel}",
        r"  \usepackage{paratype}",
        r"\else",
        r"  \usepackage{fontspec}",
        r"  \usepackage[english]{babel}",
        r"  \defaultfontfeatures{Ligatures=TeX}",
        r"  \IfFontExistsTF{PT Serif}{",
        r"    \setmainfont{PT Serif}",
        r"    \setsansfont{PT Sans}",
        r"    \setmonofont{PT Mono}",
        r"  }{",
        r"    \setmainfont{DejaVu Serif}",
        r"    \setsansfont{DejaVu Sans}",
        r"    \setmonofont{DejaVu Sans Mono}",
        r"  }",
        r"\fi",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{xcolor}",
        r"\usepackage{booktabs,array,adjustbox,tabularx,multirow}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=black,citecolor=black}",
        r"\ifPDFTeX",
        r"  \pdfsuppresswarningpagegroup=1",
        r"\fi",
        r"",
        r"\ifPDFTeX",
        r"  \DeclareUnicodeCharacter{202F}{\,}",
        r"  \DeclareUnicodeCharacter{2013}{-}",
        r"  \DeclareUnicodeCharacter{2014}{-}",
        r"  \DeclareUnicodeCharacter{2011}{-}",
        r"  \DeclareUnicodeCharacter{2212}{-}",
        r"  \DeclareUnicodeCharacter{2026}{...}",
        r"\fi",
        r"",
        rf"\title{{{title}}}",
        rf"\subtitle{{{subtitle}}}",
        r"\author{Generated from open-unlearning metrics}",
        r"\date{}",
        r"",
        r"\begin{document}",
        r"",
    ]

    if neutral_theme:
        insert_lines = [
            r"\setbeamercolor{background canvas}{bg=white}",
            r"\setbeamercolor{normal text}{fg=black,bg=white}",
            r"\setbeamercolor{structure}{fg=black}",
            r"\setbeamercolor{title}{fg=black}",
            r"\setbeamercolor{frametitle}{fg=black,bg=white}",
            r"\setbeamercolor{footline}{fg=black,bg=white}",
            r"\hypersetup{colorlinks=false,hidelinks}",
            r"",
        ]
    else:
        insert_lines = [
            r"\definecolor{Primary}{HTML}{143B63}",
            r"\definecolor{Accent}{HTML}{0F766E}",
            r"\definecolor{Warm}{HTML}{B45309}",
            r"\definecolor{SoftBlue}{HTML}{EFF6FF}",
            r"\definecolor{SoftTeal}{HTML}{ECFDF5}",
            r"\definecolor{SoftGray}{HTML}{F3F4F6}",
            r"\definecolor{SoftRed}{HTML}{FEE2E2}",
            r"",
            r"\setbeamercolor{structure}{fg=Primary}",
            r"\setbeamercolor{title}{fg=Primary}",
            r"\setbeamercolor{frametitle}{fg=Primary,bg=white}",
            r"\setbeamercolor{block title}{fg=white,bg=Primary}",
            r"\setbeamercolor{block body}{fg=black,bg=SoftGray}",
            r"\setbeamercolor{alertblock title}{fg=white,bg=Accent}",
            r"\setbeamercolor{alertblock body}{fg=black,bg=SoftTeal}",
            r"",
        ]
    lines[lines.index(rf"\title{{{title}}}") : lines.index(rf"\title{{{title}}}")] = insert_lines

    for frame in frames:
        lines.append(frame)
        lines.append("")

    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def normalize_numeric_token(token: str | None) -> tuple[float, str]:
    if token is None:
        return (0.0, "")
    return (float(token.replace("p", ".")), token)


def simple_ce_sort_key(method_name: str) -> tuple[float, float, float, str]:
    match = SIMPLE_CE_RE.fullmatch(method_name)
    if match is None:
        return (float("inf"), float("inf"), float("inf"), method_name)
    cf_value, _ = normalize_numeric_token(match.group("cf"))
    ret_value, _ = normalize_numeric_token(match.group("ret"))
    gamma_value, _ = normalize_numeric_token(match.group("gamma"))
    return (cf_value, ret_value, gamma_value, method_name)


def simple_ce_display_name(method_name: str, variant_tag: str | None = None) -> str:
    match = SIMPLE_CE_RE.fullmatch(method_name)
    if match is None:
        return method_name
    prefix = "SimpleCE" if variant_tag is None else f"SimpleCE_{variant_tag}"
    cf_value = (match.group("cf") or "--").replace("p", ".")
    ret_value = (match.group("ret") or "--").replace("p", ".")
    gamma_value = (match.group("gamma") or "--").replace("p", ".")
    return f"{prefix} cf={cf_value} ret={ret_value} gamma={gamma_value}"


def build_bundles(
    root_map: dict[str, Path],
    split: str,
    lr: str,
    metrics: list[tuple[str, str]],
    *,
    wrong_generation_index: dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, str]]]]]]
    | None = None,
    wrong_generation_labels_by_source: dict[str, str | None] | None = None,
) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    return {
        source_name: load_table_bundle(
            root,
            split,
            lr,
            metrics,
            wrong_generation_index=wrong_generation_index,
            wrong_generation_label=(
                None
                if wrong_generation_labels_by_source is None
                else wrong_generation_labels_by_source.get(source_name)
            ),
        )
        for source_name, root in root_map.items()
    }


def row_spec_has_source_rows(
    bundles: dict[str, dict[str, dict[str, dict[str, str]]]],
    row_spec: tuple[str, str, str, str],
    metrics: list[tuple[str, str]],
) -> bool:
    source_name, source_method, _display_name, _color = row_spec
    source_bundle = bundles[source_name]
    return any(
        source_bundle[metric_name].get(source_method) is not None
        for metric_name, _ in metrics
        if metric_name not in WRONG_GENERATION_METRIC_NAMES
    )


def filter_row_specs(
    row_specs: list[tuple[str, str, str, str]],
    bundles: dict[str, dict[str, dict[str, dict[str, str]]]],
    metrics: list[tuple[str, str]],
) -> list[tuple[str, str, str, str]]:
    return [row_spec for row_spec in row_specs if row_spec_has_source_rows(bundles, row_spec, metrics)]


def build_combined_row_specs(
    simnpo_root: Path | None,
    simplece_new_root: Path | None,
    simplece_old_root: Path | None,
) -> list[tuple[str, str, str, str]]:
    row_specs = list(COMBINED_ROW_SPECS)
    if simnpo_root is not None:
        row_specs.append(SIMNPO_ROW_SPEC)
    if simplece_new_root is not None:
        variant_tag = "new" if simplece_old_root is not None else None
        for method_name in COMBINED_SIMPLECE_METHODS:
            row_specs.append(
                ("simplece_new", method_name, simple_ce_display_name(method_name, variant_tag), "")
            )
            if simplece_old_root is not None:
                row_specs.append(
                    ("simplece_old", method_name, simple_ce_display_name(method_name, "old"), "")
                )
    return row_specs


def build_combined_slide_row_specs(
    simnpo_root: Path | None,
    simplece_new_root: Path | None,
    simplece_old_root: Path | None,
) -> list[tuple[str, str, str, str]]:
    return build_combined_row_specs(simnpo_root, simplece_new_root, simplece_old_root)


def load_simplece_row_specs(
    source_specs: list[tuple[str, Path, str | None, str]],
    split: str,
    lr: str,
    metrics: list[tuple[str, str]],
) -> list[tuple[str, str, str, str]]:
    first_metric = FIXED_METRICS[0][0]
    methods_by_source: dict[str, set[str]] = {}
    all_methods: set[str] = set()
    for source_name, root, _variant_tag, _color in source_specs:
        rows = load_metric_rows(root / split / lr / f"{first_metric}.tsv")
        methods = {method_name for method_name in rows if method_name.startswith("simple_ce")}
        methods_by_source[source_name] = methods
        all_methods.update(methods)

    row_specs: list[tuple[str, str, str, str]] = []
    for source_name, _root, variant_tag, color in source_specs:
        for method_name in sorted(methods_by_source[source_name], key=simple_ce_sort_key):
            row_specs.append(
                (source_name, method_name, simple_ce_display_name(method_name, variant_tag), color)
            )
    return row_specs


def build_output_text(
    *,
    header_comment: str,
    row_specs_by_split_lr: dict[tuple[str, str], list[tuple[str, str, str, str]]],
    bundles_by_split_lr: dict[tuple[str, str], dict[str, dict[str, dict[str, dict[str, str]]]]],
    metrics: list[tuple[str, str]],
    caption_prefix: str,
    label_prefix: str,
    split_lrs: list[tuple[str, str]] | None = None,
) -> str:
    abbreviations = [
        f"{metric_abbrev}={metric_name}"
        for metric_name, metric_abbrev in metrics
    ]
    sections = [
        header_comment,
        "% Abbreviations: " + ", ".join(abbreviations) + ".",
        "% Each table is epoch-specific and uses either epoch 2 or epoch 5.",
        "",
    ]

    first_table = True
    split_lrs = split_lrs or [(split, lr) for split in SPLITS for lr in LRS]
    for split, lr in split_lrs:
        row_specs = row_specs_by_split_lr[(split, lr)]
        bundles = bundles_by_split_lr[(split, lr)]
        for epoch_column, epoch_label in EPOCH_SPECS:
            if not first_table:
                sections.append("")
                sections.append("")
            sections.append(f"% Split: {split} | LR: {lr} | Epoch: {epoch_label}")
            sections.append(
                build_table(
                    caption=(
                        f"{caption_prefix} for {SPLIT_LABELS[split]} at LR={lr}, epoch {epoch_label}. "
                        "Values are percentages."
                    ),
                    label=f"tab:{label_prefix}-{split.replace('_', '-')}-{lr.replace('-', '')}-ep{epoch_label}",
                    epoch_column=epoch_column,
                    bundles=bundles,
                    row_specs=row_specs,
                    metrics=metrics,
                )
            )
            first_table = False
    sections.append("")
    return "\n".join(sections)


def build_frames(
    *,
    row_specs_by_split_lr: dict[tuple[str, str], list[tuple[str, str, str, str]]],
    bundles_by_split_lr: dict[tuple[str, str], dict[str, dict[str, dict[str, dict[str, str]]]]],
    metrics: list[tuple[str, str]],
    title_prefix: str,
    compact: bool = False,
    split_lrs: list[tuple[str, str]] | None = None,
) -> list[str]:
    frames: list[str] = []
    split_lrs = split_lrs or [(split, lr) for split in SPLITS for lr in LRS]
    for split, lr in split_lrs:
        row_specs = row_specs_by_split_lr[(split, lr)]
        bundles = bundles_by_split_lr[(split, lr)]
        for epoch_column, epoch_label in EPOCH_SPECS:
            frames.append(
                build_slide_frame(
                    frame_title=f"{title_prefix} | {SPLIT_LABELS[split]} | LR = {lr} | Epoch = {epoch_label}",
                    epoch_column=epoch_column,
                    bundles=bundles,
                    row_specs=row_specs,
                    metrics=metrics,
                    compact=compact,
                )
            )
    return frames


def lr_sort_key(lr: str) -> tuple[int, float | str]:
    if lr in LRS:
        return (LRS.index(lr), 0.0)
    try:
        return (len(LRS), float(lr))
    except ValueError:
        return (len(LRS), lr)


def discover_available_split_lrs(roots: list[Path]) -> list[tuple[str, str]]:
    discovered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for split in SPLITS:
        candidate_lrs: set[str] = set()
        for root in roots:
            split_root = root / split
            if not split_root.exists():
                continue
            for child in split_root.iterdir():
                if child.is_dir():
                    candidate_lrs.add(child.name)
        for lr in sorted(candidate_lrs, key=lr_sort_key):
            item = (split, lr)
            if item not in seen:
                seen.add(item)
                discovered.append(item)
    return discovered


def load_variant_row_specs(
    variant_sources: list[tuple[str, Path]],
    split: str,
    lr: str,
    *,
    selected_algorithms: set[str] | None = None,
    selected_method_keys: set[str] | None = None,
    display_mode: str = "full",
) -> list[tuple[str, str, str, str]]:
    first_metric = FIXED_METRICS[0][0]
    color_by_algorithm = {
        "multicf": "teal!12",
        "boundary_cf": "cyan!12",
        "span_cf": "yellow!12",
        "span_cf_samnpo": "orange!10",
        "span_cf_simnpo": "yellow!18",
        "span_cf_local_retain": "lime!12",
        "span_cf_simnpo_local_retain": "lime!18",
        "span_cf_simnpo_sam": "orange!12",
        "span_cf_simnpo_projected": "red!12",
    }

    def include_method(method_name: str) -> bool:
        info = variant_info_from_method_key(method_name)
        if info is None:
            return False
        if selected_algorithms or selected_method_keys:
            if selected_method_keys and method_name in selected_method_keys:
                return True
            algorithm = base_variant_algorithm(method_name) or info.algorithm
            return bool(selected_algorithms and algorithm in selected_algorithms)
        return True

    source_by_method: dict[str, str] = {}
    for source_name, variant_root in variant_sources:
        table_path = variant_root / split / lr / f"{first_metric}.tsv"
        if not table_path.exists():
            continue
        rows = load_metric_rows(table_path)
        for method_name in rows:
            if method_name in source_by_method:
                continue
            if include_method(method_name):
                source_by_method[method_name] = source_name

    method_names = sorted(
        source_by_method,
        key=lambda method_name: variant_sort_key(method_name) or (999, 999, method_name),
    )

    def variant_display_name(method_name: str) -> str:
        info = variant_info_from_method_key(method_name)
        if info is None:
            return method_name
        if display_mode != "compact":
            return info.display_name

        algorithm = base_variant_algorithm(method_name) or info.algorithm
        if info.order_index != 999:
            return info.display_name
        return {
            "multicf": "MultiCF",
            "boundary_cf": "BoundaryCF",
            "span_cf": "SpanCF",
            "span_cf_samnpo": "SpanCF-SAMNPO",
            "span_cf_simnpo": "SpanCF-SimNPO",
            "span_cf_local_retain": "SpanCF-LocalRetain",
            "span_cf_simnpo_local_retain": "SpanCF-SimNPO-LocalRetain",
            "span_cf_simnpo_sam": "SpanCF-SimNPO-SAM",
            "span_cf_simnpo_projected": "SpanCF-SimNPO-Projected",
        }.get(algorithm, info.display_name)

    row_specs: list[tuple[str, str, str, str]] = []
    for method_name in method_names:
        info = variant_info_from_method_key(method_name)
        if info is None:
            continue
        row_specs.append(
            (
                source_by_method[method_name],
                method_name,
                variant_display_name(method_name),
                color_by_algorithm.get(base_variant_algorithm(method_name) or "", ""),
            )
        )
    return row_specs


def main() -> None:
    args = parse_args()
    output_file = args.output_file.expanduser().resolve()
    output_slides_tex = (
        None if args.output_slides_tex is None else args.output_slides_tex.expanduser().resolve()
    )
    variant_roots = (
        []
        if args.variant_root is None
        else [root.expanduser().resolve() for root in args.variant_root]
    )
    old_root = None if args.old_root is None else args.old_root.expanduser().resolve()
    new_root = None if args.new_root is None else args.new_root.expanduser().resolve()
    simnpo_root = None if args.simnpo_root is None else args.simnpo_root.expanduser().resolve()
    simplece_new_root = (
        simnpo_root
        if args.simplece_new_root is None
        else args.simplece_new_root.expanduser().resolve()
    )
    simplece_old_root = (
        None if args.simplece_old_root is None else args.simplece_old_root.expanduser().resolve()
    )
    output_simplece_file = (
        None if args.output_simplece_file is None else args.output_simplece_file.expanduser().resolve()
    )
    output_simplece_slides_tex = (
        None
        if args.output_simplece_slides_tex is None
        else args.output_simplece_slides_tex.expanduser().resolve()
    )
    wrong_generations_root = (
        None
        if args.wrong_generations_root is None
        else args.wrong_generations_root.expanduser().resolve()
    )

    if variant_roots:
        if old_root is not None or new_root is not None:
            raise ValueError("--variant-root cannot be combined with --old-root/--new-root")
        if output_simplece_file is not None or output_simplece_slides_tex is not None:
            raise ValueError("SimpleCE-only outputs are not supported with --variant-root")

        variant_sources = [
            ("variant", variant_roots[0])
        ] if len(variant_roots) == 1 else [
            (f"variant{index + 1}", root) for index, root in enumerate(variant_roots)
        ]
        selected_algorithms = (
            None
            if not args.variant_algorithm
            else {value.strip() for value in args.variant_algorithm if value and value.strip()}
        )
        selected_method_keys = (
            None
            if not args.variant_method_key
            else {value.strip() for value in args.variant_method_key if value and value.strip()}
        )

        wrong_generation_index: dict[
            str,
            dict[str, dict[str, dict[str, dict[str, dict[str, str]]]]],
        ] | None = None
        wrong_generation_labels_by_source: dict[str, str | None] | None = None
        available_wrong_generation_labels: set[str] = set()
        if wrong_generations_root is not None:
            wrong_generation_index, available_wrong_generation_labels = load_wrong_generation_index(
                wrong_generations_root
            )
            wrong_generation_labels_by_source = {
                source_name: resolve_wrong_generation_label(root, available_wrong_generation_labels)
                for source_name, root in variant_sources
            }

        metrics = discover_metrics(
            [root for _source_name, root in variant_sources],
            include_wrong_generation_metrics=wrong_generations_root is not None,
        )
        split_lrs = discover_available_split_lrs([root for _source_name, root in variant_sources])
        if not split_lrs:
            roots_text = ", ".join(str(root) for _source_name, root in variant_sources)
            raise FileNotFoundError(f"No split/LR tables found under {roots_text}")

        bundles_by_split_lr = {
            (split, lr): build_bundles(
                {source_name: root for source_name, root in variant_sources},
                split,
                lr,
                metrics,
                wrong_generation_index=wrong_generation_index,
                wrong_generation_labels_by_source=wrong_generation_labels_by_source,
            )
            for split, lr in split_lrs
        }
        row_specs_by_split_lr = {
            (split, lr): load_variant_row_specs(
                variant_sources,
                split,
                lr,
                selected_algorithms=selected_algorithms,
                selected_method_keys=selected_method_keys,
                display_mode=args.variant_display,
            )
            for split, lr in split_lrs
        }
        split_lrs = [
            (split, lr)
            for split, lr in split_lrs
            if row_specs_by_split_lr[(split, lr)]
        ]
        if not split_lrs:
            raise FileNotFoundError("No matching variant rows found for the requested selection.")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_text = build_output_text(
            header_comment="% New-method tables generated by src/tools/build_results_combine_tables.py",
            row_specs_by_split_lr=row_specs_by_split_lr,
            bundles_by_split_lr=bundles_by_split_lr,
            metrics=metrics,
            caption_prefix="New-method results",
            label_prefix="new-methods",
            split_lrs=split_lrs,
        )
        output_file.write_text(output_text, encoding="utf-8")

        if output_slides_tex is not None:
            output_slides_tex.parent.mkdir(parents=True, exist_ok=True)
            frames = build_frames(
                row_specs_by_split_lr=row_specs_by_split_lr,
                bundles_by_split_lr=bundles_by_split_lr,
                metrics=metrics,
                title_prefix="New Method Tables",
                compact=True,
                split_lrs=split_lrs,
            )
            slides_tex = build_slides_tex(
                title="New Method Tables",
                subtitle=f"{len(frames)} split/LR/epoch slides",
                frames=frames,
            )
            output_slides_tex.write_text(slides_tex, encoding="utf-8")
        return

    if old_root is None or new_root is None:
        raise ValueError("--old-root and --new-root are required unless --variant-root is used")

    combined_roots = {"old": old_root, "new": new_root}
    if simnpo_root is not None:
        combined_roots["simnpo"] = simnpo_root
    if simplece_new_root is not None:
        combined_roots["simplece_new"] = simplece_new_root
    if simplece_old_root is not None:
        combined_roots["simplece_old"] = simplece_old_root
    available_wrong_generation_labels: set[str] = set()
    wrong_generation_index: dict[
        str,
        dict[str, dict[str, dict[str, dict[str, dict[str, str]]]]],
    ] | None = None
    wrong_generation_labels_by_source: dict[str, str | None] | None = None
    if wrong_generations_root is not None:
        wrong_generation_index, available_wrong_generation_labels = load_wrong_generation_index(
            wrong_generations_root
        )
        wrong_generation_labels_by_source = {
            source_name: resolve_wrong_generation_label(root, available_wrong_generation_labels)
            for source_name, root in combined_roots.items()
        }

    metrics = discover_metrics(
        list(combined_roots.values()),
        include_wrong_generation_metrics=wrong_generations_root is not None,
    )
    split_lrs = discover_available_split_lrs(list(combined_roots.values()))

    combined_bundles_by_split_lr = {
        (
            split,
            lr,
        ): build_bundles(
            combined_roots,
            split,
            lr,
            metrics,
            wrong_generation_index=wrong_generation_index,
            wrong_generation_labels_by_source=wrong_generation_labels_by_source,
        )
        for split, lr in split_lrs
    }
    combined_row_specs_by_split_lr = {}
    combined_slide_row_specs_by_split_lr = {}
    for split, lr in split_lrs:
        bundles = combined_bundles_by_split_lr[(split, lr)]
        combined_row_specs_by_split_lr[(split, lr)] = filter_row_specs(
            build_combined_row_specs(simnpo_root, simplece_new_root, simplece_old_root),
            bundles,
            metrics,
        )
        combined_slide_row_specs_by_split_lr[(split, lr)] = filter_row_specs(
            build_combined_slide_row_specs(simnpo_root, simplece_new_root, simplece_old_root),
            bundles,
            metrics,
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_text = build_output_text(
        header_comment="% Combined tables generated by src/tools/build_results_combine_tables.py",
        row_specs_by_split_lr=combined_row_specs_by_split_lr,
        bundles_by_split_lr=combined_bundles_by_split_lr,
        metrics=metrics,
        caption_prefix="Combined old/new results",
        label_prefix="combined",
        split_lrs=split_lrs,
    )
    output_file.write_text(output_text, encoding="utf-8")

    if output_slides_tex is not None:
        output_slides_tex.parent.mkdir(parents=True, exist_ok=True)
        combined_frames = build_frames(
            row_specs_by_split_lr=combined_slide_row_specs_by_split_lr,
            bundles_by_split_lr=combined_bundles_by_split_lr,
            metrics=metrics,
            title_prefix="Combined Tables",
            compact=True,
            split_lrs=split_lrs,
        )
        slides_tex = build_slides_tex(
            title="Combined DualCF Tables",
            subtitle=f"{len(combined_frames)} split/LR/epoch slides",
            frames=combined_frames,
        )
        output_slides_tex.write_text(slides_tex, encoding="utf-8")

    if output_simplece_file is not None:
        if simplece_new_root is None:
            raise ValueError(
                "--simplece-new-root or --simnpo-root is required when writing SimpleCE-only outputs"
            )
        output_simplece_file.parent.mkdir(parents=True, exist_ok=True)
        if simplece_old_root is None:
            simplece_source_specs = [("simplece", simplece_new_root, None, "blue!12")]
            simplece_slide_source_specs = [("simplece", simplece_new_root, None, "")]
            simplece_root_map = {"simplece": simplece_new_root}
        else:
            simplece_source_specs = [
                ("simplece_old", simplece_old_root, "old", "orange!12"),
                ("simplece_new", simplece_new_root, "new", "blue!12"),
            ]
            simplece_slide_source_specs = [
                ("simplece_old", simplece_old_root, "old", ""),
                ("simplece_new", simplece_new_root, "new", ""),
            ]
            simplece_root_map = {
                "simplece_old": simplece_old_root,
                "simplece_new": simplece_new_root,
            }
        simplece_row_specs_by_split_lr = {
            (split, lr): load_simplece_row_specs(simplece_source_specs, split, lr, metrics)
            for split, lr in split_lrs
        }
        simplece_slide_row_specs_by_split_lr = {
            (split, lr): load_simplece_row_specs(simplece_slide_source_specs, split, lr, metrics)
            for split, lr in split_lrs
        }
        simplece_bundles_by_split_lr = {
            (
                split,
                lr,
            ): build_bundles(
                simplece_root_map,
                split,
                lr,
                metrics,
                wrong_generation_index=wrong_generation_index,
                wrong_generation_labels_by_source=(
                    None
                    if wrong_generation_labels_by_source is None
                    else {
                        source_name: resolve_wrong_generation_label(
                            root,
                            available_wrong_generation_labels,
                        )
                        for source_name, root in simplece_root_map.items()
                    }
                ),
            )
            for split, lr in split_lrs
        }
        simplece_output_text = build_output_text(
            header_comment="% SimpleCE-only tables generated by src/tools/build_results_combine_tables.py",
            row_specs_by_split_lr=simplece_row_specs_by_split_lr,
            bundles_by_split_lr=simplece_bundles_by_split_lr,
            metrics=metrics,
            caption_prefix="SimpleCE ablations",
            label_prefix="simplece",
            split_lrs=split_lrs,
        )
        output_simplece_file.write_text(simplece_output_text, encoding="utf-8")

        if output_simplece_slides_tex is not None:
            output_simplece_slides_tex.parent.mkdir(parents=True, exist_ok=True)
            simplece_frames = build_frames(
                row_specs_by_split_lr=simplece_slide_row_specs_by_split_lr,
                bundles_by_split_lr=simplece_bundles_by_split_lr,
                metrics=metrics,
                title_prefix="SimpleCE Ablations",
                split_lrs=split_lrs,
            )
            simplece_slides_tex = build_slides_tex(
                title="SimpleCE Ablation Tables",
                subtitle=f"{len(simplece_frames)} split/LR/epoch slides",
                frames=simplece_frames,
                neutral_theme=True,
            )
            output_simplece_slides_tex.write_text(simplece_slides_tex, encoding="utf-8")


if __name__ == "__main__":
    main()
