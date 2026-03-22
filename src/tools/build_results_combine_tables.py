#!/usr/bin/env python3
"""Build combined LaTeX tables from structured-saves trees."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SPLITS = ["duet_rare", "duet_popular", "duet_merged", "rwku"]
LRS = ["1e-4", "5e-5"]
EPOCH_SPECS = [("2.0", "2"), ("5.0", "5")]
METRICS = [
    ("forget_qa_rouge", "F"),
    ("holdout_qa_rouge", "H"),
    ("forget_qa_cos_sim", "FC"),
    ("holdout_qa_cos_sim", "HC"),
    ("utility_avg", "U"),
    ("mmlu_pro_400_acc", "M"),
    ("truthfulqa_bin_200_acc", "T"),
    ("winogrande_200_acc", "W"),
    ("arc_200_acc", "A"),
]
METRIC_DIRECTION = {
    "forget_qa_rouge": r"$\downarrow$",
    "holdout_qa_rouge": r"$\uparrow$",
    "forget_qa_cos_sim": r"$\downarrow$",
    "holdout_qa_cos_sim": r"$\uparrow$",
    "utility_avg": r"$\uparrow$",
    "mmlu_pro_400_acc": r"$\uparrow$",
    "truthfulqa_bin_200_acc": r"$\uparrow$",
    "winogrande_200_acc": r"$\uparrow$",
    "arc_200_acc": r"$\uparrow$",
}
COMBINED_ROW_SPECS = [
    ("old", "full", "Full-old", "blue!10"),
    ("old", "d_only", "d-only-old", "orange!10"),
    ("old", "a_only", "a-only-old", "green!10"),
    ("old", "dpo", "DPO-old", "gray!8"),
    ("old", "simple_ce", "Simple-CE", "gray!8"),
    ("old", "ga", "GA", "gray!8"),
    ("old", "npo", "NPO", "gray!8"),
    ("old", "simnpo", "SimNPO", "gray!8"),
    ("old", "npo_sam", "NPO-SAM", "gray!8"),
    ("old", "loku", "LoKU", "gray!8"),
    ("new", "full", "Full-new", "blue!20"),
    ("new", "d_only", "d-only-new", "orange!20"),
    ("new", "a_only", "a-only-new", "green!20"),
    ("new", "dpo", "DPO-new", "gray!15"),
]
SIMNPO_ROW_SPEC = ("simnpo", "simnpo", "SimNPO", "red!12")
COMBINED_SIMPLECE_SLIDE_METHODS = [
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
        required=True,
        help="Path to the old structured-saves directory (for example metrics-ep5-all-v2/structured-saves).",
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        required=True,
        help="Path to the new structured-saves directory (for example metrics-ep5-dualfc-new_cf/structured-saves).",
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
        "--simnpo-root",
        type=Path,
        help="Optional path to a structured-saves directory that contains simnpo/simple_ce rows.",
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


def load_metric_rows(table_path: Path) -> dict[str, dict[str, str]]:
    if not table_path.exists():
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


def load_table_bundle(root: Path, split: str, lr: str) -> dict[str, dict[str, dict[str, str]]]:
    bundle: dict[str, dict[str, dict[str, str]]] = {}
    for metric_name, _metric_abbrev in METRICS:
        bundle[metric_name] = load_metric_rows(root / split / lr / f"{metric_name}.tsv")
    return bundle


def format_percent(raw_value: str | None) -> str:
    if raw_value in {None, ""}:
        return "--"
    return f"{float(raw_value) * 100.0:.1f}"


def escape_latex(text: str) -> str:
    return text.replace("_", r"\_")


def build_header_cells() -> list[str]:
    cells = ["Method"]
    for metric_name, metric_abbrev in METRICS:
        direction = METRIC_DIRECTION[metric_name]
        cells.append(f"{metric_abbrev}{direction}")
    return cells


def build_row_cells(
    epoch_column: str,
    bundles: dict[str, dict[str, dict[str, dict[str, str]]]],
    row_specs: list[tuple[str, str, str, str]],
) -> list[str]:
    lines: list[str] = []
    for source_name, source_method, display_name, color in row_specs:
        source_bundle = bundles[source_name]
        value_cells = [display_name]
        for metric_name, _metric_abbrev in METRICS:
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
) -> str:
    header_cells = build_header_cells()
    row_lines = build_row_cells(epoch_column, bundles, row_specs)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l*{9}{r}}",
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
) -> str:
    header_cells = build_header_cells()
    row_lines = build_row_cells(epoch_column, bundles, row_specs)

    lines = [
        r"\begin{frame}[t]",
        rf"\frametitle{{{frame_title}}}",
        r"{\tiny Values are percentages. F / FC are lower-is-better; H / HC / U / M / T / W / A are higher-is-better.}",
        r"",
        r"{\tiny F/H = ROUGE,\ FC/HC = cosine similarity,\ U = utility\_avg,\ M = MMLU-Pro,\ T = TruthfulQA,\ W = Winogrande,\ A = ARC.}",
        r"",
        r"\vspace{0.35em}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{adjustbox}{max width=\textwidth,max totalheight=0.78\textheight,center}",
        r"\begin{tabular}{l*{9}{r}}",
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


def simple_ce_display_name(method_name: str) -> str:
    match = SIMPLE_CE_RE.fullmatch(method_name)
    if match is None:
        return method_name
    cf_value = (match.group("cf") or "--").replace("p", ".")
    ret_value = (match.group("ret") or "--").replace("p", ".")
    gamma_value = (match.group("gamma") or "--").replace("p", ".")
    return f"SimpleCE cf={cf_value} ret={ret_value} gamma={gamma_value}"


def build_bundles(root_map: dict[str, Path], split: str, lr: str) -> dict[str, dict[str, dict[str, dict[str, str]]]]:
    return {source_name: load_table_bundle(root, split, lr) for source_name, root in root_map.items()}


def build_combined_row_specs(simnpo_root: Path | None) -> list[tuple[str, str, str, str]]:
    row_specs = list(COMBINED_ROW_SPECS)
    if simnpo_root is not None:
        row_specs.append(SIMNPO_ROW_SPEC)
    return row_specs


def build_combined_slide_row_specs(simnpo_root: Path | None) -> list[tuple[str, str, str, str]]:
    row_specs = build_combined_row_specs(simnpo_root)
    if simnpo_root is not None:
        for method_name in COMBINED_SIMPLECE_SLIDE_METHODS:
            row_specs.append(("simnpo", method_name, simple_ce_display_name(method_name), ""))
    return row_specs


def load_simplece_row_specs(
    simplece_root: Path,
    split: str,
    lr: str,
    *,
    with_colors: bool = True,
) -> list[tuple[str, str, str, str]]:
    first_metric = METRICS[0][0]
    rows = load_metric_rows(simplece_root / split / lr / f"{first_metric}.tsv")
    methods = sorted(
        (method_name for method_name in rows if method_name.startswith("simple_ce")),
        key=simple_ce_sort_key,
    )
    color = "blue!12" if with_colors else ""
    return [("simplece", method_name, simple_ce_display_name(method_name), color) for method_name in methods]


def build_output_text(
    *,
    header_comment: str,
    row_specs_by_split_lr: dict[tuple[str, str], list[tuple[str, str, str, str]]],
    bundles_by_split_lr: dict[tuple[str, str], dict[str, dict[str, dict[str, dict[str, str]]]]],
    caption_prefix: str,
    label_prefix: str,
) -> str:
    sections = [
        header_comment,
        "% Abbreviations: F=forget_qa_rouge, H=holdout_qa_rouge, FC=forget_qa_cos_sim, HC=holdout_qa_cos_sim, U=utility_avg, M=MMLU-Pro, T=TruthfulQA, W=Winogrande, A=ARC.",
        "% Each table is epoch-specific and uses either epoch 2 or epoch 5.",
        "",
    ]

    first_table = True
    for split in SPLITS:
        for lr in LRS:
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
                    )
                )
                first_table = False
    sections.append("")
    return "\n".join(sections)


def build_frames(
    *,
    row_specs_by_split_lr: dict[tuple[str, str], list[tuple[str, str, str, str]]],
    bundles_by_split_lr: dict[tuple[str, str], dict[str, dict[str, dict[str, dict[str, str]]]]],
    title_prefix: str,
) -> list[str]:
    frames: list[str] = []
    for split in SPLITS:
        for lr in LRS:
            row_specs = row_specs_by_split_lr[(split, lr)]
            bundles = bundles_by_split_lr[(split, lr)]
            for epoch_column, epoch_label in EPOCH_SPECS:
                frames.append(
                    build_slide_frame(
                        frame_title=f"{title_prefix} | {SPLIT_LABELS[split]} | LR = {lr} | Epoch = {epoch_label}",
                        epoch_column=epoch_column,
                        bundles=bundles,
                        row_specs=row_specs,
                    )
                )
    return frames


def main() -> None:
    args = parse_args()
    old_root = args.old_root.expanduser().resolve()
    new_root = args.new_root.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    output_slides_tex = (
        None if args.output_slides_tex is None else args.output_slides_tex.expanduser().resolve()
    )
    simnpo_root = None if args.simnpo_root is None else args.simnpo_root.expanduser().resolve()
    output_simplece_file = (
        None if args.output_simplece_file is None else args.output_simplece_file.expanduser().resolve()
    )
    output_simplece_slides_tex = (
        None
        if args.output_simplece_slides_tex is None
        else args.output_simplece_slides_tex.expanduser().resolve()
    )

    combined_roots = {"old": old_root, "new": new_root}
    if simnpo_root is not None:
        combined_roots["simnpo"] = simnpo_root

    combined_row_specs_by_split_lr = {
        (split, lr): build_combined_row_specs(simnpo_root) for split in SPLITS for lr in LRS
    }
    combined_slide_row_specs_by_split_lr = {
        (split, lr): build_combined_slide_row_specs(simnpo_root) for split in SPLITS for lr in LRS
    }
    combined_bundles_by_split_lr = {
        (split, lr): build_bundles(combined_roots, split, lr) for split in SPLITS for lr in LRS
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_text = build_output_text(
        header_comment="% Combined tables generated by src/tools/build_results_combine_tables.py",
        row_specs_by_split_lr=combined_row_specs_by_split_lr,
        bundles_by_split_lr=combined_bundles_by_split_lr,
        caption_prefix="Combined old/new results",
        label_prefix="combined",
    )
    output_file.write_text(output_text, encoding="utf-8")

    if output_slides_tex is not None:
        output_slides_tex.parent.mkdir(parents=True, exist_ok=True)
        combined_frames = build_frames(
            row_specs_by_split_lr=combined_slide_row_specs_by_split_lr,
            bundles_by_split_lr=combined_bundles_by_split_lr,
            title_prefix="Combined Tables",
        )
        slides_tex = build_slides_tex(
            title="Combined DualCF Tables",
            subtitle=f"{len(combined_frames)} split/LR/epoch slides",
            frames=combined_frames,
        )
        output_slides_tex.write_text(slides_tex, encoding="utf-8")

    if output_simplece_file is not None:
        if simnpo_root is None:
            raise ValueError("--simnpo-root is required when writing SimpleCE-only outputs")
        output_simplece_file.parent.mkdir(parents=True, exist_ok=True)
        simplece_row_specs_by_split_lr = {
            (split, lr): load_simplece_row_specs(simnpo_root, split, lr) for split in SPLITS for lr in LRS
        }
        simplece_slide_row_specs_by_split_lr = {
            (split, lr): load_simplece_row_specs(simnpo_root, split, lr, with_colors=False)
            for split in SPLITS
            for lr in LRS
        }
        simplece_bundles_by_split_lr = {
            (split, lr): build_bundles({"simplece": simnpo_root}, split, lr) for split in SPLITS for lr in LRS
        }
        simplece_output_text = build_output_text(
            header_comment="% SimpleCE-only tables generated by src/tools/build_results_combine_tables.py",
            row_specs_by_split_lr=simplece_row_specs_by_split_lr,
            bundles_by_split_lr=simplece_bundles_by_split_lr,
            caption_prefix="SimpleCE ablations",
            label_prefix="simplece",
        )
        output_simplece_file.write_text(simplece_output_text, encoding="utf-8")

        if output_simplece_slides_tex is not None:
            output_simplece_slides_tex.parent.mkdir(parents=True, exist_ok=True)
            simplece_frames = build_frames(
                row_specs_by_split_lr=simplece_slide_row_specs_by_split_lr,
                bundles_by_split_lr=simplece_bundles_by_split_lr,
                title_prefix="SimpleCE Ablations",
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
