#!/usr/bin/env python3
"""Build combined LaTeX tables from old and new structured-saves trees."""

from __future__ import annotations

import argparse
import csv
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
ROW_SPECS = [
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
SPLIT_LABELS = {
    "duet_rare": "DUET Rare",
    "duet_popular": "DUET Popular",
    "duet_merged": "DUET Merged",
    "rwku": "RWKU",
}


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


def build_header_cells() -> list[str]:
    cells = ["Method"]
    for metric_name, metric_abbrev in METRICS:
        direction = METRIC_DIRECTION[metric_name]
        cells.append(f"{metric_abbrev}{direction}")
    return cells


def build_row_cells(
    epoch_column: str,
    old_bundle: dict[str, dict[str, dict[str, str]]],
    new_bundle: dict[str, dict[str, dict[str, str]]],
) -> list[str]:
    lines: list[str] = []
    for source_name, source_method, display_name, color in ROW_SPECS:
        source_bundle = old_bundle if source_name == "old" else new_bundle
        value_cells = [display_name]
        for metric_name, _metric_abbrev in METRICS:
            method_row = source_bundle[metric_name].get(source_method)
            raw_value = None if method_row is None else method_row.get(epoch_column)
            value_cells.append(format_percent(raw_value))

        lines.append(rf"\rowcolor{{{color}}}")
        lines.append(" & ".join(value_cells) + r" \\")
    return lines


def build_table(
    split: str,
    lr: str,
    epoch_column: str,
    epoch_label: str,
    old_root: Path,
    new_root: Path,
) -> str:
    old_bundle = load_table_bundle(old_root, split, lr)
    new_bundle = load_table_bundle(new_root, split, lr)

    caption = (
        f"Combined old/new results for {SPLIT_LABELS[split]} at LR={lr}, epoch {epoch_label}. "
        "Values are percentages."
    )
    label = f"tab:combined-{split.replace('_', '-')}-{lr.replace('-', '')}-ep{epoch_label}"
    header_cells = build_header_cells()
    row_lines = build_row_cells(epoch_column, old_bundle, new_bundle)

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
    split: str,
    lr: str,
    epoch_column: str,
    epoch_label: str,
    old_root: Path,
    new_root: Path,
) -> str:
    old_bundle = load_table_bundle(old_root, split, lr)
    new_bundle = load_table_bundle(new_root, split, lr)
    header_cells = build_header_cells()
    row_lines = build_row_cells(epoch_column, old_bundle, new_bundle)

    lines = [
        r"\begin{frame}[t]",
        rf"\frametitle{{{SPLIT_LABELS[split]} | LR = {lr} | Epoch = {epoch_label}}}",
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


def build_slides_tex(old_root: Path, new_root: Path) -> str:
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
        r"\title{Combined Old/New DualCF Tables}",
        r"\subtitle{16 split/LR/epoch slides}",
        r"\author{Generated from open-unlearning metrics}",
        r"\date{}",
        r"",
        r"\begin{document}",
        r"",
    ]

    for split in SPLITS:
        for lr in LRS:
            for epoch_column, epoch_label in EPOCH_SPECS:
                lines.append(build_slide_frame(split, lr, epoch_column, epoch_label, old_root, new_root))
                lines.append("")

    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def build_output_text(old_root: Path, new_root: Path) -> str:
    sections = [
        "% Combined tables generated by src/tools/build_results_combine_tables.py",
        "% Abbreviations: F=forget_qa_rouge, H=holdout_qa_rouge, FC=forget_qa_cos_sim, HC=holdout_qa_cos_sim, U=utility_avg, M=MMLU-Pro, T=TruthfulQA, W=Winogrande, A=ARC.",
        "% Each table is epoch-specific and uses either epoch 2 or epoch 5.",
        "",
    ]

    first_table = True
    for split in SPLITS:
        for lr in LRS:
            for epoch_column, epoch_label in EPOCH_SPECS:
                if not first_table:
                    sections.append("")
                    sections.append("")
                sections.append(f"% Split: {split} | LR: {lr} | Epoch: {epoch_label}")
                sections.append(build_table(split, lr, epoch_column, epoch_label, old_root, new_root))
                first_table = False
    sections.append("")
    return "\n".join(sections)


def main() -> None:
    args = parse_args()
    old_root = args.old_root.expanduser().resolve()
    new_root = args.new_root.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    output_slides_tex = (
        None if args.output_slides_tex is None else args.output_slides_tex.expanduser().resolve()
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_text = build_output_text(old_root, new_root)
    output_file.write_text(output_text, encoding="utf-8")

    if output_slides_tex is not None:
        output_slides_tex.parent.mkdir(parents=True, exist_ok=True)
        slides_tex = build_slides_tex(old_root, new_root)
        output_slides_tex.write_text(slides_tex, encoding="utf-8")


if __name__ == "__main__":
    main()
