"""
Visualize DUET ROUGE results for pop_static_wga vs wga (+ pop_dynam_b_wga + ada_WGD).

Creates a 2x2 grid:
 - Forget Rare
 - Retain Rare
 - Forget Popular
 - Retain Popular

Changes vs prior version:
  1) X axis uses equal spacing categorical positions for lr (not numeric distance).
  2) More colors via a larger palette keyed by (beta_label, alpha).
  3) Two legends, one for rare (top row), one for popular (bottom row).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_NAME = "DUET_SUMMARY.json"

_RE_LR = re.compile(r"_lr([^_]+)")
_RE_BETA = re.compile(r"_beta([^_]+)")
_RE_ALPHA = re.compile(r"_alpha([^_]+)")

_RE_TASK = re.compile(
    r"^duet_[^_]+_(.+?)_(WGA|pop_static_wga|pop_dynam_b_wga|ada_WGD)_",
    re.IGNORECASE,
)


def _alpha_tag_to_float(alpha_tag: str) -> float:
    s = alpha_tag.strip()
    if "p" in s:
        s = s.replace("p", ".")
    return float(s)


def parse_task_name(task_name: str) -> Dict[str, Optional[object]]:
    forget_split: Optional[str] = None
    method_token: Optional[str] = None

    m_task = _RE_TASK.match(task_name)
    if m_task:
        forget_split = m_task.group(1)
        method_token = m_task.group(2)
    else:
        parts = task_name.split("_")
        try:
            i_method = next(
                i
                for i, p in enumerate(parts)
                if p in ("WGA", "pop_static_wga", "pop_dynam_b_wga", "ada_WGD")
            )
            if i_method > 2:
                forget_split = "_".join(parts[2:i_method])
                method_token = parts[i_method]
        except StopIteration:
            forget_split = None
            method_token = None

    m_lr = _RE_LR.search(task_name)
    m_beta = _RE_BETA.search(task_name)
    m_alpha = _RE_ALPHA.search(task_name)

    lr_val: Optional[float] = None
    if m_lr:
        try:
            lr_val = float(m_lr.group(1))
        except Exception:
            lr_val = None

    beta_val: Optional[float] = None
    if m_beta:
        try:
            beta_val = float(m_beta.group(1))
        except Exception:
            beta_val = None

    alpha_val: Optional[float] = None
    if m_alpha:
        try:
            alpha_val = _alpha_tag_to_float(m_alpha.group(1))
        except Exception:
            alpha_val = None

    popularity: Optional[str] = None
    if forget_split:
        s = forget_split.lower()
        if "popular" in s:
            popularity = "popular"
        elif "rare" in s:
            popularity = "rare"

    return {
        "forget_split": forget_split,
        "popularity": popularity,
        "lr": lr_val,
        "beta": beta_val,
        "alpha": alpha_val,
        "_method_token": method_token,
    }


def find_summaries(root: Path) -> List[Path]:
    return list(root.glob("**/evals/" + SUMMARY_NAME))


def load_runs(
    method_dirs: Dict[str, Path] | None = None, base_unlearn: Optional[Path] = None
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    paths: List[Tuple[str, Path]] = []

    if method_dirs:
        for method, base in method_dirs.items():
            if not base.exists():
                continue
            for sp in find_summaries(base):
                paths.append((method, sp))
    else:
        root = base_unlearn or Path(".") / "saves" / "unlearn"
        if root.exists():
            for sp in find_summaries(root):
                paths.append(("infer", sp))

    for method_hint, summary_path in paths:
        try:
            with summary_path.open("r") as fh:
                summary = json.load(fh)
        except Exception:
            continue

        run_dir = summary_path.parent.parent
        task_name = run_dir.name
        meta = parse_task_name(task_name)

        if method_hint != "infer":
            method = method_hint
        else:
            token = (meta.get("_method_token") or "").lower()
            tname = task_name.lower()
            if token == "pop_static_wga" or "pop_static_wga" in tname:
                method = "pop_static_wga"
            elif token == "pop_dynam_b_wga" or "pop_dynam_b_wga" in tname:
                method = "pop_dynam_b_wga"
            elif token == "wga" or "_wga_" in tname:
                method = "wga"
            elif token == "ada_wgd" or "_ada_wgd_" in tname:
                method = "ada_WGD"
            # ada_WGD
            else:
                continue

        rows.append(
            {
                "method": method,
                "task_name": task_name,
                **{k: v for k, v in meta.items() if k != "_method_token"},
                "forget_qa_rouge": summary.get("forget_qa_rouge"),
                "holdout_qa_rouge": summary.get("holdout_qa_rouge"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "method",
                "task_name",
                "forget_split",
                "popularity",
                "lr",
                "beta",
                "alpha",
                "forget_qa_rouge",
                "holdout_qa_rouge",
            ]
        )

    df = pd.DataFrame(rows)
    for col in ("lr", "beta", "alpha"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["lr"])
    return df


def _make_beta_label(series: pd.Series) -> pd.Series:
    def _fn(v):
        try:
            return f"{float(v):g}" if pd.notna(v) else "dynamic"
        except Exception:
            return "dynamic"

    return series.apply(_fn)


def build_color_map(df: pd.DataFrame) -> Dict[Tuple[str, float], object]:
    """
    Assign a stable color per (beta_label, alpha) across all plots.
    Uses a larger palette than default cycle.
    """
    if df.empty:
        return {}

    tmp = df.copy()
    tmp["beta_label"] = _make_beta_label(tmp["beta"])
    # Treat missing alpha as a single "dynamic" bucket using sentinel -1.0
    tmp["alpha_f"] = pd.to_numeric(tmp["alpha"], errors="coerce").fillna(-1.0)

    combos = (
        tmp[["beta_label", "alpha_f"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["beta_label", "alpha_f"])
    )
    keys = [
        (str(r.beta_label), float(r.alpha_f)) for r in combos.itertuples(index=False)
    ]

    # Build a big palette by concatenating tab20 + tab20b + tab20c if available.
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        try:
            cmap = plt.get_cmap(cmap_name)
            palette.extend(list(cmap.colors))
        except Exception:
            pass

    # Fallback if something weird happens
    if not palette:
        palette = list(
            plt.get_cmap("hsv")(i / max(len(keys), 1)) for i in range(len(keys))
        )

    color_map: Dict[Tuple[str, float], object] = {}
    for i, k in enumerate(keys):
        color_map[k] = palette[i % len(palette)]
    return color_map


def _format_lr_label(x: float) -> str:
    if x == 0 or pd.isna(x):
        return str(x)
    s = f"{x:.0e}"
    s = s.replace("e+0", "e+").replace("e-0", "e-")
    return s


def make_subplot(
    ax,
    data: pd.DataFrame,
    metric_key: str,
    title: str,
    color_map: Dict[Tuple[str, float], object],
) -> None:
    ax.set_title(title)
    ax.set_xlabel("lr")
    ax.set_ylabel("ROUGE score")

    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    data = data.copy()
    # Do not drop rows missing alpha; bucket them under a dynamic label
    data = data.dropna(subset=["lr", metric_key]).copy()
    data["beta_label"] = _make_beta_label(data["beta"])
    data["alpha_f"] = pd.to_numeric(data["alpha"], errors="coerce").fillna(-1.0)

    # Categorical lr positions for equal spacing
    unique_lrs = sorted(pd.unique(data["lr"]))
    lr_to_pos = {lr: i for i, lr in enumerate(unique_lrs)}
    data["lr_pos"] = data["lr"].map(lr_to_pos)

    style_map = {
        "wga": {
            "linestyle": ":",
            "marker": "o",
            "linewidth": 1.0,
            "alpha": 1.0,
            "markersize": 3,
        },
        "pop_static_wga": {
            "linestyle": "-",
            "marker": "o",
            "linewidth": 1.2,
            "alpha": 1.0,
            "markersize": 3,
        },
        "pop_dynam_b_wga": {
            "linestyle": "-",
            "marker": "*",
            "linewidth": 2.0,
            "alpha": 0.6,
            "markersize": 6,
        },
        # ada_WGD: star marker, thicker line, half-transparent
        "ada_wgd": {
            "linestyle": "-",
            "marker": "*",
            "linewidth": 2.8,
            "alpha": 0.6,
            "markersize": 7,
        },
    }

    labels_used = set()

    for (method, beta_label, alpha_f), g in data.groupby(
        ["method", "beta_label", "alpha_f"]
    ):
        g = g.sort_values("lr_pos")

        beta_text = str(beta_label)
        alpha_text = "dynamic" if float(alpha_f) == -1.0 else f"{float(alpha_f):g}"
        label = f"{method} β={beta_text}, α={alpha_text}"
        show_label = label not in labels_used

        m = str(method).lower()
        st = style_map.get(
            m,
            {
                "linestyle": "-",
                "marker": "o",
                "linewidth": 1.0,
                "alpha": 1.0,
                "markersize": 3,
            },
        )

        ckey = (str(beta_label), float(alpha_f))
        color = color_map.get(ckey, None)

        ax.plot(
            g["lr_pos"],
            g[metric_key],
            marker=st["marker"],
            markersize=st["markersize"],
            linewidth=st["linewidth"],
            linestyle=st["linestyle"],
            alpha=st["alpha"],
            color=color,
            label=(label if show_label else None),
        )
        labels_used.add(label)

    # Set equal-spaced categorical ticks with lr labels
    ax.set_xticks(list(range(len(unique_lrs))))
    ax.set_xticklabels([_format_lr_label(v) for v in unique_lrs])

    import matplotlib.ticker as mticker

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    ax.grid(True, linestyle="--", alpha=0.2)


def _collect_legend_lines(ax_list) -> Dict[str, object]:
    handles_map: Dict[str, object] = {}
    for ax in ax_list:
        for ln in ax.get_lines():
            lbl = ln.get_label()
            if not lbl or lbl.startswith("_"):
                continue
            if lbl not in handles_map:
                handles_map[lbl] = ln
    return handles_map


def plot_grid(df: pd.DataFrame, save_path: Optional[Path] = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.2, right=0.76)

    popular_df = df[df["popularity"] == "popular"]
    rare_df = df[df["popularity"] == "rare"]

    color_map = build_color_map(df)

    make_subplot(axes[0][0], rare_df, "forget_qa_rouge", "Forget Rare ROUGE", color_map)
    make_subplot(
        axes[0][1], rare_df, "holdout_qa_rouge", "Retain Rare Holdout ROUGE", color_map
    )
    make_subplot(
        axes[1][0], popular_df, "forget_qa_rouge", "Forget Popular ROUGE", color_map
    )
    make_subplot(
        axes[1][1],
        popular_df,
        "holdout_qa_rouge",
        "Retain Popular Holdout ROUGE",
        color_map,
    )

    # Remove y-label text on right column
    axes[0][1].set_ylabel("")
    axes[1][1].set_ylabel("")

    # Two legends, one per popularity row
    rare_handles_map = _collect_legend_lines([axes[0][0], axes[0][1]])
    pop_handles_map = _collect_legend_lines([axes[1][0], axes[1][1]])

    if rare_handles_map:
        labels = list(rare_handles_map.keys())
        handles = [rare_handles_map[lbl] for lbl in labels]
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.78, 0.98),
            frameon=False,
            fontsize=9,
            title="Rare",
        )

    if pop_handles_map:
        labels = list(pop_handles_map.keys())
        handles = [pop_handles_map[lbl] for lbl in labels]
        fig.legend(
            handles,
            labels,
            loc="lower left",
            bbox_to_anchor=(0.78, 0.02),
            frameon=False,
            fontsize=9,
            title="Popular",
        )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig, axes


def main(base_dir: Path) -> None:
    method_dirs = {
        "pop_static_wga": base_dir / "saves" / "unlearn" / "pop_static_wga",
        "wga": base_dir / "saves" / "unlearn" / "wga",
        # "pop_dynam_b_wga": base_dir / "saves" / "unlearn" / "pop_dynam_b_wga",
        "ada_WGD": base_dir / "saves" / "unlearn" / "ada_WGD",
    }

    df = load_runs(method_dirs)
    fig_path = base_dir / "saves" / "plots" / "duet_wga_family_equal_lr_spacing.png"
    plot_grid(df, fig_path)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize DUET ROUGE with equal-spaced lr categories"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Repository root (default: current dir)",
    )

    if "ipykernel" in sys.modules:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    main(args.base_dir)
