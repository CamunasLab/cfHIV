import os
import glob
import functools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"
import seaborn as sns
from matplotlib.patches import Patch

from ete3 import NCBITaxa

# =============================================================================
# Globals
# =============================================================================
ncbi = NCBITaxa()


# =============================================================================
# 1) File discovery + parsing
# =============================================================================
def build_file_list(domain="bacteria", include_plasma=True, include_controls=True):
    """
    Return a sorted list of unique *.counts filepaths.

    Parameters
    ----------
    domain : {"bacteria","viruses","archaea","all"}
        If "all", looks for blast/SAMPLE_*.counts
        Else, looks for blast/SAMPLE_*.{domain}.counts
    include_plasma : bool
        Whether to include SAMPLE_100* samples.
    include_controls : bool
        Whether to include NTC/water samples.

    Returns
    -------
    list[str]
        Sorted unique filepaths.
    """
    if domain == "all":
        files = sorted(glob.glob("blast/SAMPLE_*.counts"))
    else:
        files = sorted(glob.glob(f"blast/SAMPLE_*.{domain}.counts"))

    keep = []
    for f in files:
        b = os.path.basename(f)
        is_plasma = b.startswith("SAMPLE_100")
        is_control = ("NTC" in b) or ("water" in b.lower())

        if is_plasma and not include_plasma:
            continue
        if is_control and not include_controls:
            continue

        keep.append(f)

    return sorted(set(keep))


def is_placeholder_file(path, max_bytes=2048):
    """
    Return True if file is empty or just contains '""' (common placeholder case).
    """
    with open(path, "rb") as fh:
        txt = fh.read(max_bytes).decode("utf-8", errors="ignore").strip()
    return txt == "" or txt == '""'


def read_counts_as_series(path):
    """
    Read one *.counts file into a pandas Series.

    - If placeholder/empty -> empty Series
    - Sums across columns and divides by 2 (original behavior)
    - Drops NA accessions, ensures string index
    - Collapses duplicate accessions by summing
    """
    if is_placeholder_file(path):
        return pd.Series(dtype=float)

    df = pd.read_table(path, index_col=0)
    if df.shape[0] == 0:
        return pd.Series(dtype=float)

    s = df.sum(axis=1) / 2.0
    s = s[~s.index.isna()]
    s.index = s.index.astype(str)

    if s.index.has_duplicates:
        s = s.groupby(level=0).sum()

    return s


def build_counts_for_domain(DOMAIN, ia, drop_bad_sample="SAMPLE_100200_1"):
    """
    Build a counts matrix collapsed to:
      accession -> taxid

    Parameters
    ----------
    DOMAIN : str
        "bacteria", "viruses", ...
    ia : pd.DataFrame
        Mapping accession -> taxid, expects:
          - ia.index = accession strings
          - ia["taxid1"] = integer taxid
    drop_bad_sample : str
        Basename to exclude entirely.

    Returns
    -------
    counts_taxid : pd.DataFrame
        Rows = taxid, cols = sample name, values = counts
    meta : pd.DataFrame
        Index = sample names, column "group" in {"plasma","water","ntc"}
    use_files : list[str]
        Files used
    """
    use_files = build_file_list(domain=DOMAIN, include_plasma=True, include_controls=True)
    use_files = [f for f in use_files if drop_bad_sample not in os.path.basename(f)]

    samples = pd.Index([os.path.basename(f).split(".")[0] for f in use_files])
    samples = samples[~samples.duplicated(keep="first")]

    # Accession x sample matrix
    counts_acc = pd.DataFrame(0.0, index=ia.index, columns=samples)

    for f in use_files:
        sname = os.path.basename(f).split(".")[0]
        if sname not in counts_acc.columns:
            continue

        s = read_counts_as_series(f)
        if len(s) == 0:
            continue

        s = s.loc[s.index.intersection(counts_acc.index)]
        if len(s) == 0:
            continue

        counts_acc[sname] = counts_acc[sname].add(s, fill_value=0.0)

    counts_acc = counts_acc.loc[counts_acc.sum(axis=1) > 0]

    # Collapse accession -> taxid
    counts_acc["taxid"] = ia.loc[counts_acc.index, "taxid1"].values
    counts_taxid = counts_acc.groupby("taxid").sum()
    counts_taxid = counts_taxid.loc[counts_taxid.sum(axis=1) > 0]

    # Metadata
    meta = pd.DataFrame(index=counts_taxid.columns)
    meta["group"] = "plasma"
    meta.loc[meta.index.str.contains("NTC", case=False), "group"] = "ntc"
    meta.loc[meta.index.str.contains("water", case=False), "group"] = "water"

    return counts_taxid, meta, use_files


# =============================================================================
# 2) Taxonomy helpers
# =============================================================================
@functools.lru_cache(maxsize=200000)
def taxid_to_rank_taxid(taxid, rank="genus"):
    """
    Map a taxid -> taxid at requested rank (genus/family/etc).

    Returns np.nan on any failure.
    """
    try:
        lineage = ncbi.get_lineage(int(taxid))
        ranks = ncbi.get_rank(lineage)
        for t in lineage:
            if ranks.get(t) == rank:
                return t
    except Exception:
        return np.nan

    return np.nan


def collapse_to_rank(counts_taxid_df, rank):
    """
    Collapse a taxid-indexed count matrix to a higher rank (genus/family/etc).

    Output index is the *name* at that rank (human-readable).
    """
    taxa = counts_taxid_df.index.to_series()
    rank_taxids = taxa.apply(lambda t: taxid_to_rank_taxid(int(t), rank=rank))

    x = counts_taxid_df.copy()
    x["rank_taxid"] = rank_taxids.values
    x = x.dropna(subset=["rank_taxid"]).groupby("rank_taxid").sum()

    names = ncbi.get_taxid_translator(list(x.index.astype(int)))
    x.index = [names.get(int(t), str(t)) for t in x.index]
    return x


# =============================================================================
# 4) Plotting: volcano + clustermaps
# =============================================================================
def volcano_plot(res, fdr_thr=0.05, lfc_thr=1.0, label_top=10):
    """
    Basic volcano plot (no family coloring). Labels top hits by FDR.
    """
    from adjustText import adjust_text

    df = res.copy()
    sns.set_theme(font_scale=2)

    df["FDR"] = pd.to_numeric(df["FDR"], errors="coerce")
    df["PValue"] = pd.to_numeric(df["PValue"], errors="coerce")
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
    df = df.dropna(subset=["logFC", "PValue", "FDR"])

    df["neglog10p"] = -np.log10(df["PValue"].clip(lower=1e-300))
    sig = (df["FDR"] < fdr_thr) & (df["logFC"].abs() >= lfc_thr)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(df.loc[~sig, "logFC"], df.loc[~sig, "neglog10p"], s=18, alpha=0.35)
    ax.scatter(df.loc[sig, "logFC"], df.loc[sig, "neglog10p"], s=24, alpha=0.9)

    ax.axvline(-lfc_thr, linestyle="--", color="grey")
    ax.axvline(lfc_thr, linestyle="--", color="grey")
    ax.axhline(-np.log10(fdr_thr), linestyle="--", color="grey")

    ax.set_xlabel("log2 fold-change (plasma vs water)")
    ax.set_ylabel("-log10(PValue)")
    ax.set_title("Volcano plot bnabs")

    top = df.sort_values("FDR").head(label_top)
    texts = []
    for name, row in top.iterrows():
        texts.append(ax.text(row["logFC"], row["neglog10p"], str(name), fontsize=10))

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", lw=0.5, color="grey"),
        expand_points=(1.2, 1.3),
        expand_text=(1.2, 1.3),
    )

    fig.tight_layout()
    plt.show()
    return fig


def volcano_plot_fdr(res, fdr_thr=0.05, lfc_thr=1.0, label_top=10):
    """
    Volcano plot styled like the 'right' example:
    - white background, no grid
    - thick black axes/spines
    - gray nonsig points, orange sig points
    - thicker dashed threshold lines
    - labels with leader lines (uses adjustText if available)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Optional: use adjustText if installed
    try:
        from adjustText import adjust_text
        have_adjust = True
    except ImportError:
        have_adjust = False

    df = res.copy()
    df["FDR"] = pd.to_numeric(df["FDR"], errors="coerce")
    df["PValue"] = pd.to_numeric(df["PValue"], errors="coerce")
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
    df = df.dropna(subset=["logFC", "FDR"])

    df["neglog10fdr"] = -np.log10(df["FDR"].clip(lower=1e-300))
    sig = (df["FDR"] < fdr_thr) & (df["logFC"].abs() >= lfc_thr)

    # ---- Figure / style to match right ----
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.set_facecolor("white")
    ax.grid(False)

    # Thicker spines (axes border)
    for sp in ax.spines.values():
        sp.set_linewidth(3)
        sp.set_color("black")

    ax.tick_params(axis="both", which="both", width=2.5, length=6, labelsize=18)

    # Points: gray for nonsig, orange for sig
    ax.scatter(
        df.loc[~sig, "logFC"], df.loc[~sig, "neglog10fdr"],
        s=14, color="#171616", alpha=0.55, linewidth=0
    )
    ax.scatter(
        df.loc[sig, "logFC"], df.loc[sig, "neglog10fdr"],
        s=34, color="orange", alpha=0.95, edgecolor="none"
    )

    # Threshold lines (thicker dashed)
    thr_color = "#7a7a7a"
    ax.axvline(-lfc_thr, linestyle="--", color=thr_color, linewidth=3, zorder=0)
    ax.axvline(lfc_thr,  linestyle="--", color=thr_color, linewidth=3, zorder=0)
    ax.axhline(-np.log10(fdr_thr), linestyle="--", color=thr_color, linewidth=3, zorder=0)

    # Labels/title like right
    ax.set_xlabel("log2 fold-change\n(plasma vs NC)", fontsize=22)
    ax.set_ylabel("-log10(FDR)", fontsize=22)
    ax.set_title("")  # right plot effectively has no big title inside axes

    # ---- Label top hits (usually label among significant) ----
    # If you want EXACTLY like right: label top by FDR (or top sig by FDR)
    top = df.loc[sig].sort_values("FDR").head(label_top) if sig.any() else df.sort_values("FDR").head(label_top)

    texts = []
    for name, row in top.iterrows():
        texts.append(
            ax.text(
                row["logFC"], row["neglog10fdr"], str(name),
                fontsize=16, color="#4a4a4a", zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.9)
            )
        )

    if have_adjust and texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", lw=2, color="#7a7a7a"),
            expand_points=(1.4, 1.6),
            expand_text=(1.2, 1.4),
        )
    elif (not have_adjust) and texts:
        # simple fallback: nudge labels a bit so they’re readable
        for t in texts:
            x, y = t.get_position()
            t.set_position((x + 0.15, y + 0.2))

    fig.tight_layout()
    sns.despine()
    plt.show()
    return fig, ax

def clustermap_hits(
    logcpm, meta_pw, res,
    min_abs_logfc=1.0,
    fdr_thr=0.05,
    cmap = "magma",
    metric="correlation",
    method="average",
    figsize=(10, 10),
    cluster_cols=False,
):
    """
    Heatmap of edgeR hits (plasma vs water), with group color bar.
    """
    sns.set_theme(font_scale=1)

    hits = res.index[(res["FDR"] < fdr_thr) & (res["logFC"].abs() >= min_abs_logfc)]
    order = meta_pw.sort_values("group").index

    X = logcpm.loc[logcpm.index.intersection(hits), order]
    Xz = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1) + 1e-6, axis=0)

    lut = {"plasma": "red", "water": "blue"}
    col_colors = meta_pw.loc[order, "group"].map(lut)

    g = sns.clustermap(
        Xz,
        col_colors=col_colors,
        metric=metric,
        method=method,
        figsize=figsize,
        center=0,
        vmin=-1,
        vmax=1,
        cmap=cmap,
        xticklabels=False,
        yticklabels=True,
        col_cluster=cluster_cols,
        row_cluster=True,
        linewidths=0,
    )

    legend_elements = [
        Patch(facecolor=lut["plasma"], label="Plasma"),
        Patch(facecolor=lut["water"], label="Water"),
    ]
    g.ax_col_dendrogram.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
    )

    g.ax_heatmap.set_xlabel("Samples")
    g.ax_heatmap.set_ylabel("Genera")
    plt.show()


def clustermap_bnabs(
    logcpm,
    design_df,
    res,
    min_abs_logfc=2.0,
    fdr_thr=0.05,
    metric="correlation",
    method="average",
    figsize=(10, 10),
    cluster_cols=True,
    sort_by="bNAbs",
    annotate_location=True,
):
    """
    Plasma-only heatmap for bNAbs DE hits, annotated by bNAbs and optionally Location.
    """
    sns.set_theme(font_scale=1)

    hits = res.index[(res["FDR"] < fdr_thr) & (res["logFC"].abs() >= min_abs_logfc)]
    hits = logcpm.index.intersection(hits)
    if len(hits) == 0:
        raise ValueError("No taxa pass thresholds. Try lowering min_abs_logfc or relaxing fdr_thr.")

    d = design_df.copy()
    if isinstance(d.index, pd.MultiIndex):
        d.index = d.index.map(lambda x: "_".join(map(str, x)))

    d["bNAbs"] = d["bNAbs"].astype(str)
    if "Location" in d.columns:
        d["Location"] = d["Location"].astype(str)

    order = d.sort_values(sort_by).index if sort_by in d.columns else d.index
    order = pd.Index(order)

    X = logcpm.loc[hits, logcpm.columns.intersection(order)]
    X = X.loc[:, order.intersection(X.columns)]

    Xz = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1) + 1e-6, axis=0)

    bn_levels = pd.Index(d.loc[order, "bNAbs"]).unique().tolist()
    palette_bn = {"Y": "#7B3294", "N": "#008837"}
    bn_colors = d.loc[order, "bNAbs"].map(palette_bn)

    col_colors = pd.DataFrame({"bNAbs": bn_colors}, index=order)
    legend_elements = [Patch(facecolor=palette_bn[k], label=f"bNAbs: {k}") for k in bn_levels]

    if annotate_location and ("Location" in d.columns):
        loc_levels = pd.Index(d.loc[order, "Location"]).unique().tolist()
        palette_loc = dict(zip(loc_levels, sns.color_palette(n_colors=len(loc_levels))))
        loc_colors = d.loc[order, "Location"].map(palette_loc)
        col_colors["Location"] = loc_colors
        legend_elements += [Patch(facecolor=palette_loc[k], label=f"Loc: {k}") for k in loc_levels]

    g = sns.clustermap(
        Xz,
        col_colors=col_colors,
        metric=metric,
        method=method,
        figsize=figsize,
        center=0,
        vmin=-2,
        vmax=2,
        xticklabels=False,
        yticklabels=True,
        col_cluster=cluster_cols,
    )

    g.ax_heatmap.set_xlabel("Plasma samples")
    g.ax_heatmap.set_ylabel("Genera")
    g.ax_col_dendrogram.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(1, 1.25),
        ncol=3,
        frameon=False,
    )

    plt.show()


# =============================================================================
# 5) Family annotation + family-colored volcano
# =============================================================================
def build_name_to_taxid_map(names, rank="genus"):
    """
    Build dict {name -> taxid} using NCBI name translator, preferring matching `rank`.
    """
    names = [str(n) for n in names]
    d = ncbi.get_name_translator(names)

    out = {}
    for name in names:
        taxids = d.get(name, [])
        if not taxids:
            out[name] = np.nan
            continue

        ranks = ncbi.get_rank(taxids)
        best = next((t for t in taxids if ranks.get(t) == rank), taxids[0])
        out[name] = best

    return out


def add_family_to_res_from_map(res, name_to_taxid_map, input_rank="genus"):
    """
    Add family annotations to a result table indexed by genus name.
    """
    out = res.copy()

    out["genus_taxid"] = [name_to_taxid_map.get(str(n), np.nan) for n in out.index]
    out["family_taxid"] = out["genus_taxid"].apply(
        lambda t: taxid_to_rank_taxid(int(t), rank="family") if pd.notna(t) else np.nan
    )

    fam_taxids = sorted({int(t) for t in out["family_taxid"].dropna().unique()})
    fam_names = ncbi.get_taxid_translator(fam_taxids) if fam_taxids else {}

    out["family"] = out["family_taxid"].apply(
        lambda t: fam_names.get(int(t), "Unknown") if pd.notna(t) else "Unknown"
    )

    return out


def add_phylum_to_res_from_map(res, name_to_taxid_map, input_rank="genus"):
    """
    Add family annotations to a result table indexed by genus name.
    """
    out = res.copy()

    out["genus_taxid"] = [name_to_taxid_map.get(str(n), np.nan) for n in out.index]
    out["phylum_taxid"] = out["genus_taxid"].apply(
        lambda t: taxid_to_rank_taxid(int(t), rank="phylum") if pd.notna(t) else np.nan
    )

    fam_taxids = sorted({int(t) for t in out["phylum_taxid"].dropna().unique()})
    fam_names = ncbi.get_taxid_translator(fam_taxids) if fam_taxids else {}

    out["phylum"] = out["phylum_taxid"].apply(
        lambda t: fam_names.get(int(t), "Unknown") if pd.notna(t) else "Unknown"
    )

    return out


def add_class_to_res_from_map(res, name_to_taxid_map, input_rank="genus"):
    """
    Add family annotations to a result table indexed by genus name.
    """
    out = res.copy()

    out["genus_taxid"] = [name_to_taxid_map.get(str(n), np.nan) for n in out.index]
    out["class_taxid"] = out["genus_taxid"].apply(
        lambda t: taxid_to_rank_taxid(int(t), rank="class") if pd.notna(t) else np.nan
    )

    fam_taxids = sorted({int(t) for t in out["class_taxid"].dropna().unique()})
    fam_names = ncbi.get_taxid_translator(fam_taxids) if fam_taxids else {}

    out["class"] = out["class_taxid"].apply(
        lambda t: fam_names.get(int(t), "Unknown") if pd.notna(t) else "Unknown"
    )

    return out


def build_shared_family_palette(
    res_list,
    fdr_thr=0.05,
    lfc_thr=1.0,
    top_n_families=12,
    base_palette="tab20b",
    other_color="lightgrey",
    unknown_color="silver",
):
    """
    Make a palette based on the most frequent significant families across multiple res tables.
    """
    fam_counts = {}

    for res in res_list:
        df = res.copy()
        df["FDR"] = pd.to_numeric(df.get("FDR"), errors="coerce")
        df["PValue"] = pd.to_numeric(df.get("PValue"), errors="coerce")
        df["logFC"] = pd.to_numeric(df.get("logFC"), errors="coerce")
        df = df.dropna(subset=["logFC", "PValue"])

        if "family" not in df.columns:
            df["family"] = "Unknown"

        sig = (df["FDR"] < fdr_thr) & (df["logFC"].abs() >= lfc_thr) if "FDR" in df.columns else False
        vc = df.loc[sig, "family"].value_counts() if np.any(sig) else pd.Series(dtype=int)

        for fam, n in vc.items():
            fam_counts[fam] = fam_counts.get(fam, 0) + int(n)

    fam_sorted = sorted(fam_counts.items(), key=lambda x: (-x[1], x[0]))
    top_fams = [f for f, _ in fam_sorted[:top_n_families]]

    colors = sns.color_palette(base_palette, n_colors=max(len(top_fams), 1))
    palette = {fam: col for fam, col in zip(top_fams, colors)}
    palette["Other"] = other_color
    palette["Unknown"] = unknown_color

    return palette, top_fams


def apply_family_plot_bucket(res, top_fams):
    """
    Create/overwrite res["family_plot"] to bucket families into top families vs "Other".
    """
    df = res.copy()
    if "family" not in df.columns:
        df["family"] = "Unknown"
    df["family_plot"] = np.where(df["family"].isin(top_fams), df["family"], "Other")
    return df


def set_volcano_style(font_scale=1.0):
    sns.set_theme(style="white", context="notebook", font_scale=font_scale)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["text.usetex"] = False


def volcano_masks(df, mode="p_fc", fdr_thr=0.05, p_thr=1e-6, lfc_thr=2.0):
    """
    Decide which points are "highlighted".

    mode:
      - "fdr_lfc": (FDR < fdr_thr) AND (|logFC| >= lfc_thr)
      - "p_fc":    (PValue < p_thr) AND (|logFC| >= lfc_thr)
    """
    if mode == "fdr_lfc":
        return (df["FDR"] < fdr_thr) & (df["logFC"].abs() >= lfc_thr)
    if mode == "p_fc":
        return (df["PValue"] < p_thr) & (df["logFC"].abs() >= lfc_thr)
    raise ValueError("mode must be 'fdr_lfc' or 'p_fc'")


def volcano_plot_family_fixed(
    res,
    palette,
    title="Volcano",
    mode="p_fc",
    fdr_thr=0.05,
    p_thr=1e-6,
    lfc_thr=2.0,
    show_cutoffs=False,
    label_top=0,
    label_col="name",
    figsize=(9, 7),
    highlight_families=None,
    keep_unknown=True,
):
    """
    Family-colored volcano plot.

    - black circles: non-highlighted points
    - colored squares: highlighted points, colored by df["family_plot"]
    - if highlight_families is provided: set family_plot to family/Other/Unknown accordingly
    """
    set_volcano_style(font_scale=1.0)

    df = res.copy()
    for c in ["FDR", "PValue", "logFC"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["logFC", "PValue"])

    if "FDR" not in df.columns:
        df["FDR"] = np.nan

    df["neglog10p"] = -np.log10(df["PValue"].clip(lower=1e-300))

    if "family" not in df.columns:
        df["family"] = "Unknown"

    # Build family_plot if requested
    if highlight_families is not None:
        highlight_families = set(highlight_families)

        if keep_unknown:
            df["family_plot"] = np.where(
                df["family"].isin(highlight_families),
                df["family"],
                np.where(df["family"] == "Unknown", "Unknown", "Other"),
            )
        else:
            df["family_plot"] = np.where(
                df["family"].isin(highlight_families),
                df["family"],
                "Other",
            )
    else:
        if "family_plot" not in df.columns:
            df["family_plot"] = df["family"]

    hi = volcano_masks(df, mode=mode, fdr_thr=fdr_thr, p_thr=p_thr, lfc_thr=lfc_thr)

    fig, ax = plt.subplots(figsize=figsize)

    # Background (non-highlight)
    ax.scatter(
        df.loc[~hi, "logFC"],
        df.loc[~hi, "neglog10p"],
        s=10,
        alpha=1,
        color="#000000",
        edgecolor="none",
        marker="o",
        rasterized=True,
    )

    # Legend order control: families first, then Unknown, then Other
    levels = pd.Index(df.loc[hi, "family_plot"].dropna().unique())
    core = sorted([x for x in levels if x not in ("Unknown", "Other")])

    hue_order = core[:]
    if "Unknown" in levels:
        hue_order.append("Unknown")
    if "Other" in levels:
        hue_order.append("Other")

    # If you want categorical ordering, keep your edited approach:
    # df["family_plot"] = pd.Categorical(df["family_plot"], categories=hue_order, ordered=True)

    sns.scatterplot(
        data=df.loc[hi],
        x="logFC",
        y="neglog10p",
        hue="family_plot",
        hue_order=hue_order,
        palette=palette,
        s=36,
        alpha=1,
        ax=ax,
        linewidth=0,
        marker="s",
    )

    if show_cutoffs:
        if mode == "fdr_lfc" and np.isfinite(fdr_thr):
            ax.axhline(-np.log10(fdr_thr), linestyle="--", color="0.5", linewidth=1)
        if mode == "p_fc":
            ax.axhline(-np.log10(p_thr), linestyle="--", color="0.5", linewidth=1)
        ax.axvline(-lfc_thr, linestyle="--", color="0.5", linewidth=1)
        ax.axvline(lfc_thr, linestyle="--", color="0.5", linewidth=1)

    ax.set_xlabel("log2 fold-change")
    ax.set_ylabel("-log10(PValue)")
    ax.set_title(title)
    ax.legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    if label_top and label_top > 0:
        dlab = df.loc[hi].sort_values("PValue").head(label_top)
        for idx, r in dlab.iterrows():
            lab = r.get(label_col, str(idx))
            ax.text(r["logFC"], r["neglog10p"], str(lab), fontsize=9)

    fig.tight_layout()
    sns.despine()
    fig.savefig(f"figure_{title}.svg", format="svg")
    plt.show()


def volcano_plot_family_fixed2(
    res,
    palette,
    title="Volcano",
    mode="p_fc",
    fdr_thr=0.05,
    p_thr=1e-6,
    lfc_thr=2.0,
    show_cutoffs=False,
    label_top=0,
    label_col="name",
    figsize=(9, 7),
    highlight_families=None,
    keep_unknown=True,
):
    """
    Family-colored volcano plot.

    - black circles: non-highlighted points
    - colored squares: highlighted points, colored by df["family_plot"]
    - if highlight_families is provided: set family_plot to family/Other/Unknown accordingly
    """
    set_volcano_style(font_scale=1.0)

    df = res.copy()
    for c in ["FDR", "PValue", "logFC"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["logFC", "PValue"])

    if "FDR" not in df.columns:
        df["FDR"] = np.nan

    df["neglog10p"] = -np.log10(df["FDR"].clip(lower=1e-300))

    if "family" not in df.columns:
        df["family"] = "Unknown"

    # Build family_plot if requested
    if highlight_families is not None:
        highlight_families = set(highlight_families)

        if keep_unknown:
            df["family_plot"] = np.where(
                df["family"].isin(highlight_families),
                df["family"],
                np.where(df["family"] == "Unknown", "Unknown", "Other"),
            )
        else:
            df["family_plot"] = np.where(
                df["family"].isin(highlight_families),
                df["family"],
                "Other",
            )
    else:
        if "family_plot" not in df.columns:
            df["family_plot"] = df["family"]

    hi = volcano_masks(df, mode=mode, fdr_thr=fdr_thr, p_thr=p_thr, lfc_thr=lfc_thr)

    fig, ax = plt.subplots(figsize=figsize)

    # Background (non-highlight)
    ax.scatter(
        df.loc[~hi, "logFC"],
        df.loc[~hi, "neglog10p"],
        s=10,
        alpha=1,
        color="#000000",
        edgecolor="none",
        marker="o",
        rasterized=True,
    )

    # Legend order control: families first, then Unknown, then Other
    levels = pd.Index(df.loc[hi, "family_plot"].dropna().unique())
    core = sorted([x for x in levels if x not in ("Unknown", "Other")])

    hue_order = core[:]
    if "Unknown" in levels:
        hue_order.append("Unknown")
    if "Other" in levels:
        hue_order.append("Other")

    # If you want categorical ordering, keep your edited approach:
    # df["family_plot"] = pd.Categorical(df["family_plot"], categories=hue_order, ordered=True)

    sns.scatterplot(
        data=df.loc[hi],
        x="logFC",
        y="neglog10p",
        hue="family_plot",
        hue_order=hue_order,
        palette=palette,
        s=36,
        alpha=1,
        ax=ax,
        linewidth=0,
        marker="s",
    )

    if show_cutoffs:
        if mode == "fdr_lfc" and np.isfinite(fdr_thr):
            ax.axhline(-np.log10(fdr_thr), linestyle="--", color="0.5", linewidth=1)
        if mode == "p_fc":
            ax.axhline(-np.log10(p_thr), linestyle="--", color="0.5", linewidth=1)
        ax.axvline(-lfc_thr, linestyle="--", color="0.5", linewidth=1)
        ax.axvline(lfc_thr, linestyle="--", color="0.5", linewidth=1)

    ax.set_xlabel("log2 fold-change")
    ax.set_ylabel("-log10(PValue)")
    ax.set_title(title)
    ax.legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    if label_top and label_top > 0:
        dlab = df.loc[hi].sort_values("PValue").head(label_top)
        for idx, r in dlab.iterrows():
            lab = r.get(label_col, str(idx))
            ax.text(r["logFC"], r["neglog10p"], str(lab), fontsize=9)

    fig.tight_layout()
    sns.despine()
    fig.savefig(f"figure_{title}.svg", format="svg")
    plt.show()

def ensure_palette_has_families(palette, families, base_palette="tab20"):
    """
    Add colors to `palette` for any families missing from it.
    """
    palette = dict(palette)
    missing = [f for f in sorted(set(families)) if f not in palette]

    if missing:
        extra_cols = sns.color_palette(base_palette, n_colors=len(missing))
        for fam, col in zip(missing, extra_cols):
            palette[fam] = col

    return palette


# =============================================================================
# 6) Extra clustermap + z-score helpers + classification
# =============================================================================
def clustermap_shared_pw(
    logcpm_pw,
    meta_pw,
    shared_taxa,
    figsize=(10, 6),
    metric="correlation",
    method="average",
    col_cluster=False,
):
    taxa = logcpm_pw.index.intersection(shared_taxa)
    if len(taxa) == 0:
        raise ValueError("No shared taxa found in logcpm_pw index.")

    order = meta_pw.sort_values("group").index
    X = logcpm_pw.loc[taxa, order]
    Xz = X.sub(X.mean(axis=1), axis=0).div(X.std(axis=1) + 1e-6, axis=0)

    lut = {"plasma": "orange", "water": "blue"}
    col_colors = meta_pw.loc[order, "group"].map(lut)

    g = sns.clustermap(
        Xz,
        col_colors=col_colors,
        metric=metric,
        method=method,
        figsize=figsize,
        vmin=-2,
        vmax=2,
        center=0,
        xticklabels=False,
        yticklabels=True,
        col_cluster=col_cluster,
    )

    legend_elements = [
        Patch(facecolor=lut["plasma"], label="Plasma"),
        Patch(facecolor=lut["water"], label="Water"),
    ]
    g.ax_col_dendrogram.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
    )

    g.ax_heatmap.set_xlabel("Samples")
    g.ax_heatmap.set_ylabel("Shared genera")
    plt.suptitle("Significant between both (fdr<0.05, lfc>=1 both models)", fontsize=20, y=0.9)
    plt.show()


def group_z_from_counts(
    counts,
    meta,
    taxa_names,
    group_col="group",
    g1="plasma",
    g0="water",
    log1p=True,
):
    """
    Compute per-taxon standardized difference (plasma - water) from raw counts.
    """
    common_samples = counts.columns.intersection(meta.index)
    X = counts.loc[:, common_samples]
    meta2 = meta.loc[common_samples].copy()

    keep = meta2[group_col].isin([g0, g1])
    meta2 = meta2.loc[keep]
    X = X.loc[:, meta2.index]

    taxa = X.index.intersection(pd.Index(list(taxa_names)))
    if len(taxa) == 0:
        raise ValueError("None of the requested taxa_names were found in counts index.")
    X = X.loc[taxa]

    Xv = np.log1p(X) if log1p else X.astype(float)

    plasma_cols = meta2.index[meta2[group_col].eq(g1)]
    water_cols = meta2.index[meta2[group_col].eq(g0)]

    mu_plasma = Xv[plasma_cols].mean(axis=1)
    mu_water = Xv[water_cols].mean(axis=1)
    sd_plasma = Xv[plasma_cols].std(axis=1, ddof=1)
    sd_water = Xv[water_cols].std(axis=1, ddof=1)

    n1 = len(plasma_cols)
    n0 = len(water_cols)

    sd_pooled = np.sqrt(((n1 - 1) * (sd_plasma**2) + (n0 - 1) * (sd_water**2)) / (n1 + n0 - 2))
    sd_pooled = sd_pooled.replace(0, np.nan)

    z = (mu_plasma - mu_water) / sd_pooled

    out = pd.DataFrame(
        {
            "mu_plasma": mu_plasma,
            "mu_water": mu_water,
            "delta": (mu_plasma - mu_water),
            "sd_pooled": sd_pooled,
            "z_plasma_minus_water": z,
            "n_plasma": n1,
            "n_water": n0,
        }
    ).sort_values("z_plasma_minus_water", ascending=False)

    return out


def zscore_relative_to_water(
    counts,
    meta,
    taxa_names,
    group_col="group",
    water_label="water",
    plasma_label="plasma",
    log1p=True,
):
    """
    Z-score each taxon relative to the water distribution.
    """
    common_samples = counts.columns.intersection(meta.index)
    X = counts.loc[:, common_samples]
    meta2 = meta.loc[common_samples].copy()

    meta2 = meta2[meta2[group_col].isin([water_label, plasma_label])]
    X = X.loc[:, meta2.index]

    taxa = X.index.intersection(pd.Index(list(taxa_names)))
    X = X.loc[taxa]

    Xv = np.log1p(X) if log1p else X.astype(float)

    water_cols = meta2.index[meta2[group_col].eq(water_label)]
    mu_w = Xv[water_cols].mean(axis=1)
    sd_w = Xv[water_cols].std(axis=1, ddof=1).replace(0, np.nan)

    Z = Xv.sub(mu_w, axis=0).div(sd_w, axis=0)
    Z = Z.replace([np.inf, -np.inf], np.nan)

    return Z, meta2


from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def classify_genera(
    df_long,
    up_thr=0.3,
    alpha=0.05,
    use_fdr=True,
    group_col="group3",
    genus_col="Genus",
    value_col="value",
    water_label="water",
    bn_minus_label="plasma_bNAbs-",
    bn_plus_label="plasma_bNAbs+",
):
    """
    Classify genera into:
      - up_in_all_plasma: both bNAbs- and bNAbs+ are up vs water
      - up_only_in_bNAbs+: only bNAbs+ is up vs water
      - not_up: everything else
    """
    needed = {water_label, bn_minus_label, bn_plus_label}
    present = set(df_long[group_col].dropna().unique())
    missing = needed - present
    if missing:
        raise ValueError(f"Missing group labels in df_long[{group_col}]: {missing}")

    results = []

    for genus, d in df_long.groupby(genus_col):
        w = d.loc[d[group_col] == water_label, value_col].dropna().values
        m = d.loc[d[group_col] == bn_minus_label, value_col].dropna().values
        p = d.loc[d[group_col] == bn_plus_label, value_col].dropna().values

        if len(w) < 2 or len(m) < 2 or len(p) < 2:
            continue

        med_w = np.median(w)
        med_m = np.median(m)
        med_p = np.median(p)
        eff_m = med_m - med_w
        eff_p = med_p - med_w

        p_m = mannwhitneyu(m, w, alternative="greater").pvalue
        p_p = mannwhitneyu(p, w, alternative="greater").pvalue

        results.append(
            {
                "Genus": genus,
                "n_water": len(w),
                "n_bNAbs-": len(m),
                "n_bNAbs+": len(p),
                "median_water": med_w,
                "median_bNAbs-": med_m,
                "median_bNAbs+": med_p,
                "effect_bNAbs-": eff_m,
                "effect_bNAbs+": eff_p,
                "p_bNAbs-_gt_water": p_m,
                "p_bNAbs+_gt_water": p_p,
            }
        )

    res = pd.DataFrame(results).set_index("Genus")
    if res.empty:
        raise ValueError("No genera had enough samples per group to test.")

    if use_fdr:
        res["fdr_bNAbs-_gt_water"] = multipletests(res["p_bNAbs-_gt_water"], method="fdr_bh")[1]
        res["fdr_bNAbs+_gt_water"] = multipletests(res["p_bNAbs+_gt_water"], method="fdr_bh")[1]
        sig_m = res["fdr_bNAbs-_gt_water"] < alpha
        sig_p = res["fdr_bNAbs+_gt_water"] < alpha
    else:
        sig_m = res["p_bNAbs-_gt_water"] < alpha
        sig_p = res["p_bNAbs+_gt_water"] < alpha

    up_m = (res["effect_bNAbs-"] >= up_thr) & sig_m
    up_p = (res["effect_bNAbs+"] >= up_thr) & sig_p

    res["grouping"] = np.select(
        [up_m & up_p, (~up_m) & up_p],
        ["up_in_all_plasma", "up_only_in_bNAbs+"],
        default="not_up",
    )

    return res.sort_values(["grouping", "effect_bNAbs+"], ascending=[True, False])


def sample_set_median(X_centered, taxa_set):
    """
    Convenience helper: median across taxa_set for each sample.
    """
    if len(taxa_set) == 0:
        return pd.Series(dtype=float)
    return X_centered.loc[list(taxa_set)].median(axis=0)



def genus_water_metrics(df_long, genus_col="Genus", group_col="group3", value_col="value",
                        water_label="water", groups=("plasma_bNAbs-","plasma_bNAbs+")):
    rows = []
    for genus, d in df_long.groupby(genus_col):
        w = d.loc[d[group_col] == water_label, value_col].dropna().values
        if len(w) < 4:
            continue

        q1, q3 = np.percentile(w, [25, 75])
        iqr = q3 - q1
        med_w = np.median(w)
        sd_w = np.std(w, ddof=1)

        out = {
            "Genus": genus,
            "n_water": len(w),
            "water_median": med_w,
            "water_q1": q1,
            "water_q3": q3,
            "water_iqr": iqr,
            "water_sd": sd_w,
        }

        # for each plasma subgroup: median shift and robust separation
        for g in groups:
            x = d.loc[d[group_col] == g, value_col].dropna().values
            if len(x) < 3:
                out[f"n_{g}"] = len(x)
                out[f"median_{g}"] = np.nan
                out[f"shift_{g}"] = np.nan
                out[f"robustZ_{g}"] = np.nan
                out[f"frac_above_q3_{g}"] = np.nan
                continue

            med_g = np.median(x)
            shift = med_g - med_w

            # robust “z” using IQR (avoid SD blowups)
            # IQR ~ 1.349*SD for normal, so SD_robust ≈ IQR/1.349
            denom = (iqr / 1.349) if iqr > 0 else np.nan
            robustZ = shift / denom if denom and np.isfinite(denom) and denom > 0 else np.nan

            frac_above_q3 = np.mean(x > q3)

            out[f"n_{g}"] = len(x)
            out[f"median_{g}"] = med_g
            out[f"shift_{g}"] = shift
            out[f"robustZ_{g}"] = robustZ
            out[f"frac_above_q3_{g}"] = frac_above_q3

        rows.append(out)

    return pd.DataFrame(rows).set_index("Genus")
