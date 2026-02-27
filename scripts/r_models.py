import os
import glob
import functools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"
import seaborn as sns
from matplotlib.patches import Patch

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from ete3 import NCBITaxa




# =============================================================================
# 3) edgeR wrappers
# =============================================================================
def edger_plasma_vs_water(counts, meta):
    """
    edgeR DE: plasma vs water only.

    Returns
    -------
    res : pd.DataFrame
    logcpm : pd.DataFrame
    meta2 : pd.DataFrame (plasma+water only, aligned)
    """
    meta2 = meta.loc[counts.columns].copy()
    meta2 = meta2[meta2["group"].isin(["plasma", "water"])]

    X = counts.loc[:, meta2.index].round().astype(int)

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["X"] = X
        ro.globalenv["meta"] = meta2

    ro.r("""
      y <- read_counts_df(X)
      meta3 <- meta[colnames(y), , drop=FALSE]
      meta3$group <- relevel(factor(meta3$group), ref='water')
      design <- model.matrix(~ group, data=meta3)

      fit <- run_de(y, design)
      qlf <- glmQLFTest(fit, coef=2)
      res <- topTags(qlf, n=Inf)$table
      assign("res", res, envir=.GlobalEnv)

      lcpm <- cpm(y, log=TRUE, prior.count=1)
      assign("lcpm", lcpm, envir=.GlobalEnv)
      assign("lcpm_rownames", rownames(lcpm), envir=.GlobalEnv)
      assign("lcpm_colnames", colnames(lcpm), envir=.GlobalEnv)
    """)

    with localconverter(ro.default_converter + pandas2ri.converter):
        res = ro.conversion.rpy2py(ro.globalenv["res"])
        lcpm = ro.conversion.rpy2py(ro.globalenv["lcpm"])
        rn = list(ro.conversion.rpy2py(ro.globalenv["lcpm_rownames"]))
        cn = list(ro.conversion.rpy2py(ro.globalenv["lcpm_colnames"]))

    logcpm = pd.DataFrame(lcpm, index=rn, columns=cn) if isinstance(lcpm, np.ndarray) else lcpm
    return res, logcpm, meta2


def edger_bnabs_plasma(counts, meta, design_df, bNAbs_ref=None):
    """
    Plasma-only DE: taxa ~ Location + bNAbs

    Returns
    -------
    res : pd.DataFrame
    logcpm : pd.DataFrame
    d : pd.DataFrame (design aligned to plasma samples)
    """
    meta2 = meta.loc[counts.columns].copy()
    plasma_cols = meta2.index[meta2["group"].eq("plasma")]
    counts_plasma = counts.loc[:, plasma_cols]

    d = design_df.loc[counts_plasma.columns].copy()
    d["Location"] = d["Location"].astype("category")
    d["bNAbs"] = d["bNAbs"].astype("category")

    if d["bNAbs"].nunique() < 2:
        raise ValueError("bNAbs has <2 levels in these plasma samples (cannot test bNAbs).")

    X = counts_plasma.round().astype(int)

    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["X"] = X
        ro.globalenv["d"] = d

    if bNAbs_ref is not None:
        ro.globalenv["bNAbs_ref"] = bNAbs_ref
        set_ref = "d2$bNAbs <- relevel(factor(d2$bNAbs), ref=bNAbs_ref)"
    else:
        set_ref = "d2$bNAbs <- factor(d2$bNAbs)"

    ro.r(f"""
      y <- read_counts_df(X)
      d2 <- d[colnames(y), , drop=FALSE]
      d2$Location <- factor(d2$Location)
      {set_ref}

      design <- model.matrix(~ Location + bNAbs, data=d2)
      fit <- run_de(y, design)

      cn <- colnames(design)
      bn_cols <- grep("^bNAbs", cn)

      if (length(levels(d2$bNAbs)) == 2) {{
        qlf <- glmQLFTest(fit, coef=bn_cols[1])
        res <- topTags(qlf, n=Inf)$table
      }} else {{
        qlf <- glmQLFTest(fit, coef=bn_cols)
        res <- topTags(qlf, n=Inf)$table
      }}

      assign("res", res, envir=.GlobalEnv)

      lcpm <- cpm(y, log=TRUE, prior.count=1)
      assign("lcpm", lcpm, envir=.GlobalEnv)
      assign("lcpm_rownames", rownames(lcpm), envir=.GlobalEnv)
      assign("lcpm_colnames", colnames(lcpm), envir=.GlobalEnv)
    """)

    with localconverter(ro.default_converter + pandas2ri.converter):
        res = ro.conversion.rpy2py(ro.globalenv["res"])
        lcpm = ro.conversion.rpy2py(ro.globalenv["lcpm"])
        rn = list(ro.conversion.rpy2py(ro.globalenv["lcpm_rownames"]))
        cn = list(ro.conversion.rpy2py(ro.globalenv["lcpm_colnames"]))

    logcpm = pd.DataFrame(lcpm, index=rn, columns=cn) if isinstance(lcpm, np.ndarray) else lcpm
    return res, logcpm, d
