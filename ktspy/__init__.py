"""ktspy: k top scoring pairs (k-TSP) classifiers.

A top scoring pair is a pair of features (genes) whose ordering within a sample differs between
two classes (``gene_a > gene_b`` in one class, ``<`` in the other). A k-TSP classifier takes a
majority vote of k such pairs. Only the order of values *within a sample* matters, so the rules
are unchanged by any monotone per-sample transformation (library size, log, ...) and a single
sample can be classified.

* :class:`KTSPClassifier`, :class:`KTSP` -- two classes, port of Bioconductor ``switchBox``
* :class:`KTSPScore` -- the vote share as a feature (optionally out-of-fold) for a downstream model
* :class:`MultiKTSPClassifier` -- more than two classes (one-vs-one, soft voting), with
  :meth:`~MultiKTSPClassifier.explain` for the vote of every gene pair
"""

from .core import KTSP, KTSPClassifier, tsp_scores, wilcoxon_filter
from .multiclass import MultiKTSPClassifier
from .transformers import KTSPScore

__version__ = "0.1.0"
__all__ = ["KTSP", "KTSPClassifier", "MultiKTSPClassifier", "KTSPScore", "tsp_scores", "wilcoxon_filter"]
