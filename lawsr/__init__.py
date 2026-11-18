"""LAWSR: learnability-aware weighted similarity regression for file-fragment analysis.

Modules map onto the manuscript as follows:
    similarity  - byte-level components, Eqs. (7)-(11); fixed fusions, Eqs. (15)-(16)
    reference   - provenance reference, Eq. (23); type and positional cues, Eqs. (12)-(13)
    weighting   - learnability index and softmax weights, Eqs. (17)-(18)
    model       - the 64-32 regression network, Eqs. (19)-(21)
    training    - training with Adam, early stopping and checkpoint selection, Eq. (22), Algorithm 1
    metrics     - pair metrics, Eq. (24); AUC, AP, Pearson, Spearman, within-file Spearman
    retrieval   - exact-successor ranking with a fixed tie order, Eq. (25)
    sampling    - pair sampling in four equally represented modes (Govdocs1 external pairs)
    png_validator - validator for the controlled PNG gap repair
    data        - loaders for the dataset package
"""
__version__ = '1.0.0'
