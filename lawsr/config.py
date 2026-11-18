"""Protocol constants.  Every value is stated in the manuscript; the section is given in the comment.

Values marked "implementation choice" are not fixed by the manuscript text; they are documented in
README.md and can be overridden from the command line of the experiment scripts.
"""

# Fragments (Section 3.4, Table 2)
FRAGMENT_LENGTH = 512
MAX_RETAINED_BLOCKS = 80
MIN_SOURCE_BYTES = 1024

# Content components, in the order of Eq. (14)
COMPONENTS = ('s_hist', 's_ent', 's_ngram', 's_comp')
COMPONENT_NAMES = {
    's_hist': 'Histogram cosine',
    's_ent': 'Entropy similarity',
    's_ngram': 'Byte 2-gram Jaccard',
    's_comp': 'Compression similarity',
}
ZLIB_LEVEL = 6                      # zlib wrapper, DEFLATE level 6 (Section 3.2, Eq. 10)

# Network (Section 3.3, Eqs. 19-21): two hidden layers, ReLU, sigmoid output
HIDDEN_UNITS = (64, 32)

# Learnability-aware weighting (Section 3.3, Eqs. 17-18)
TEMPERATURES = (0.05, 0.10, 0.20, 0.50, 1.00)

# Training (Section 3.7)
LEARNING_RATE = 1e-3                # Adam
BATCH_SIZE = 1024
MAX_EPOCHS = 200
PATIENCE = 10                       # epochs without improvement in validation MSE
SEEDS = (0, 1, 2, 3, 4)

# Input-scaling controls (Section 3.5, Table 6)
UNSCALED = (1.0, 1.0, 1.0, 1.0)
EQUAL_SCALES = (0.25, 0.25, 0.25, 0.25)
HEURISTIC_SCALES = (0.333, 0.133, 0.400, 0.133)   # fixed before model fitting; largest on byte 2-grams

# Retrieval (Sections 3.5 and 3.7)
TIE_ORDER_SEED = 42                 # fixed permutation of fragment identifiers, shared by all methods
TOP_K = (1, 10)
P_AT = 10                           # same-file precision at ten

# Optional type cue (Section 3.6)
TYPE_CORRUPTION_PERCENT = (5, 10, 20)   # 4, 8 and 15 of the 75 test files (nested selections)
TYPE_CUE_SCALE = 1.0                # implementation choice: scale of the fifth input (extension agreement)

# Controlled PNG gap repair (Sections 3.6-3.7)
PNG_MAX_CHECKS = 10                 # the validator tests at most the first ten proposed repairs
PNG_CHECKPOINT_SEED = 0             # implementation choice: "one fixed checkpoint for each learned method"

# Implementation choices for summarizing five seeds
LEARNABILITY_POLICY = 'mean'        # learnability from the mean validation R2 and Spearman over the seeds
TEMPERATURE_POLICY = 'mean'         # temperature with the lowest mean validation MSE over the seeds
