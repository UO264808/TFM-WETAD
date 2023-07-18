"""
Configuration file for global variables common for all experiments.
You can use it by importing this module in your experiment source code.
"""

N_TRAIN = 8000
N_TEST = 3000
BATCH_SIZE = 32
EMBED_SIZE = 300
RANDOM_SEED = 33
WINDOW_SIZE = 7
EPOCHS = 5
THRESHOLD_MODE = 'percentile'
THRESHOLD_VALUE = 95