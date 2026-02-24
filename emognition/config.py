"""
Configuration for Emognition + Dual-Branch EEGNet-BiLSTM training.
"""

# ===================== PATHS =====================
DATA_ROOT = "/kaggle/input/datasets/uvindukodikara/emognition"

# ===================== EEG SETTINGS =====================
FS = 256                    # Muse 2 sampling rate (Hz)
CHANNELS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
NUM_CHANNELS = 4
QUALITY_CHANNELS = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]

# ===================== WINDOWING =====================
WIN_SEC = 4.0               # window length in seconds
OVERLAP = 0.75              # 75% overlap → more training data
SAMPLE_LENGTH = int(WIN_SEC * FS)  # 1024

# ===================== LABEL SCHEME =====================
LABEL_MODE = "4class"
EMOTIONS_USED = ["ENTHUSIASM", "FEAR", "SADNESS", "NEUTRAL"]

# ===================== SUBJECT EXCLUSION =====================
# Add subject IDs here to exclude (e.g. noisy data, all-random predictions)
# Example: EXCLUDE_SUBJECTS = ["40", "55"]
EXCLUDE_SUBJECTS = []
# After first LOSO run, subjects with <30% accuracy will be suggested for exclusion

# ===================== EXPERIMENT =====================
EXPERIMENT_MODE = "subject-independent"
TEST_SIZE = 0.15
VAL_SIZE = 0.15
SEED = 42

# ===================== TRAINING =====================
BATCH_SIZE = 32
EPOCHS = 250
LR = 0.0003
WEIGHT_DECAY = 1e-3
PATIENCE = 35
LABEL_SMOOTHING = 0.1

# ===================== DATA AUGMENTATION =====================
AUG_NOISE_STD = 0.02
AUG_TIME_SHIFT = 64
AUG_CHANNEL_DROP_P = 0.1
AUG_SCALE_RANGE = (0.9, 1.1)

# ===================== INVBASE BASELINE REMOVAL =====================
USE_INVBASE = True
INVBASE_FEATURES = 10

# ===================== HANDCRAFTED FEATURES =====================
HANDCRAFTED_DIM = 26

# ===================== EEGNET (FEATURE EXTRACTOR) =====================
F1 = 16
D = 2                       # feature_dim = 32
EEG_DROPOUT = 0.4

# ===================== CROSS-ATTENTION FUSION =====================
FUSION_DIM = 64
NUM_HEADS = 4
FUSION_DROPOUT = 0.2

# ===================== BiLSTM (CLASSIFIER) =====================
LSTM_HIDDEN = 64
LSTM_LAYERS = 2
CLS_DROPOUT = 0.5

# ===================== DEVICE =====================
DEVICE = "cuda"
