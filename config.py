import os

# --- Performance settings ---
OMP_NUM_THREADS = "8"
N_CPU = 8

# --- Grid settings ---
GRID_X = 8
GRID_Y = 8

# --- Image paths ---
TEST_DIR = "tests"
TEST_PREFIX = "test0"

SRC_PATH = f"{TEST_DIR}/{TEST_PREFIX}_src.png"
MASK_PATH = f"{TEST_DIR}/{TEST_PREFIX}_mask.png"
TGT_PATH = f"{TEST_DIR}/{TEST_PREFIX}_target.png"

# --- Offsets ---
SRC_OFFSET = (0, 0)
TGT_OFFSET = (0, 0)

# Apply environment variables
os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS
