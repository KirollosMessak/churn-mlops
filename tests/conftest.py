import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # harmless elsewhere; avoids an Anaconda/Windows OpenMP clash
