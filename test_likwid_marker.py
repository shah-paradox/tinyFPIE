"""Check what LIKWID env vars are visible to the Python process."""
import os
for k, v in sorted(os.environ.items()):
    if 'LIKWID' in k or 'likwid' in k:
        print(f"{k}={v}")
print("---")
print(f"LIKWID_FILEPATH={os.environ.get('LIKWID_FILEPATH', 'NOT SET')}")
print(f"LIKWID_MODE={os.environ.get('LIKWID_MODE', 'NOT SET')}")
print(f"LIKWID_EVENTS={os.environ.get('LIKWID_EVENTS', 'NOT SET')}")
print(f"LIKWID_THREADS={os.environ.get('LIKWID_THREADS', 'NOT SET')}")

from fpie import core_openmp
print("\n--- After import, before init ---")
for k, v in sorted(os.environ.items()):
    if 'LIKWID' in k or 'likwid' in k:
        print(f"{k}={v}")

core_openmp.likwid_init()
print("\n--- After likwid_init ---")
for k, v in sorted(os.environ.items()):
    if 'LIKWID' in k or 'likwid' in k:
        print(f"{k}={v}")

core_openmp.likwid_close()
print("Done")
