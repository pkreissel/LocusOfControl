"""Export the trained internal vector for the Bunny Edge feed generator.

Reads LOC_VECTOR (npy) and its sibling _meta.json, writes a single JSON with
vector + calibration to bunny_feed/src/vector.json (LOC_VECTOR_JSON_OUT).

Run: uv run --with numpy python export_vector_json.py
"""

import json
import os

# Data lives in training/data/, resolved against this file rather than the
# caller's cwd: run_all.sh, an editor and a cron job all invoke these scripts
# from different directories, and a bare relative default silently creates a
# second, empty corpus instead of failing. Env overrides still win, and a
# relative override still resolves against cwd -- deliberately, so a one-off
# experiment can point at a file next to it.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
def _d(name):
    return os.path.join(DATA, name)



import numpy as np

VECTOR = os.environ.get("LOC_VECTOR", _d("internal_vector.npy"))
OUT = os.environ.get("LOC_VECTOR_JSON_OUT", "bunny_feed/src/vector.json")


def main():
    v = np.load(VECTOR)
    stem = VECTOR[:-4] if VECTOR.endswith(".npy") else VECTOR
    with open(f"{stem}_meta.json") as f:
        meta = json.load(f)
    out = {
        "vector": [float(x) for x in v],
        "scale": meta["scale"],
        "bias": meta["bias"],
        "embedding_model": meta["embedding_model"],
        "embedding_task": meta["embedding_task"],
    }
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"Exported {VECTOR} ({len(v)} dims, scale={meta['scale']:.3f}, "
          f"bias={meta['bias']:.3f}) -> {OUT}")


if __name__ == "__main__":
    main()
