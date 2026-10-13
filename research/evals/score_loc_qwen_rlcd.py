"""Score the LoC axis with harshatheg/Qwen-2.5-1B-RLCD's parallel constrained decoding.

That repository ships no weights of its own. It is an inference engine over
stock `mlx-community/Qwen2.5-1.5B-Instruct-4bit`: prefill the context once, then
evaluate every schema field against the broadcast KV-cache in one batched pass
and softmax over each field's candidate first-tokens. So what this measures is
a 1.5B instruct model given the LoC rubric zero-shot, read out through
constrained decoding -- not a model trained on anything resembling this axis.
The engine's own claim is latency and schema validity, not accuracy, and
nothing here contradicts that; accuracy is simply what this repository needs.

Three fields stand in for ts_questions.py's three primitives, so the output
rows carry the same ts_score / ts_class / ts_class_probs / ts_noul fields the
TypeSafe and Laya scorers write and eval_laya_vs_jev.py reads:

  loc_class  enum internal/external/neutral   -> ts_class, ts_class_probs
  loc_level  enum "0".."4", expected value    -> ts_score   (the 0-4 score axis)
  agency     boolean                          -> ts_noul

Constrained decoding needs the candidates to differ in their FIRST token, or
the engine drops into its multi-token disambiguation path and returns a flat
fabricated distribution instead of a calibrated one. That rules out the v4 band
names ("Strong external" / "Mild external" share a token), which is why the
levels are digits and the legend is carried in the context instead.

QWEN_BACKBONE swaps the backbone for a larger or newer Qwen; the checkpoint is
the only thing the engine contributes an opinion about, and it is not a trained
part of it. See docs/reports/REPORT_laya_vs_jev.md for what the size ladder buys.

Clone the engine first (~250KB, no weights):
  git clone --depth 1 https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD \
      ../data/evals/qwen_rlcd

Run: HF_HOME=/Volumes/Daten/huggingface ../../tmp_venv/bin/python \
     score_loc_qwen_rlcd.py dataset_validation_gemini.jsonl scores_loc_qwen_rlcd_gold.jsonl
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TRAIN_DATA = os.path.join(ROOT, "training", "data")
OUT_DATA = os.path.join(ROOT, "research", "data", "evals")
ENGINE_DIR = os.environ.get("QWEN_RLCD_DIR", os.path.join(OUT_DATA, "qwen_rlcd"))

sys.path.insert(0, os.path.join(ROOT, "training"))
import ts_questions  # noqa: E402

# The engine hardcodes its checkpoint as a module constant rather than reading
# an env var, so swapping the backbone means patching `core.engine_mlx.MODEL_ID`
# before the first get_engine(). Anything that speaks ChatML and builds a plain
# KVCache works -- see the assertions in main(), which are the actual
# compatibility test.
BACKBONE = os.environ.get("QWEN_BACKBONE", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")
# Force the tiled prefill even on a backbone the engine could broadcast (it is
# selected automatically on one it cannot), and QWEN_TILED_CHECK=1 to assert the
# two paths agree before trusting the tiled one on a backbone that has no
# broadcast to compare against.
TILED = os.environ.get("QWEN_TILED") == "1"
TILED_CHECK = os.environ.get("QWEN_TILED_CHECK") == "1"
MODEL = f"harshatheg/Qwen-2.5-1B-RLCD ({BACKBONE})"

# The rubric goes in the context, not in the field descriptions: the engine's
# prefill catalog keeps only `description.split('\n')[0]` per field, so a
# multi-line rubric hung off a field is silently dropped.
LEGEND = "\n".join(f"{i} = {lvl}" for i, lvl in enumerate(ts_questions.LOC_LEVELS))
CRITERIA = "\n".join(f"{k} = {v}" for k, v in ts_questions.CLASS_CRITERIA.items())

SCHEMA = {
    "loc_class": {
        "type": "enum",
        "choices": ["internal", "external", "neutral"],
        "description": "Locus-of-control class of the post: internal, external, or neutral",
    },
    "loc_level": {
        "type": "enum",
        "choices": [str(i) for i in range(5)],
        "description": "Locus-of-control level, 0 = strong external ... 4 = internal and concrete",
    },
    "agency": {
        "type": "boolean",
        "description": "The post leaves a reader feeling more able to act on their own life",
    },
}


def run_tiled(model, tokenizer, context, schema, mx):
    """The engine's decision, computed without broadcasting the KV cache.

    run_parallel_generation() prefills at batch 1 and then widens the cache to
    batch M with mx.repeat(layer.keys, M). That is a latency optimization and it
    is also the only thing tying the engine to plain-attention backbones: a
    hybrid model (Qwen3.5 / Qwen3.8 / Qwen3-Next) holds a recurrent ArraysCache
    on its linear-attention layers, which has no `.keys` to widen -- and the
    engine skips those layers silently rather than failing, so the batched pass
    would score every field but the first against a prefix it never saw.

    Tiling the prompt to batch M and prefilling at batch M instead builds the
    cache at the right width to begin with, whatever its type. It costs M times
    the prefill FLOPs (M = 3 here), which is exactly the saving the broadcast
    exists to make -- so this is not a replacement for the engine, it is how you
    measure a backbone the engine cannot carry. On a plain-attention backbone it
    must agree with the engine to float noise; `--check` asserts that it does.
    """
    meta = schema.compile_parallel_metadata(tokenizer)
    field_items = meta["field_items"]
    M = len(field_items)

    base_prompt = (f"<|im_start|>system\nClassify JSON attributes:\n"
                   f"{schema.to_parallel_schema_str()}<|im_end|>\n"
                   f"<|im_start|>user\n{context}<|im_end|>\n"
                   f"<|im_start|>assistant\n{{\n")
    base = mx.array(tokenizer.encode(base_prompt))[None]

    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    model(mx.repeat(base, M, axis=0), cache=cache)
    logits = model(meta["suffixes_batch"], cache=cache)
    mx.eval(logits)

    out = {}
    for i, (fname, fdef) in enumerate(field_items):
        row = logits[i, meta["suffix_lengths"][i] - 1, :]
        p = mx.softmax(mx.array([float(row[t]) for t in meta["cands_per_field"][i]]))
        mx.eval(p)
        choices = ["true", "false"] if fdef.field_type == "boolean" else fdef.choices
        probs = p.tolist()
        w = int(mx.argmax(p))
        out[fname] = {"value": (w == 0) if fdef.field_type == "boolean" else fdef.choices[w],
                      "confidence": round(float(probs[w]), 4),
                      "top_choices": [{"choice": c, "probability": round(v, 4)}
                                      for c, v in zip(choices, probs)]}
    return {"field_telemetry": out, "elapsed_ms": 0.0, "prefill_ms": 0.0}


def build_context(text):
    return (
        f"{ts_questions.LOC_INSTRUCTIONS}\n\n"
        f"Levels:\n{LEGEND}\n\n"
        f"Classes:\n{CRITERIA}\n\n"
        f"Post:\n{text}"
    )


def resolve(path, default_dir):
    return path if os.path.isabs(path) or os.path.exists(path) else os.path.join(default_dir, path)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src = resolve(sys.argv[1], TRAIN_DATA)
    dst = resolve(sys.argv[2], OUT_DATA)

    rows = [r for r in load_jsonl(src) if r.get("text")]
    done = {r["uri"] for r in load_jsonl(dst)} if os.path.exists(dst) else set()
    todo = [r for r in rows if r["uri"] not in done]
    print(f"{src}: {len(rows)} posts, {len(done)} already scored, {len(todo)} to do")
    if not todo:
        return

    if not os.path.isdir(os.path.join(ENGINE_DIR, "core")):
        sys.exit(f"engine checkout not found at {ENGINE_DIR} -- see the docstring")
    sys.path.insert(0, ENGINE_DIR)
    from core.schema import StructuredSchema
    from core.engine import run_parallel_generation, USE_MLX
    if USE_MLX:
        import core.engine_mlx as EM
        EM.MODEL_ID = BACKBONE
    else:
        os.environ["MODEL_ID"] = BACKBONE

    # Unified memory is the GPU's memory here, and the engine allocates a fresh
    # prompt cache per post and then broadcasts it across the schema's fields.
    # MLX holds freed buffers in its own cache rather than returning them, so on
    # a 32GB machine with anything else running a 500-post loop walks into the
    # OOM killer (SIGKILL, exit 137) somewhere past post 200. Cap the buffer
    # cache and drop it periodically; the reclaim costs a few ms a post.
    mem = None
    if USE_MLX:
        import mlx.core as mx
        mem = mx
        mx.set_cache_limit(int(os.environ.get("QWEN_MLX_CACHE_MB", "1024")) << 20)
        info = mx.device_info()
        print(f"{info['device_name']}: {info['memory_size'] / 2**30:.0f}GB unified, "
              f"{info['max_recommended_working_set_size'] / 2**30:.0f}GB recommended working set")

    schema = StructuredSchema(SCHEMA)
    run_parallel_generation(build_context(todo[0]["text"]), schema)  # load + warm up

    score = run_parallel_generation
    if USE_MLX:
        # The broadcast trick is `mx.repeat(layer.keys, M, axis=0)` over every
        # layer of a plain KVCache. A backbone whose make_cache() returns
        # anything else -- a rotating cache for sliding-window attention, a
        # recurrent state for a linear-attention or Mamba layer -- has layers
        # with no `.keys` to repeat, and the engine skips them silently: the
        # batched pass then attends to a prefix that only row 0 ever saw, and
        # every field but the first is scored against the wrong context. Take
        # the tiled path for those rather than trusting the broadcast.
        from mlx_lm.models.cache import KVCache, make_prompt_cache
        kinds = {type(c).__name__ for c in make_prompt_cache(EM._model)}
        broadcastable = kinds == {KVCache.__name__}
        tiled = TILED or not broadcastable
        if tiled:
            print(f"  {BACKBONE} builds {sorted(kinds)}"
                  + ("" if broadcastable else " -- not broadcastable") + "; using the tiled prefill")
            score = lambda ctx, sch: run_tiled(EM._model, EM._tokenizer, ctx, sch, mx)
        if TILED_CHECK and broadcastable:
            worst = 0.0
            for r in todo[:20]:
                ctx = build_context(r["text"])
                a, b = run_parallel_generation(ctx, schema)["field_telemetry"], \
                    run_tiled(EM._model, EM._tokenizer, ctx, schema, mx)["field_telemetry"]
                for fn in a:
                    pa = {c["choice"]: c["probability"] for c in a[fn]["top_choices"]}
                    pb = {c["choice"]: c["probability"] for c in b[fn]["top_choices"]}
                    worst = max(worst, max(abs(pa[k] - pb[k]) for k in pa))
            print(f"  tiled vs broadcast over 20 posts x {len(SCHEMA)} fields: "
                  f"max |dp| = {worst:.4f}")
    assert not any(schema._parallel_metadata["has_collisions"]), (
        "a field's choices share a first token; the engine would fall back to its "
        "multi-token path and return a fabricated flat distribution")
    print(f"engine ready (mlx={USE_MLX}) backbone={BACKBONE}")

    t_run = time.time()
    with open(dst, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            t = time.time()
            r = score(build_context(row["text"]), schema)
            dt = time.time() - t
            tel = r["field_telemetry"]
            cls = {c["choice"]: c["probability"] for c in tel["loc_class"]["top_choices"]}
            lvl = {c["choice"]: c["probability"] for c in tel["loc_level"]["top_choices"]}
            agency = {c["choice"]: c["probability"] for c in tel["agency"]["top_choices"]}
            f.write(json.dumps({
                "uri": row["uri"],
                "latency_s": round(dt, 3),
                # Expected value over the five levels, the same read-out
                # ts_questions' `score` primitive uses, on the same 0-4 axis.
                "ts_score": round(sum(int(k) * v for k, v in lvl.items()), 4),
                "ts_score_probs": lvl,
                "ts_score_conf": tel["loc_level"]["confidence"],
                "ts_class": tel["loc_class"]["value"],
                "ts_class_probs": cls,
                "ts_class_conf": tel["loc_class"]["confidence"],
                "ts_noul": agency.get("true", 0.0),
                "elapsed_ms": r["elapsed_ms"],
                "prefill_ms": r["prefill_ms"],
                "model": MODEL,
            }, ensure_ascii=False) + "\n")
            f.flush()
            if mem is not None and i % 25 == 0:
                mem.clear_cache()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t_run
                gb = (f"  mem {mem.get_active_memory() / 2**30:.1f}+{mem.get_cache_memory() / 2**30:.1f}GB "
                      f"peak {mem.get_peak_memory() / 2**30:.1f}GB" if mem is not None else "")
                print(f"  {i}/{len(todo)}  {el / i:.2f}s/post  "
                      f"eta {(len(todo) - i) * el / i / 60:.1f}min{gb}", flush=True)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
