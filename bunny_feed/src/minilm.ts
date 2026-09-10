// Green Earth hands us an embedding for free. This decodes it.
//
// `/candidates/generate` with `hydrate_embeddings: true` returns
// `minilm_l12_embedding` on each candidate: base64 of 384 little-endian float32s,
// already L2-normalised. Measured against 300 live candidates, that is
// all-MiniLM-L12-v2, mean-pooled, truncated at 128 tokens — cosine 1.0000 (min
// 0.9995) against the local model at those settings, which is what lets a vector
// trained offline in that space be trusted here.
//
// Why bother, when we already pay Jina: scoring a hydrated embedding is a dot
// product, so EVERY candidate can be scored for nothing, and Jina only has to see
// the shortlist that survives. Measured in Node: 4000 candidates decode and score
// in ~20ms (~5µs each). Hydration itself costs no latency (500 candidates: 1.4-2.2s
// hydrated vs 3.0-3.3s not, i.e. inside the noise) and ~2KB of base64 per candidate.
//
// A probe trained on this space is not a weaker stand-in for the Jina one — on the
// locus-of-control axis it matched it offline (val AUC 0.795 vs 0.794) and beat it
// on live posts (62% vs 42% internal by an independent judge). See
// REPORT_ge_embeddings.md.

export const MINILM_MODEL = 'sentence-transformers/all-MiniLM-L12-v2';
export const MINILM_DIMS = 384;

/**
 * Decode one `minilm_l12_embedding` blob, or null if it is absent or malformed.
 *
 * Returns null rather than throwing: a single unparseable candidate must cost that
 * candidate, never the request. Callers treat null as "this one skips the free
 * stage", and the caller-side coverage check decides whether enough of the pool
 * survived to prefilter on at all.
 */
export function decodeEmbedding(b64: string | undefined | null): Float32Array | null {
  if (!b64) return null;
  try {
    // atob, not Buffer: the edge runtime has no Node globals.
    const bin = atob(b64);
    // A float32 is 4 bytes, so a correct blob is exactly 4 * MINILM_DIMS. Anything
    // else is a different model or a truncated payload — either way not something
    // this vector may be scored against.
    if (bin.length !== MINILM_DIMS * 4) return null;
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  } catch {
    return null;
  }
}
