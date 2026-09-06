// Identifies which model produced a vector, so cached embeddings can be scoped
// to it and automatically invalidated (via cache miss) when the model changes.
export const EMBEDDING_MODEL = 'jina-embeddings-v5-text-small';

export async function getEmbeddings(texts: string[], apiKey: string): Promise<{ embeddings: number[][]; tokens: number }> {
  const response = await fetch('https://api.jina.ai/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      task: 'classification',
      input: texts.map(t => t.slice(0, 2000))
    })
  });
  if (!response.ok) {
    const text = await response.text();
    const msg = `Jina API error: ${response.status} ${text}`;
    console.error(`[Jina API Error]: ${msg}`);
    throw new Error(msg);
  }
  const data = await response.json();
  return {
    embeddings: data.data.map((d: any) => d.embedding),
    tokens: data.usage?.total_tokens ?? 0,
  };
}

// --- Bounded-parallel embedding -------------------------------------------
//
// A single request for all cache misses is the slowest way to do this: measured
// against 471 live candidates, one request took ~6.2s while splitting the same
// work across 8 concurrent requests took ~3.9s.
//
// The gains flatten fast, though, because Jina throttles on token throughput per
// key rather than per request — fanning out removes serialization overhead, it
// does not create throughput. 12-way (~3.7s) and 16-way (~3.4s) barely improve
// on 8-way, and back-to-back bursts get actively penalized: 30 concurrent
// requests measured *slower* than one (~12.8s), because each successive request
// lands further into the rate limiter's queue. So we cap at 8 and keep a floor
// on batch size, rather than splitting as finely as possible.
const JINA_MAX_PARALLEL = 8;
const JINA_MIN_BATCH = 32;

// Embed `texts`, splitting across up to JINA_MAX_PARALLEL concurrent requests.
// Returns embeddings positionally aligned to `texts`, with `null` in any slot
// whose chunk failed — one bad chunk costs those candidates, not the whole feed.
// Throws only if every chunk failed, preserving "total Jina outage = loud error".
export async function getEmbeddingsBatched(
  texts: string[],
  apiKey: string,
): Promise<{ embeddings: (number[] | null)[]; tokens: number; chunks: number; failedChunks: number }> {
  const embeddings: (number[] | null)[] = new Array(texts.length).fill(null);
  if (texts.length === 0) return { embeddings, tokens: 0, chunks: 0, failedChunks: 0 };

  const target = Math.max(1, Math.min(JINA_MAX_PARALLEL, Math.ceil(texts.length / JINA_MIN_BATCH)));
  const size = Math.ceil(texts.length / target);
  const ranges: [number, number][] = [];
  for (let i = 0; i < texts.length; i += size) ranges.push([i, Math.min(i + size, texts.length)]);

  const settled = await Promise.allSettled(ranges.map(([s, e]) => getEmbeddings(texts.slice(s, e), apiKey)));

  let tokens = 0;
  let failedChunks = 0;
  let firstErr: any = null;
  settled.forEach((r, ci) => {
    if (r.status === 'fulfilled') {
      tokens += r.value.tokens;
      const [s] = ranges[ci];
      for (let k = 0; k < r.value.embeddings.length; k++) embeddings[s + k] = r.value.embeddings[k];
    } else {
      failedChunks++;
      if (!firstErr) firstErr = r.reason;
      console.error(`[Jina] chunk ${ci + 1}/${ranges.length} failed:`, r.reason?.message ?? String(r.reason));
    }
  });

  if (failedChunks === ranges.length) throw firstErr;
  return { embeddings, tokens, chunks: ranges.length, failedChunks };
}
