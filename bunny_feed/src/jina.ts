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
