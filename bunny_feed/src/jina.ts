export async function getEmbeddings(texts: string[], apiKey: string): Promise<number[][]> {
  const response = await fetch('https://api.jina.ai/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: 'jina-embeddings-v5-text-small',
      task: 'classification',
      input: texts.map(t => t.slice(0, 2000))
    })
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Jina API error: ${response.status} ${text}`);
  }
  const data = await response.json();
  return data.data.map((d: any) => d.embedding);
}
