import { BskyAgent } from '@atproto/api';

const CANDIDATE_FEED_URI = 'at://did:plc:wrmpulygwvuhjn2c3jbalgqj/app.bsky.feed.generator/your-feed';

export async function fetchCandidatePosts(agent: BskyAgent, limit: number = 50) {
  const response = await agent.app.bsky.feed.getFeed({
    feed: CANDIDATE_FEED_URI,
    limit
  });
  
  if (!response.data.feed) return [];

  const candidates = response.data.feed.map(item => {
    return {
      uri: item.post.uri,
      cid: item.post.cid,
      // @ts-ignore
      text: item.post.record.text || '',
    };
  });
  
  return candidates;
}
