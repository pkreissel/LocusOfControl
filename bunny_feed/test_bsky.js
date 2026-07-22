const { Agent } = require('@atproto/api');

async function main() {
  const agent = new Agent('https://public.api.bsky.app');
  try {
    const res = await agent.app.bsky.feed.getFeed({
      feed: 'at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot',
      limit: 10
    });
    console.log("Success:", res.data.feed.length);
  } catch (err) {
    console.error("Error:", err.message);
  }
}
main();
