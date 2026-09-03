import { createClient } from "@libsql/client";
import * as dotenv from "dotenv";

dotenv.config({ path: "../.env" });

async function main() {
  const url = process.env.BUNNY_DATABASE_URL || process.env.BUNNY_DB_URL;
  const authToken = process.env.BUNNY_DATABASE_AUTH_TOKEN || process.env.BUNNY_DB_TOKEN;

  if (!url || !authToken) {
    console.error("Missing database credentials in ../.env");
    process.exit(1);
  }

  const client = createClient({ url, authToken });
  try {
    const result = await client.execute("SELECT SUM(tokens) as total_tokens, SUM(requests) as total_requests FROM embedding_usage");
    console.log(JSON.stringify(result.rows));
  } catch (err) {
    console.error(err);
  }
}

main();
