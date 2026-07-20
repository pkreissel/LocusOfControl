import { createClient } from "@libsql/client/web";

declare const process: any;

function getDbClient() {
  const url = typeof process !== 'undefined' && process.env ? process.env.BUNNY_DB_URL : undefined;
  const authToken = typeof process !== 'undefined' && process.env ? process.env.BUNNY_DB_TOKEN : undefined;

  if (!url || !authToken) {
    throw new Error("Missing BUNNY_DB_URL or BUNNY_DB_TOKEN in environment");
  }

  return createClient({ url, authToken });
}

export async function saveUserSession(did: string, handle: string, accessJwt: string, refreshJwt: string) {
  const db = getDbClient();
  // Ensure table exists (in production, this would be a separate migration step)
  await db.execute(`
    CREATE TABLE IF NOT EXISTS users (
      did TEXT PRIMARY KEY,
      handle TEXT,
      access_token TEXT,
      refresh_token TEXT,
      updated_at TIMESTAMP
    )
  `);

  await db.execute({
    sql: `INSERT INTO users (did, handle, access_token, refresh_token, updated_at) 
          VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
          ON CONFLICT(did) DO UPDATE SET 
          handle=excluded.handle, 
          access_token=excluded.access_token, 
          refresh_token=excluded.refresh_token,
          updated_at=CURRENT_TIMESTAMP`,
    args: [did, handle, accessJwt, refreshJwt]
  });
}

export async function getUserSession(did: string): Promise<any | null> {
  const db = getDbClient();
  const result = await db.execute({
    sql: `SELECT access_token, refresh_token, handle FROM users WHERE did = ?`,
    args: [did]
  });
  if (result.rows.length === 0) return null;
  return {
    accessJwt: result.rows[0].access_token as string,
    refreshJwt: result.rows[0].refresh_token as string,
    handle: result.rows[0].handle as string
  };
}
