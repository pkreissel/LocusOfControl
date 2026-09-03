import { createClient } from "@libsql/client";
import { drizzle } from "drizzle-orm/libsql";
import * as schema from "./src/schema";
import * as dotenv from "dotenv";
import { resolve } from "path";

dotenv.config({ path: resolve(__dirname, ".env") });

const SOURCE_URL = process.env.BUNNY_DATABASE_URL!;
const SOURCE_TOKEN = process.env.BUNNY_DATABASE_AUTH_TOKEN!;

const TARGET_URL = "libsql://01M0A02SPJY2J7P1V5E08K2C03-feedforward.lite.bunnydb.net/";
const TARGET_TOKEN = process.env.BUNNY_NEW_TOKEN!;

if (!TARGET_TOKEN) {
  console.error("Missing BUNNY_NEW_TOKEN in .env");
  process.exit(1);
}

const sourceClient = createClient({ url: SOURCE_URL, authToken: SOURCE_TOKEN });
const sourceDb = drizzle(sourceClient, { schema });

const targetClient = createClient({ url: TARGET_URL, authToken: TARGET_TOKEN });
const targetDb = drizzle(targetClient, { schema });

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

async function migrateTable(tableName: string, tableSchema: any) {
  console.log(`Migrating ${tableName}...`);
  const batchSize = 1000;
  let offset = 0;
  let totalMigrated = 0;

  while (true) {
    const rows = await sourceDb.select().from(tableSchema).limit(batchSize).offset(offset);
    if (rows.length === 0) break;
    
    let retries = 5;
    while (retries > 0) {
      try {
        await targetDb.insert(tableSchema).values(rows).onConflictDoNothing();
        break;
      } catch (e: any) {
        console.error(`Error inserting batch, retries left ${retries - 1}:`, e.message);
        retries--;
        if (retries === 0) throw e;
        await sleep(5000); // Wait longer on error
      }
    }
    
    totalMigrated += rows.length;
    console.log(`Inserted batch offset ${offset} to ${offset + rows.length} into ${tableName}`);
    offset += batchSize;
    await sleep(200); // Sleep briefly between batches
  }
  
  console.log(`Finished migrating ${totalMigrated} rows for ${tableName}\n`);
}

async function run() {
  console.log(`Starting migration from ${SOURCE_URL} to ${TARGET_URL}`);
  
  await migrateTable("embeddingUsage", schema.embeddingUsage);
  await migrateTable("postEmbeddings", schema.postEmbeddings);
  await migrateTable("feedSessions", schema.feedSessions);
  await migrateTable("feedInteractions", schema.feedInteractions);
  
  console.log("Migration complete.");
  process.exit(0);
}

run().catch((e) => {
  console.error("Migration failed:", e);
  process.exit(1);
});
