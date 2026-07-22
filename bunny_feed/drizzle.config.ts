import { defineConfig } from 'drizzle-kit';
import * as dotenv from 'dotenv';
dotenv.config();

export default defineConfig({
  schema: './src/schema.ts',
  out: './drizzle',
  dialect: 'turso',
  dbCredentials: {
    url: process.env.BUNNY_DATABASE_URL || process.env.BUNNY_DB_URL || '',
    authToken: process.env.BUNNY_DATABASE_AUTH_TOKEN || process.env.BUNNY_DB_TOKEN || '',
  },
});
