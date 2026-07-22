import type { SessionStore, Session } from "@atproto/oauth-client";
import type { InternalStateData, StateStore } from "@atproto/oauth-client";
import type { RuntimeImplementation } from "@atproto/oauth-client";
import { JoseKey } from "@atproto/jwk-jose";
import { getDbClient } from "./db";
import { eq, sql, and } from "drizzle-orm";
import { oauthStates, oauthSessions, oauthLocks } from "./schema";

export class WebCryptoRuntimeImplementation implements RuntimeImplementation {
  async requestLock<T>(name: string, fn: () => T | Promise<T>): Promise<T> {
    const db = getDbClient();
    // TTL must comfortably exceed the critical section (an OAuth token refresh:
    // one PDS round-trip + a session write). MAX_WAIT must exceed the TTL so a
    // stale lock left by a recycled isolate is always reclaimed within the
    // retry window rather than failing the request.
    const LOCK_TTL_MS = 15000;
    const MAX_WAIT_MS = 20000;
    const deadline = Date.now() + MAX_WAIT_MS;

    let acquired = false;
    let myExpiry = 0;

    while (true) {
      const now = Date.now();
      const expiresAt = now + LOCK_TTL_MS;
      try {
        const result = await db.insert(oauthLocks)
          .values({ name, expiresAt })
          .onConflictDoUpdate({
            target: oauthLocks.name,
            set: { expiresAt },
            where: sql`${oauthLocks.expiresAt} < ${now}`
          })
          .returning();
        if (result.length > 0) {
          acquired = true;
          myExpiry = expiresAt;
          break;
        }
      } catch (e) {
        console.error(`[OAuth Lock] error acquiring lock ${name}:`, e);
      }
      if (Date.now() >= deadline) break;
      // Jittered backoff to avoid a thundering herd when the lock frees.
      await new Promise(r => setTimeout(r, 100 + Math.floor(Math.random() * 100)));
    }

    if (!acquired) {
      throw new Error(`Failed to acquire OAuth lock for ${name} (likely held by another request)`);
    }

    try {
      return await fn();
    } finally {
      try {
        // Scope the release to the lock we actually hold: if our critical
        // section outran the TTL and another request re-acquired, expires_at
        // no longer matches ours and this delete is a harmless no-op.
        await db.delete(oauthLocks)
          .where(and(eq(oauthLocks.name, name), eq(oauthLocks.expiresAt, myExpiry)));
      } catch (e) {
        console.error(`[OAuth Lock] error releasing lock ${name}:`, e);
      }
    }
  }

  // Must be a JoseKey (extractable, serialisable to a private JWK). A
  // WebcryptoKey wraps a non-extractable CryptoKeyPair which cannot survive a
  // JSON/SQL round-trip through the state/session stores below.
  async createKey(algs: any) { return JoseKey.generate(algs); }
  getRandomValues(byteLength: number) { return crypto.getRandomValues(new Uint8Array(byteLength)); }
  async digest(data: Uint8Array, { name }: { name: string }) {
    const alg = name === 'sha256' ? 'SHA-256' : name === 'sha384' ? 'SHA-384' : name === 'sha512' ? 'SHA-512' : null;
    if (!alg) throw new Error(`Unsupported digest alg: ${name}`);
    return new Uint8Array(await crypto.subtle.digest(alg, data as any));
  }
}

// The OAuth client stores a `dpopKey` (a Key class instance) inside both the
// state and session records. Plain JSON.stringify drops the class prototype
// (so getters like `algorithms` and methods like `createJwt` are lost) and the
// underlying key material. We serialise the key as its private JWK on write and
// reconstruct a JoseKey on read, mirroring @atproto/oauth-client-node's
// `toDpopKeyStore` helper.
function serializeWithKey(value: { dpopKey?: any }): string {
  const { dpopKey, ...rest } = value as any;
  const dpopJwk = dpopKey?.privateJwk;
  if (dpopKey && !dpopJwk) throw new Error('Private DPoP JWK is missing (non-extractable key?)');
  return JSON.stringify({ ...rest, dpopJwk });
}

async function deserializeWithKey<T>(data: string): Promise<T> {
  const { dpopJwk, ...rest } = JSON.parse(data);
  if (!dpopJwk) return rest as T;
  const dpopKey = await JoseKey.fromJWK(dpopJwk);
  return { ...rest, dpopKey } as T;
}

let _initPromise: Promise<void> | null = null;
export function initOAuthTables(): Promise<void> {
  if (!_initPromise) {
    _initPromise = Promise.resolve();
  }
  return _initPromise;
}

export class LibSqlStateStore implements StateStore {
  async get(key: string): Promise<InternalStateData | undefined> {
    const db = getDbClient();
    const result = await db.select({ data: oauthStates.data }).from(oauthStates).where(eq(oauthStates.key, key)).limit(1);
    if (result.length === 0) return undefined;
    return deserializeWithKey<InternalStateData>(result[0].data);
  }

  async set(key: string, value: InternalStateData): Promise<void> {
    const db = getDbClient();
    const data = serializeWithKey(value);
    await db.insert(oauthStates)
      .values({ key, data })
      .onConflictDoUpdate({
        target: oauthStates.key,
        set: { data, updatedAt: sql`CURRENT_TIMESTAMP` }
      });
  }

  async del(key: string): Promise<void> {
    const db = getDbClient();
    await db.delete(oauthStates).where(eq(oauthStates.key, key));
  }
}

// In-isolate read-through cache for OAuth session reads.
//
// atproto's OAuthSession calls getSession() once per upstream HTTP request:
// once in restore(), then again inside every getFeed page's fetchHandler.
// Within a single feed request those calls run in the SAME isolate microseconds
// apart, so this cache turns all but the first into guaranteed hits and removes
// the redundant Turso SELECTs (and re-imports of the DPoP JoseKey). The TTL is
// short so a token rotated by another isolate is picked up quickly; forced
// refreshes go through the getter and write back here via set(), so the cache
// never serves a token the library then treats as stale.
const SESSION_CACHE_TTL_MS = 15_000;
const SESSION_CACHE_MAX = 2_000;
const sessionMemCache = new Map<string, { value: Session; exp: number }>();

// The library refreshes a token when it is within ~40s of expiry (10s fixed +
// up to 30s jitter; see SessionGetter.isStale). If we serve a cached token
// inside that window, the getter — running under our distributed lock — may try
// to refresh a token another isolate has ALREADY rotated (its single-use
// refresh_token is spent), get invalid_grant, and (because we provide a lock,
// the library's no-lock recovery path is skipped) DELETE the still-valid
// session. So near expiry we must bypass the cache and re-read the current row.
// Margin is set comfortably above isStale's ~40s.
const REFRESH_WINDOW_MS = 60_000;

function tokenExpiresWithin(value: Session, ms: number): boolean {
  const exp = (value as any)?.tokenSet?.expires_at;
  if (exp == null) return false;
  return new Date(exp).getTime() < Date.now() + ms; // mirrors isStale()
}

function cacheSession(key: string, value: Session) {
  sessionMemCache.set(key, { value, exp: Date.now() + SESSION_CACHE_TTL_MS });
  if (sessionMemCache.size > SESSION_CACHE_MAX) {
    // Map preserves insertion order; drop the oldest entry.
    const oldest = sessionMemCache.keys().next().value;
    if (oldest !== undefined) sessionMemCache.delete(oldest);
  }
}

export class LibSqlSessionStore implements SessionStore {
  async get(key: string): Promise<Session | undefined> {
    const cached = sessionMemCache.get(key);
    // Serve from cache ONLY when the token is comfortably far from the refresh
    // window. Near expiry, always re-read Turso so we pick up a token another
    // isolate may have already rotated — otherwise we'd refresh a spent token
    // under the lock and the library would delete the still-valid session.
    if (cached && cached.exp > Date.now() && !tokenExpiresWithin(cached.value, REFRESH_WINDOW_MS)) {
      return cached.value;
    }

    const db = getDbClient();
    const result = await db.select({ data: oauthSessions.data }).from(oauthSessions).where(eq(oauthSessions.key, key)).limit(1);
    if (result.length === 0) {
      sessionMemCache.delete(key);
      return undefined;
    }
    const value = await deserializeWithKey<Session>(result[0].data);
    cacheSession(key, value);
    return value;
  }

  async set(key: string, value: Session): Promise<void> {
    const db = getDbClient();
    const data = serializeWithKey(value);
    // If this throws, the OAuth client treats the just-refreshed token as
    // unstorable: it REVOKES the new token and DELETES the session (see
    // SessionGetter.onStoreError). That turns a transient Turso write blip into
    // a forced re-login, so retry a few times before giving up.
    let lastErr: unknown;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        await db.insert(oauthSessions)
          .values({ key, data })
          .onConflictDoUpdate({
            target: oauthSessions.key,
            set: { data, updatedAt: sql`CURRENT_TIMESTAMP` }
          });
        // Write-through so the freshest token (e.g. just refreshed) is served next.
        cacheSession(key, value);
        return;
      } catch (e) {
        lastErr = e;
        await new Promise(r => setTimeout(r, 100 * (attempt + 1)));
      }
    }
    throw lastErr;
  }

  async del(key: string): Promise<void> {
    sessionMemCache.delete(key);
    const db = getDbClient();
    await db.delete(oauthSessions).where(eq(oauthSessions.key, key));
  }
}
