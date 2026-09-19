// Is this candidate German? A function-word test over the text GE already sends.
//
// Why the feed needs to know: the free prefilter runs on GE's hydrated MiniLM
// embeddings, and `all-MiniLM-L12-v2` is an English-only model. Measured
// out-of-fold over the 2822-row labelled corpus (REPORT_minilm_vs_jina_by_lang.md):
// the MiniLM probe beats Jina by +0.089 AUC on English and *trails* it by 0.085 on
// German (0.588 vs 0.674), an interaction of +0.174 with CI [+0.057, +0.292]. At
// the shipped 20%-equivalent cut that costs 70% of German internal posts, thrown
// away before Jina — the multilingual space that can actually read them — ever
// sees the post. So German candidates have to go round the free stage.
//
// Why a stopword ratio and not something cleverer:
//
//   - GE does not tell us. `/candidates/generate` returns at_uri, content,
//     minilm_l12_embedding, score, generator_name, author_did, author_username,
//     contains_images/video, image_count, video_count, external_uri, like_count —
//     and no language field (checked live 2026-09-11). The post's own `langs` lives
//     on the AT record, which this worker never fetches.
//   - The embedding itself would do it — language is linearly decodable in MiniLM
//     at AUC 0.966 — but at 95% German recall it also drags 14-24% of the English
//     pool onto the paid path, which is worse than this on both recall and cost.
//   - A real detector (franc, CLD3) is a dependency and a bundle for a decision
//     this makes at 0.955 accuracy on the corpus it was measured against.
//
// Against the 1759 corpus rows carrying a declared `de` or `en` tag: accuracy
// 0.955, German recall 0.858, precision 0.908, and 2.1% of English rows falsely
// flagged. Both error directions are cheap here — a missed German post gets the
// prefilter it gets today, and a falsely flagged English one merely skips a free
// stage and is judged by Jina instead. See test_lang.ts, which re-measures those
// numbers against the corpus so a change to the word lists has to face them.
//
// Deliberately German-only, not "non-English": German is the language the deficit
// was measured in, and the corpus has too few rows in anything else to claim the
// same for them. Other languages keep today's behaviour until someone measures
// them.

// Function words, not vocabulary: they are the words a short post cannot avoid,
// they do not date, and they overlap little between the two languages. The few
// that do ("in", "an", "was", "so") are dropped from BOTH lists below, so a word
// that is evidence for neither language votes for neither. Cancelling them in the
// de-vs-en comparison alone was not enough: it left them inflating the German
// ratio in the umlaut clause, which flagged "Reading Kafka in Köln this week" as
// German on the strength of one shared "in" and one name.
const DE_WORDS_RAW = [
  'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'einem', 'einen',
  'und', 'oder', 'aber', 'nicht', 'ist', 'sind', 'war', 'waren', 'wird', 'werden',
  'wurde', 'hat', 'haben', 'hatte', 'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr',
  'mich', 'mir', 'dich', 'dir', 'sich', 'uns', 'euch', 'auf', 'für', 'mit', 'von',
  'zu', 'im', 'in', 'am', 'an', 'als', 'auch', 'noch', 'nur', 'schon', 'wenn',
  'weil', 'dass', 'dann', 'doch', 'man', 'sehr', 'mehr', 'immer', 'kann', 'muss',
  'soll', 'wie', 'was', 'wer', 'wo', 'warum', 'ja', 'nein', 'nichts', 'alles',
  'etwas', 'beim', 'zum', 'zur', 'vom', 'über', 'unter', 'gegen', 'ohne', 'durch',
];

const EN_WORDS_RAW = [
  'the', 'a', 'an', 'and', 'or', 'but', 'not', 'is', 'are', 'was', 'were', 'will',
  'would', 'has', 'have', 'had', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
  'him', 'her', 'them', 'of', 'to', 'in', 'on', 'at', 'for', 'with', 'from', 'by',
  'about', 'as', 'that', 'this', 'these', 'those', 'there', 'here', 'what', 'who',
  'where', 'why', 'how', 'can', 'could', 'should', 'must', 'do', 'does', 'did',
  'just', 'only', 'very', 'more', 'most', 'some', 'any', 'if', 'then', 'than', 'so',
];

// Enforced here rather than by hand-editing the two lists, so adding a word to one
// of them cannot silently reintroduce a word that is evidence for neither.
const SHARED = new Set(DE_WORDS_RAW.filter(w => EN_WORDS_RAW.includes(w)));
const DE_WORDS = new Set(DE_WORDS_RAW.filter(w => !SHARED.has(w)));
const EN_WORDS = new Set(EN_WORDS_RAW.filter(w => !SHARED.has(w)));

// Letters only, and not the ones inside a link. Digits, punctuation and emoji carry
// no evidence either way, and leaving them out of the denominator keeps the ratios
// comparable between a bare sentence and the same sentence trailing three hashtags.
// A URL is stripped whole — a slug is not prose, and `/uns/` or `/ueber/` would
// otherwise be the only "words" a link-only post has. Same regex the length gate
// uses on the same text (URL_RE in index.ts), for the same reason.
const URL_RE = /(?:https?:\/\/|www\.)\S+/g;
const WORD_RE = /[A-Za-zÄÖÜäöüß']+/g;
const UMLAUT_RE = /[ÄÖÜäöüß]/g;

// A post that is mostly German function words is German. The umlaut clause is the
// second chance for the case this misses most often — a short post with one or two
// function words and a distinctly German vocabulary ("Bundesländer", "größer") —
// and is deliberately weak on its own: umlauts alone would flag English posts
// quoting a name, so it only fires when the German words are also present.
const MIN_DE_RATIO_WITH_UMLAUTS = 0.05;
const MIN_UMLAUT_DENSITY = 0.002;

/** True if this text reads as German. Empty or word-less text is not German. */
export function looksGerman(text: string | null | undefined): boolean {
  const prose = (text || '').replace(URL_RE, ' ');
  const words = prose.match(WORD_RE);
  if (!words || words.length === 0) return false;
  let de = 0, en = 0;
  for (const w of words) {
    const lower = w.toLowerCase();
    if (DE_WORDS.has(lower)) de++;
    if (EN_WORDS.has(lower)) en++;
  }
  const deRatio = de / words.length;
  const enRatio = en / words.length;
  if (deRatio > enRatio) return true;
  const umlauts = (prose.match(UMLAUT_RE) || []).length;
  const density = umlauts / Math.max(prose.length, 1);
  return deRatio >= MIN_DE_RATIO_WITH_UMLAUTS && density > MIN_UMLAUT_DENSITY;
}

// The reader's content languages, as their client sends them.
//
// The Bluesky app puts its Settings -> Languages -> Content Languages list in
// `Accept-Language` on getFeed (social-app src/lib/api/feed/custom.ts), and the
// appview copies that header onto the getFeedSkeleton call it makes to us
// (atproto packages/bsky/src/api/app/bsky/feed/getFeed.ts). So this is the
// reader's own filter, arriving one hop late but intact.
//
// It is worth reading because the client ENFORCES that list on every custom feed
// (FeedTuner.preferredLangOnly in social-app src/lib/api/feed-manip.ts): a post
// outside it is dropped from the page before the reader ever sees it. A feed that
// ignores the header ranks posts that are discarded on arrival — measured here on
// 2026-09-19, a German-only reader was served 150 posts and shown 19.
//
// Parses both shapes seen in the wild: the app's bare comma list ("de,en") and
// the weighted browser form ("de-DE,de;q=0.9,en;q=0.8"). Region subtags are
// dropped because the client matches on the 2-letter code and so must we.
export function readerLangs(header: string | null | undefined): string[] {
  const raw = header;
  if (!raw) return [];
  const out: string[] = [];
  for (const part of raw.split(',')) {
    const code = part.split(';')[0].trim().toLowerCase().split('-')[0];
    if (code && code !== '*' && !out.includes(code)) out.push(code);
  }
  return out;
}

// Is this a reader whose client will hide the English majority of this feed?
//
// Deliberately narrow, and asymmetric with the language list above. German is the
// only language this worker can detect (lang.ts, and see the note there on why it
// is German-only), so German is the only preference it can act on: a French-only
// reader is no better served by German posts than by English ones, and gets
// today's behaviour rather than a guess. A reader who lists English alongside
// German sees the whole feed already and is left alone.
export function prefersGermanOnly(langs: string[]): boolean {
  return langs.includes('de') && !langs.includes('en');
}

// Move the posts this reader's client will actually display to the front, keeping
// the order the ranker chose within each group.
//
// Partition rather than filter, because the client's own filter has an escape
// hatch: a page with no post in the reader's languages is shown in full rather
// than shown empty (feed-manip.ts, "if the language filter cleared out the entire
// page, return the original set"). So an English tail behind the German is not
// waste — it is what the reader pages into once the German runs out, and dropping
// it would leave them with a short feed instead of a long one.
export function germanFirst<T extends { text: string }>(list: T[]): void {
  const first: T[] = [], rest: T[] = [];
  for (const c of list) (looksGerman(c.text) ? first : rest).push(c);
  if (!first.length || !rest.length) return;
  list.length = 0;
  for (const c of first) list.push(c);
  for (const c of rest) list.push(c);
}
