import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { ensureBackendReady } from "@/lib/auth-migrations";
import { consumeBetaLimit } from "@/lib/beta-limits";
import { generateChatCompletionWithModel, getDefaultAIModel } from "@/lib/ai";
import {
  discoverMoviesByGenres,
  discoverMovies,
  discoverTV,
  discoverTVByGenres,
  enrichMovieForFeed,
  getGenreIdsByNames,
  getMovieRecommendations,
  getMovieSimilar,
  getPopularMovies,
  getPopularTV,
  getTrendingMovies,
  getTrendingTV,
  getTVRecommendations,
  getTVSimilar,
  searchMultiple,
  TMDBMovie,
} from "@/lib/tmdb";
import { getStoredTasteProfile, AITasteProfile } from "@/lib/taste-profile";
import {
  getUserAvoidPreferences,
  mergeAvoidPreferences,
} from "@/lib/taste-preferences";
import { getUserTastePayload, MediaSeed } from "@/lib/user-movies";

export const runtime = "nodejs";

interface FeedRequestBody {
  likedMovieIds?: number[];
  dislikedMovieIds?: number[];
  savedMovieIds?: number[];
  watchedMovieIds?: number[];
  likedSeeds?: MediaSeed[];
  savedSeeds?: MediaSeed[];
  likedSnapshots?: TasteMovieSnapshot[];
  savedSnapshots?: TasteMovieSnapshot[];
  dislikedSnapshots?: TasteMovieSnapshot[];
  excludeKeys?: string[];
  likedGenres?: string[];
  dislikedGenres?: string[];
  avoidedGenres?: string[];
  avoidTerms?: string[];
  excludeMovieIds?: number[];
  cursor?: number;
  aiProfile?: AITasteProfile | null;
  storyRerankAllowed?: boolean;
  skipStoryRerank?: boolean;
}

interface Candidate {
  movie: TMDBMovie;
  score: number;
  reasons: Set<string>;
  contentScore?: number;
  matchedSignals?: string[];
  aiReason?: string;
}

const MAX_PERSONAL_SEEDS = 8;
const FEED_SIZE = 24;
const MAX_DISCOVERY_PAGE = 80;
const CONTENT_MATCH_THRESHOLD = 62;
const MIN_ACCEPTABLE_CONTENT_MATCH = 52;
const MIN_PERSONALIZED_FEED_SIZE = 12;
const MAX_CONTENT_CANDIDATES = 420;
const MODEL = getDefaultAIModel();

interface TasteMovieSnapshot {
  id: number;
  title: string;
  overview: string;
  genres: string[];
  mediaType: "movie" | "tv";
  release_date?: string;
  vote_average?: number;
  popularity?: number;
}

interface TasteFingerprint {
  movies: TasteMovieSnapshot[];
  termWeights: Map<string, number>;
  genreWeights: Map<string, number>;
  mediaWeights: Map<"movie" | "tv", number>;
  topTerms: string[];
  totalTermWeight: number;
}

function normalizeCursor(value: unknown) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) return 0;
  return Math.floor(parsed);
}

function cursorPage(cursor: number, offset = 0) {
  return ((cursor + offset) % MAX_DISCOVERY_PAGE) + 1;
}

function stringArray(value: unknown, maxLength: number) {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string").slice(0, maxLength)
    : [];
}

function numberArray(value: unknown, maxLength: number) {
  return Array.isArray(value)
    ? value.map(Number).filter((item) => Number.isInteger(item) && item > 0).slice(0, maxLength)
    : [];
}

function seedArray(value: unknown, maxLength: number): MediaSeed[] {
  if (!Array.isArray(value)) return [];

  return value
    .flatMap((item) => {
      if (!item || typeof item !== "object") return [];
      const record = item as Record<string, unknown>;
      const id = Number(record.id);
      if (!Number.isInteger(id) || id <= 0) return [];
      return [
        {
          id,
          mediaType: record.mediaType === "tv" ? "tv" : "movie",
        } satisfies MediaSeed,
      ];
    })
    .slice(0, maxLength);
}

function snapshotArray(value: unknown, maxLength: number): TasteMovieSnapshot[] {
  if (!Array.isArray(value)) return [];

  const snapshots = new Map<string, TasteMovieSnapshot>();

  for (const item of value) {
    if (!item || typeof item !== "object") continue;

    const record = item as Record<string, unknown>;
    const id = Number(record.id);
    const title = typeof record.title === "string" ? record.title.trim() : "";
    const overview =
      typeof record.overview === "string" ? record.overview.trim() : "";

    if (!Number.isInteger(id) || id <= 0 || !title || !overview) continue;

    const mediaType = record.mediaType === "tv" ? "tv" : "movie";
    const genres = stringArray(record.genres, 8);
    const key = `${mediaType}:${id}`;

    snapshots.set(key, {
      id,
      title: title.slice(0, 160),
      overview: overview.slice(0, 900),
      genres,
      mediaType,
      release_date:
        typeof record.release_date === "string"
          ? record.release_date.slice(0, 24)
          : undefined,
      vote_average:
        typeof record.vote_average === "number" &&
        Number.isFinite(record.vote_average)
          ? record.vote_average
          : undefined,
      popularity:
        typeof record.popularity === "number" && Number.isFinite(record.popularity)
          ? record.popularity
          : undefined,
    });

    if (snapshots.size >= maxLength) break;
  }

  return Array.from(snapshots.values());
}

function compactFeedBody(body: FeedRequestBody): FeedRequestBody {
  return {
    likedMovieIds: numberArray(body.likedMovieIds, 80),
    dislikedMovieIds: numberArray(body.dislikedMovieIds, 80),
    savedMovieIds: numberArray(body.savedMovieIds, 80),
    watchedMovieIds: numberArray(body.watchedMovieIds, 120),
    likedSeeds: seedArray(body.likedSeeds, 20),
    savedSeeds: seedArray(body.savedSeeds, 20),
    likedSnapshots: snapshotArray(body.likedSnapshots, 24),
    savedSnapshots: snapshotArray(body.savedSnapshots, 24),
    dislikedSnapshots: snapshotArray(body.dislikedSnapshots, 24),
    excludeKeys: stringArray(body.excludeKeys, 1600),
    likedGenres: stringArray(body.likedGenres, 20),
    dislikedGenres: stringArray(body.dislikedGenres, 20),
    avoidedGenres: stringArray(body.avoidedGenres, 24),
    avoidTerms: stringArray(body.avoidTerms, 40),
    excludeMovieIds: numberArray(body.excludeMovieIds, 1200),
    cursor: normalizeCursor(body.cursor),
    aiProfile: null,
    storyRerankAllowed: false,
    skipStoryRerank: body.skipStoryRerank === true,
  };
}

function uniqueNumbers(...arrays: Array<number[] | undefined>) {
  return Array.from(new Set(arrays.flatMap((items) => items || [])));
}

function uniqueStrings(...arrays: Array<string[] | undefined>) {
  return Array.from(new Set(arrays.flatMap((items) => items || [])));
}

function uniqueSeeds(...arrays: Array<MediaSeed[] | undefined>) {
  const seeds = new Map<string, MediaSeed>();

  arrays.forEach((items) => {
    (items || []).forEach((seed) => {
      seeds.set(`${seed.mediaType}:${seed.id}`, seed);
    });
  });

  return Array.from(seeds.values());
}

function uniqueSnapshots(...arrays: Array<TasteMovieSnapshot[] | undefined>) {
  const snapshots = new Map<string, TasteMovieSnapshot>();

  arrays.forEach((items) => {
    (items || []).forEach((snapshot) => {
      snapshots.set(`${snapshot.mediaType}:${snapshot.id}`, snapshot);
    });
  });

  return Array.from(snapshots.values());
}

function addCandidate(
  candidates: Map<string, Candidate>,
  movie: TMDBMovie,
  score: number,
  reason: string
) {
  if (!movie.poster_path || movie.adult) return;

  const key = `${movie.mediaType || "movie"}:${movie.id}`;
  const existing = candidates.get(key);
  if (existing) {
    existing.score += score;
    existing.reasons.add(reason);
    return;
  }

  candidates.set(key, {
    movie,
    score,
    reasons: new Set([reason]),
  });
}

function searchableMovieText(movie: TMDBMovie) {
  return [
    movie.title,
    movie.original_title,
    movie.overview,
    movie.genres.join(" "),
  ]
    .join(" ")
    .toLowerCase();
}

function matchesAvoidedTerm(movie: TMDBMovie, terms: string[] | undefined) {
  const normalizedTerms = (terms || [])
    .map((term) => term.toLowerCase().trim())
    .filter((term) => term.length >= 2);

  if (normalizedTerms.length === 0) return false;

  const searchableText = searchableMovieText(movie);
  return normalizedTerms.some((term) => searchableText.includes(term));
}

const STOPWORDS = new Set([
  "about",
  "after",
  "again",
  "against",
  "along",
  "also",
  "amid",
  "among",
  "around",
  "away",
  "back",
  "been",
  "before",
  "being",
  "between",
  "both",
  "but",
  "cannot",
  "come",
  "comes",
  "could",
  "down",
  "during",
  "each",
  "even",
  "ever",
  "find",
  "finds",
  "from",
  "gets",
  "goes",
  "have",
  "having",
  "her",
  "him",
  "his",
  "into",
  "its",
  "just",
  "life",
  "lives",
  "make",
  "makes",
  "must",
  "new",
  "now",
  "one",
  "only",
  "other",
  "own",
  "set",
  "she",
  "their",
  "them",
  "then",
  "there",
  "they",
  "this",
  "through",
  "two",
  "under",
  "when",
  "where",
  "while",
  "who",
  "with",
  "world",
  "young",
]);

function tokenize(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .map((token) => token.replace(/^-+|-+$/g, ""))
    .filter((token) => token.length >= 4 && !STOPWORDS.has(token));
}

function tokenSignals(value: string) {
  const tokens = tokenize(value);
  const signals = new Set(tokens);

  for (let index = 0; index < tokens.length - 1; index += 1) {
    const first = tokens[index];
    const second = tokens[index + 1];
    if (first && second) signals.add(`${first} ${second}`);
  }

  return signals;
}

function addWeight<Key extends string>(
  map: Map<Key, number>,
  key: Key,
  value: number
) {
  map.set(key, (map.get(key) || 0) + value);
}

function addMovieToFingerprint(
  movie: TasteMovieSnapshot,
  fingerprint: TasteFingerprint,
  direction: 1 | -1
) {
  const synopsisSignals = tokenSignals(movie.overview);

  synopsisSignals.forEach((signal) => {
    addWeight(fingerprint.termWeights, signal, direction * (signal.includes(" ") ? 2.8 : 1.2));
  });

  movie.genres.forEach((genre) => {
    addWeight(fingerprint.genreWeights, genre, direction * 4);
  });

  addWeight(fingerprint.mediaWeights, movie.mediaType, direction * 2);
}

function buildTasteFingerprint(body: FeedRequestBody): TasteFingerprint | null {
  const liked = uniqueSnapshots(body.likedSnapshots, body.savedSnapshots).filter(
    (movie) => movie.overview
  );
  if (liked.length < 2) return null;

  const fingerprint: TasteFingerprint = {
    movies: liked,
    termWeights: new Map(),
    genreWeights: new Map(),
    mediaWeights: new Map(),
    topTerms: [],
    totalTermWeight: 0,
  };

  liked.forEach((movie) => addMovieToFingerprint(movie, fingerprint, 1));
  (body.dislikedSnapshots || []).forEach((movie) =>
    addMovieToFingerprint(movie, fingerprint, -1)
  );

  (body.aiProfile?.likedThemes || []).forEach((theme) => {
    addWeight(fingerprint.termWeights, theme.toLowerCase(), 4);
  });
  (body.aiProfile?.dislikedThemes || []).forEach((theme) => {
    addWeight(fingerprint.termWeights, theme.toLowerCase(), -5);
  });

  const positiveTerms = Array.from(fingerprint.termWeights.entries())
    .filter(([, weight]) => weight > 1.5)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 70);

  fingerprint.topTerms = positiveTerms.slice(0, 18).map(([term]) => term);
  fingerprint.totalTermWeight =
    positiveTerms.reduce((total, [, weight]) => total + weight, 0) || 1;
  fingerprint.termWeights = new Map(positiveTerms);

  return fingerprint;
}

function qualityScore(movie: TMDBMovie) {
  const votes = Number.isFinite(movie.vote_count) ? movie.vote_count : 0;
  const rating = Number.isFinite(movie.vote_average) ? movie.vote_average : 0;
  const voteConfidence = Math.min(Math.log10(votes + 1) / 3, 1);

  return (
    rating * 0.8 * voteConfidence +
    Math.min(movie.popularity / 30, 4) +
    Math.min(Math.log10(votes + 1) * 1.6, 5)
  );
}

function scoreContentMatch(candidate: Candidate, fingerprint: TasteFingerprint) {
  const text = searchableMovieText(candidate.movie);
  let termScore = 0;
  const matchedTerms: string[] = [];

  Array.from(fingerprint.termWeights.entries()).forEach(([term, weight]) => {
    if (text.includes(term)) {
      termScore += weight;
      if (matchedTerms.length < 5 && weight > 2) matchedTerms.push(term);
    }
  });

  const normalizedTermScore = Math.min(
    termScore * 8,
    58
  );

  let genreScore = 0;
  candidate.movie.genres.forEach((genre) => {
    genreScore += Math.max(fingerprint.genreWeights.get(genre) || 0, 0);
  });
  genreScore = Math.min(genreScore * 1.6, 16);

  const mediaType = candidate.movie.mediaType === "tv" ? "tv" : "movie";
  const mediaScore = Math.max(fingerprint.mediaWeights.get(mediaType) || 0, 0);
  const sourceScore = Math.min(candidate.score / 7, 10);
  const freshnessPenalty = candidate.movie.vote_count < 25 ? 8 : 0;
  const rawScore = Math.round(
    normalizedTermScore +
      genreScore +
      Math.min(mediaScore, 4) +
      sourceScore +
      Math.min(qualityScore(candidate.movie), 8) -
      freshnessPenalty
  );
  const score = normalizedTermScore < 8 ? Math.min(rawScore, 49) : rawScore;

  candidate.contentScore = Math.max(1, Math.min(score, 100));
  candidate.matchedSignals = matchedTerms;
  return candidate.contentScore;
}

function scoreByTaste(candidate: Candidate, body: FeedRequestBody) {
  const likedGenres = new Set(body.likedGenres || []);
  const dislikedGenres = new Set(body.dislikedGenres || []);
  const avoidedGenres = new Set(body.avoidedGenres || []);
  const likedSeedMedia = new Set((body.likedSeeds || []).map((seed) => seed.mediaType));
  const savedSeedMedia = new Set((body.savedSeeds || []).map((seed) => seed.mediaType));
  const likedThemes = (body.aiProfile?.likedThemes || []).map((theme) =>
    theme.toLowerCase()
  );
  const dislikedThemes = (body.aiProfile?.dislikedThemes || []).map((theme) =>
    theme.toLowerCase()
  );
  const searchableText = [
    candidate.movie.title,
    candidate.movie.overview,
    candidate.movie.genres.join(" "),
  ]
    .join(" ")
    .toLowerCase();

  candidate.movie.genres.forEach((genre) => {
    if (likedGenres.has(genre)) candidate.score += 8;
    if (dislikedGenres.has(genre)) candidate.score -= 14;
    if (avoidedGenres.has(genre)) candidate.score -= 34;
    const aiGenreWeight = body.aiProfile?.genreWeights[genre];
    if (typeof aiGenreWeight === "number") {
      candidate.score += aiGenreWeight * 2;
    }
  });

  const candidateMedia = candidate.movie.mediaType === "tv" ? "tv" : "movie";
  if (likedSeedMedia.has(candidateMedia)) candidate.score += 5;
  if (savedSeedMedia.has(candidateMedia)) candidate.score += 3;

  likedThemes.forEach((theme) => {
    if (searchableText.includes(theme)) candidate.score += 7;
  });
  dislikedThemes.forEach((theme) => {
    if (searchableText.includes(theme)) candidate.score -= 12;
  });
  if (matchesAvoidedTerm(candidate.movie, body.avoidTerms)) {
    candidate.score -= 48;
  }

  candidate.score += Math.min(candidate.movie.vote_average, 10);
  candidate.score += Math.min(candidate.movie.popularity / 50, 8);
  candidate.score += Math.min(candidate.movie.vote_count / 1000, 4);
  if (candidate.movie.vote_count < 25) candidate.score -= 8;
}

function makeFeedReason(candidate: Candidate) {
  const reasons = Array.from(candidate.reasons);
  if (candidate.aiReason && candidate.contentScore) {
    return candidate.aiReason;
  }

  if (candidate.contentScore && candidate.contentScore >= MIN_ACCEPTABLE_CONTENT_MATCH) {
    const signals = (candidate.matchedSignals || [])
      .slice(0, 3)
      .map((signal) => signal.replace(/\b\w/g, (letter) => letter.toUpperCase()));

    if (signals.length > 0) {
      return `Strong story match: shares ${signals.join(", ")} with titles you liked.`;
    }

    return "Strong story match based on the synopsis patterns in titles you liked.";
  }

  if (reasons.includes("because of movies you liked")) {
    return "Because it lines up with movies you have liked.";
  }
  if (reasons.includes("because of your saved movies")) {
    return "Because it resembles movies you saved for later.";
  }
  if (reasons.includes("because of your favorite genres")) {
    return "Because it matches genres you keep responding to.";
  }
  if (reasons.includes("AI taste profile")) {
    return "Because FlickBuddy's AI taste profile found a deeper match.";
  }
  if (reasons.includes("trending this week")) {
    return "Trending right now and likely worth a look.";
  }
  if (reasons.includes("fresh discovery")) {
    return "A fresh discovery lane pick to keep your feed varied.";
  }
  return "A popular pick that fits the discovery feed.";
}

function contentLikedSeeds(body: FeedRequestBody) {
  return (
    body.likedSeeds?.length
      ? body.likedSeeds
      : (body.likedMovieIds || []).map((id) => ({ id, mediaType: "movie" as const }))
  ).slice(0, 6);
}

function contentSavedSeeds(body: FeedRequestBody) {
  return (
    body.savedSeeds?.length
      ? body.savedSeeds
      : (body.savedMovieIds || []).map((id) => ({ id, mediaType: "movie" as const }))
  ).slice(0, 4);
}

async function addContentCandidateBatch({
  candidates,
  body,
  fingerprint,
  cursor,
  offset,
}: {
  candidates: Map<string, Candidate>;
  body: FeedRequestBody;
  fingerprint: TasteFingerprint;
  cursor: number;
  offset: number;
}) {
  const likedSeeds = contentLikedSeeds(body);
  const savedSeeds = contentSavedSeeds(body);
  const likedGenreIds = getGenreIdsByNames(body.likedGenres || []);
  const page = cursorPage(cursor, offset);
  const nextPage = cursorPage(cursor, offset + 1);
  const searchTerms = Array.from(
    new Set([
      ...fingerprint.topTerms.filter((term) => term.includes(" ")).slice(0, 6),
      ...(body.aiProfile?.recommendedQueries || []),
      ...(body.aiProfile?.referenceTitles || []),
    ])
  ).slice(0, 10);

  const [
    likedRecommendationBatches,
    likedSimilarBatches,
    savedRecommendationBatches,
    genreBatches,
    searchMatches,
    broadDiscoveryBatches,
  ] = await Promise.all([
    Promise.all(
      likedSeeds.flatMap((seed) => [
        getSeedRecommendations(seed, page),
        getSeedRecommendations(seed, nextPage),
      ])
    ),
    Promise.all(likedSeeds.map((seed) => getSeedSimilar(seed, page))),
    Promise.all(savedSeeds.map((seed) => getSeedRecommendations(seed, page))),
    likedGenreIds.length > 0
      ? Promise.all([
          discoverMoviesByGenres(likedGenreIds, page, "popularity.desc", 60),
          discoverTVByGenres(likedGenreIds, page, "popularity.desc", 60),
          discoverMoviesByGenres(likedGenreIds, nextPage, "vote_average.desc", 140),
          discoverTVByGenres(likedGenreIds, nextPage, "vote_average.desc", 140),
        ])
      : Promise.resolve([]),
    offset === 0 && searchTerms.length > 0
      ? searchMultiple(searchTerms)
      : Promise.resolve([]),
    offset >= 5
      ? Promise.all([
          discoverMovies(page),
          discoverTV(page),
          getTrendingMovies(page),
          getTrendingTV(page),
        ])
      : Promise.resolve([]),
  ]);

  likedRecommendationBatches.flat().forEach((movie) =>
    addCandidate(candidates, movie, 38, "story match from liked titles")
  );
  likedSimilarBatches.flat().forEach((movie) =>
    addCandidate(candidates, movie, 34, "similar to titles you liked")
  );
  savedRecommendationBatches.flat().forEach((movie) =>
    addCandidate(candidates, movie, 24, "because of your saved movies")
  );
  genreBatches.flat().forEach((movie) =>
    addCandidate(candidates, movie, 14, "because of your favorite genres")
  );
  searchMatches.forEach((movie) =>
    addCandidate(candidates, movie, 22, "synopsis search match")
  );
  broadDiscoveryBatches.flat().forEach((movie) =>
    addCandidate(candidates, movie, 5, "fresh discovery")
  );
}

function rankContentCandidates(
  candidates: Map<string, Candidate>,
  body: FeedRequestBody,
  fingerprint: TasteFingerprint
) {
  const excludedKeys = new Set(body.excludeKeys || []);
  const excluded = new Set([
    ...(excludedKeys.size === 0 ? body.excludeMovieIds || [] : []),
    ...(body.dislikedMovieIds || []),
    ...(body.watchedMovieIds || []),
  ]);
  const avoidedGenres = new Set(body.avoidedGenres || []);

  return Array.from(candidates.values())
    .filter((candidate) => {
      const key = `${candidate.movie.mediaType || "movie"}:${candidate.movie.id}`;
      return (
        candidate.movie.overview &&
        !excludedKeys.has(key) &&
        !excluded.has(candidate.movie.id) &&
        !candidate.movie.genres.some((genre) => avoidedGenres.has(genre)) &&
        !matchesAvoidedTerm(candidate.movie, body.avoidTerms)
      );
    })
    .map((candidate) => {
      scoreContentMatch(candidate, fingerprint);
      return candidate;
    })
    .sort(
      (a, b) =>
        (b.contentScore || 0) - (a.contentScore || 0) ||
        qualityScore(b.movie) - qualityScore(a.movie)
    );
}

function parseModelJSON(raw: string) {
  const cleaned = raw
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  try {
    return JSON.parse(cleaned) as Record<string, unknown>;
  } catch {
    const match = cleaned.match(/\{[\s\S]*\}/);
    if (!match) throw new Error("Story rerank response was not JSON");
    return JSON.parse(match[0]) as Record<string, unknown>;
  }
}

function truncate(value: string, maxLength: number) {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength).trim()}...`;
}

function candidateKey(movie: Pick<TMDBMovie, "id" | "mediaType">) {
  return `${movie.mediaType === "tv" ? "tv" : "movie"}:${movie.id}`;
}

function buildStoryRankPrompt({
  fingerprint,
  candidates,
  body,
}: {
  fingerprint: TasteFingerprint;
  candidates: Candidate[];
  body: FeedRequestBody;
}) {
  const liked = fingerprint.movies.slice(0, 12).map((movie) => ({
    title: movie.title,
    mediaType: movie.mediaType,
    genres: movie.genres,
    overview: truncate(movie.overview, 520),
  }));
  const candidateList = candidates.slice(0, 56).map((candidate) => ({
    key: candidateKey(candidate.movie),
    title: candidate.movie.title,
    mediaType: candidate.movie.mediaType || "movie",
    genres: candidate.movie.genres,
    overview: truncate(candidate.movie.overview || "", 520),
    vote_average: candidate.movie.vote_average,
    vote_count: candidate.movie.vote_count,
    retrievalScore: candidate.score,
    heuristicScore: candidate.contentScore || 0,
  }));

  return `You are FlickBuddy's story-level recommender. Rank TMDB candidates by how strongly their synopsis, themes, premise, tone, and genre blend relate to movies and series the user liked during onboarding.

Liked onboarding titles:
${JSON.stringify(liked, null, 2)}

Avoid preferences:
- genres: ${(body.avoidedGenres || []).join(", ") || "none"}
- terms: ${(body.avoidTerms || []).join(", ") || "none"}

Candidate titles:
${JSON.stringify(candidateList, null, 2)}

Return ONLY valid JSON:
{
  "ranked": [
    {
      "key": "movie:123",
      "score": 92,
      "reason": "specific one-sentence synopsis-level reason",
      "matchedThemes": ["theme or story signal"]
    }
  ]
}

Rules:
- Only include candidates that are genuinely strong matches to the liked-title synopses.
- Score 90-100 for excellent story/premise overlap, 75-89 for good overlap, 60-74 for acceptable overlap.
- Do not include weak matches that only share a broad genre.
- Penalize candidates matching avoid preferences.
- Use only candidate keys from the provided list.`;
}

async function rerankContentCandidatesWithAI({
  body,
  fingerprint,
  candidates,
}: {
  body: FeedRequestBody;
  fingerprint: TasteFingerprint;
  candidates: Candidate[];
}) {
  if (!body.storyRerankAllowed || candidates.length === 0) return [];

  try {
    const raw = await generateChatCompletionWithModel(
      buildStoryRankPrompt({ fingerprint, candidates, body }),
      MODEL,
      0.2,
      3200
    );
    const parsed = parseModelJSON(raw);
    const ranked = Array.isArray(parsed.ranked) ? parsed.ranked : [];
    const candidateMap = new Map(
      candidates.map((candidate) => [candidateKey(candidate.movie), candidate])
    );
    const used = new Set<string>();

    return ranked
      .flatMap((item) => {
        if (!item || typeof item !== "object") return [];
        const record = item as Record<string, unknown>;
        const key = typeof record.key === "string" ? record.key : "";
        const candidate = candidateMap.get(key);
        if (!candidate || used.has(key)) return [];
        used.add(key);

        const score = Math.min(Math.max(Math.round(Number(record.score)), 1), 100);
        if (!Number.isFinite(score) || score < MIN_ACCEPTABLE_CONTENT_MATCH) {
          return [];
        }

        const themes = Array.isArray(record.matchedThemes)
          ? record.matchedThemes
              .filter((theme): theme is string => typeof theme === "string")
              .map((theme) => theme.trim())
              .filter(Boolean)
              .slice(0, 5)
          : [];

        candidate.contentScore = score;
        candidate.matchedSignals = themes;
        candidate.aiReason =
          typeof record.reason === "string"
            ? record.reason.trim().slice(0, 220)
            : undefined;
        candidate.reasons.add("AI story rerank");
        return [candidate];
      })
      .slice(0, FEED_SIZE);
  } catch (error) {
    console.error("Story rerank failed:", error);
    return [];
  }
}

async function buildContentBasedFeed(body: FeedRequestBody) {
  const fingerprint = buildTasteFingerprint(body);
  if (!fingerprint) return null;

  const cursor = normalizeCursor(body.cursor);
  const candidates = new Map<string, Candidate>();
  let ranked: Candidate[] = [];

  for (const offset of [0, 2, 5, 9, 14, 21, 34]) {
    await addContentCandidateBatch({
      candidates,
      body,
      fingerprint,
      cursor,
      offset,
    });

    ranked = rankContentCandidates(candidates, body, fingerprint);
    const strongCount = ranked.filter(
      (candidate) => (candidate.contentScore || 0) >= CONTENT_MATCH_THRESHOLD
    ).length;

    if (strongCount >= FEED_SIZE || candidates.size >= MAX_CONTENT_CANDIDATES) {
      break;
    }
  }

  ranked = rankContentCandidates(candidates, body, fingerprint);
  const aiRanked = await rerankContentCandidatesWithAI({
    body,
    fingerprint,
    candidates: ranked.slice(0, 56),
  });
  if (aiRanked.length > 0) {
    return Promise.all(
      aiRanked.map((candidate) =>
        enrichMovieForFeed(
          candidate.movie,
          Math.min(Math.max(candidate.contentScore || 1, 1), 100),
          makeFeedReason(candidate)
        )
      )
    );
  }

  const matches = ranked.filter(
    (candidate) => (candidate.contentScore || 0) >= MIN_ACCEPTABLE_CONTENT_MATCH
  );
  const selected = matches.slice(0, FEED_SIZE);

  if (selected.length < MIN_PERSONALIZED_FEED_SIZE) return null;

  return Promise.all(
    selected.map((candidate) =>
      enrichMovieForFeed(
        candidate.movie,
        Math.min(Math.max(candidate.contentScore || 1, 1), 100),
        makeFeedReason(candidate)
      )
    )
  );
}

async function buildFeed(body: FeedRequestBody) {
  const contentResults = await buildContentBasedFeed(body);
  if (contentResults?.length) return contentResults;

  const cursor = normalizeCursor(body.cursor);
  const likedSeeds = (
    body.likedSeeds?.length
      ? body.likedSeeds
      : (body.likedMovieIds || []).map((id) => ({ id, mediaType: "movie" as const }))
  ).slice(0, MAX_PERSONAL_SEEDS);
  const savedSeeds = (
    body.savedSeeds?.length
      ? body.savedSeeds
      : (body.savedMovieIds || []).map((id) => ({ id, mediaType: "movie" as const }))
  ).slice(0, MAX_PERSONAL_SEEDS);
  const likedGenreIds = getGenreIdsByNames(body.likedGenres || []);
  const hasTasteSignals =
    likedSeeds.length > 0 ||
    savedSeeds.length > 0 ||
    likedGenreIds.length > 0 ||
    Boolean(body.aiProfile);
  const primaryPage = cursorPage(cursor);
  const secondaryPage = cursorPage(cursor, 3);
  const explorationPage = cursorPage(cursor, 9);
  const deepDiscoveryPage = cursorPage(cursor, 17);
  const qualityDiscoveryPage = cursorPage(cursor, 31);
  const recommendationPage = cursorPage(cursor, 1);
  const aiSearchTerms = [
    ...(body.aiProfile?.recommendedQueries || []),
    ...(body.aiProfile?.referenceTitles || []),
  ].slice(cursor % 3, cursor % 3 + 6);

  const [
    trendingMovies,
    trendingTV,
    popularMovies,
    popularTV,
    discoveryMovies,
    discoveryTV,
    genreMovieMatches,
    genreTVMatches,
    deepDiscoveryMovies,
    deepDiscoveryTV,
    qualityGenreMovieMatches,
    qualityGenreTVMatches,
    aiSearchMatches,
    likedRecommendationBatches,
    savedRecommendationBatches,
  ] = await Promise.all([
    getTrendingMovies(primaryPage),
    getTrendingTV(primaryPage),
    getPopularMovies(secondaryPage),
    getPopularTV(secondaryPage),
    discoverMovies(explorationPage),
    discoverTV(explorationPage),
    discoverMoviesByGenres(likedGenreIds, secondaryPage),
    discoverTVByGenres(likedGenreIds, secondaryPage),
    discoverMovies(deepDiscoveryPage),
    discoverTV(deepDiscoveryPage),
    discoverMoviesByGenres(likedGenreIds, qualityDiscoveryPage, "vote_average.desc", 140),
    discoverTVByGenres(likedGenreIds, qualityDiscoveryPage, "vote_average.desc", 140),
    aiSearchTerms.length > 0 ? searchMultiple(aiSearchTerms) : Promise.resolve([]),
    Promise.all(likedSeeds.map((seed) => getSeedRecommendations(seed, recommendationPage))),
    Promise.all(savedSeeds.map((seed) => getSeedRecommendations(seed, recommendationPage))),
  ]);

  const candidates = new Map<string, Candidate>();
  const trendingScore = hasTasteSignals ? 9 : 16;
  const popularScore = hasTasteSignals ? 4 : 8;
  const discoveryScore = hasTasteSignals ? 10 : 12;

  trendingMovies.forEach((movie) =>
    addCandidate(candidates, movie, trendingScore, "trending this week")
  );
  trendingTV.forEach((movie) =>
    addCandidate(candidates, movie, trendingScore, "trending this week")
  );
  popularMovies.forEach((movie) => addCandidate(candidates, movie, popularScore, "popular on TMDB"));
  popularTV.forEach((movie) => addCandidate(candidates, movie, popularScore, "popular on TMDB"));
  discoveryMovies.forEach((movie) =>
    addCandidate(candidates, movie, discoveryScore, "fresh discovery")
  );
  discoveryTV.forEach((movie) => addCandidate(candidates, movie, discoveryScore, "fresh discovery"));
  [...genreMovieMatches, ...genreTVMatches].forEach((movie) =>
    addCandidate(candidates, movie, 26, "because of your favorite genres")
  );
  [...deepDiscoveryMovies, ...deepDiscoveryTV].forEach((movie) =>
    addCandidate(candidates, movie, hasTasteSignals ? 7 : 10, "fresh discovery")
  );
  [...qualityGenreMovieMatches, ...qualityGenreTVMatches].forEach((movie) =>
    addCandidate(candidates, movie, 18, "because of your favorite genres")
  );
  aiSearchMatches.forEach((movie) =>
    addCandidate(candidates, movie, 38, "AI taste profile")
  );

  likedRecommendationBatches.flat().forEach((movie) =>
    addCandidate(candidates, movie, 44, "because of movies you liked")
  );

  savedRecommendationBatches.flat().forEach((movie) =>
    addCandidate(candidates, movie, 36, "because of your saved movies")
  );

  const excludedKeys = new Set(body.excludeKeys || []);
  const excluded = new Set([
    ...(excludedKeys.size === 0 ? body.excludeMovieIds || [] : []),
    ...(body.dislikedMovieIds || []),
    ...(body.watchedMovieIds || []),
  ]);
  const avoidedGenres = new Set(body.avoidedGenres || []);

  const rankedCandidates = diversifyCandidates(
    Array.from(candidates.values())
    .filter((candidate) => {
      const key = `${candidate.movie.mediaType || "movie"}:${candidate.movie.id}`;
      return (
        !excludedKeys.has(key) &&
        !excluded.has(candidate.movie.id) &&
        !candidate.movie.genres.some((genre) => avoidedGenres.has(genre)) &&
        !matchesAvoidedTerm(candidate.movie, body.avoidTerms)
      );
    })
    .map((candidate) => {
      scoreByTaste(candidate, body);
      candidate.score += Math.random() * 6;
      return candidate;
    })
    .sort((a, b) => b.score - a.score),
    FEED_SIZE
  );

  return Promise.all(
    rankedCandidates.map((candidate) =>
      enrichMovieForFeed(
        candidate.movie,
        Math.min(Math.max(Math.round(candidate.score), 1), 100),
        makeFeedReason(candidate)
      )
    )
  );
}

function diversifyCandidates(candidates: Candidate[], limit: number) {
  const selected: Candidate[] = [];
  const genreCounts = new Map<string, number>();
  const mediaCounts = new Map<string, number>();
  const maxPerPrimaryGenre = Math.max(5, Math.ceil(limit / 4));
  const maxPerMediaType = Math.ceil(limit * 0.72);

  for (const candidate of candidates) {
    if (selected.length >= limit) break;

    const primaryGenre = candidate.movie.genres[0] || "Unknown";
    const mediaType = candidate.movie.mediaType === "tv" ? "tv" : "movie";
    const genreCount = genreCounts.get(primaryGenre) || 0;
    const mediaCount = mediaCounts.get(mediaType) || 0;

    if (genreCount >= maxPerPrimaryGenre || mediaCount >= maxPerMediaType) {
      continue;
    }

    selected.push(candidate);
    genreCounts.set(primaryGenre, genreCount + 1);
    mediaCounts.set(mediaType, mediaCount + 1);
  }

  if (selected.length >= limit) return selected;

  for (const candidate of candidates) {
    if (selected.length >= limit) break;
    if (selected.includes(candidate)) continue;
    selected.push(candidate);
  }

  return selected;
}

async function getSeedRecommendations(seed: MediaSeed, page: number) {
  return seed.mediaType === "tv"
    ? getTVRecommendations(seed.id, page)
    : getMovieRecommendations(seed.id, page);
}

async function getSeedSimilar(seed: MediaSeed, page: number) {
  return seed.mediaType === "tv"
    ? getTVSimilar(seed.id, page)
    : getMovieSimilar(seed.id, page);
}

export async function GET() {
  await ensureBackendReady();
  const results = await buildFeed({});
  return NextResponse.json({ results, nextCursor: 1 });
}

export async function POST(request: NextRequest) {
  await ensureBackendReady();
  let body: FeedRequestBody = {};

  try {
    body = (await request.json()) as FeedRequestBody;
  } catch {
    body = {};
  }
  body = compactFeedBody(body);

  const session = await auth.api.getSession({
    headers: request.headers,
  });

  if (session) {
    const serverTaste = getUserTastePayload(session.user.id);
    const serverAvoids = getUserAvoidPreferences(session.user.id);
    const avoids = mergeAvoidPreferences(serverAvoids, {
      genres: body.avoidedGenres,
      terms: body.avoidTerms,
    });
    body = {
      likedMovieIds: uniqueNumbers(serverTaste.likedMovieIds, body.likedMovieIds),
      dislikedMovieIds: uniqueNumbers(
        serverTaste.dislikedMovieIds,
        body.dislikedMovieIds
      ),
      savedMovieIds: uniqueNumbers(serverTaste.savedMovieIds, body.savedMovieIds),
      watchedMovieIds: uniqueNumbers(
        serverTaste.watchedMovieIds,
        body.watchedMovieIds
      ),
      likedSeeds: uniqueSeeds(serverTaste.likedSeeds, body.likedSeeds),
      savedSeeds: uniqueSeeds(serverTaste.savedSeeds, body.savedSeeds),
      likedSnapshots: uniqueSnapshots(
        serverTaste.likedSnapshots,
        body.likedSnapshots
      ),
      savedSnapshots: uniqueSnapshots(
        serverTaste.savedSnapshots,
        body.savedSnapshots
      ),
      dislikedSnapshots: uniqueSnapshots(
        serverTaste.dislikedSnapshots,
        body.dislikedSnapshots
      ),
      likedGenres: uniqueStrings(serverTaste.likedGenres, body.likedGenres),
      dislikedGenres: uniqueStrings(
        serverTaste.dislikedGenres,
        body.dislikedGenres
      ),
      avoidedGenres: avoids.genres,
      avoidTerms: avoids.terms,
      excludeKeys: uniqueStrings(body.excludeKeys, serverTaste.excludeKeys),
      excludeMovieIds: uniqueNumbers(
        body.excludeMovieIds,
        serverTaste.excludeMovieIds
      ),
      cursor: normalizeCursor(body.cursor),
      aiProfile: getStoredTasteProfile(session.user.id),
      skipStoryRerank: body.skipStoryRerank,
    };
  }

  const hasStoryRerankInput =
    uniqueSnapshots(body.likedSnapshots, body.savedSnapshots).length >= 2;
  if (hasStoryRerankInput && !body.skipStoryRerank) {
    const storyLimit = consumeBetaLimit({
      request,
      userId: session?.user.id,
      action: "feed_story_rerank",
    });
    body.storyRerankAllowed = storyLimit.allowed;
  }

  const results = await buildFeed(body);
  return NextResponse.json({
    results,
    nextCursor: normalizeCursor(body.cursor) + 1,
  });
}
