import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { betaLimitHeaders, consumeBetaLimit } from "@/lib/beta-limits";
import { ensureBackendReady } from "@/lib/auth-migrations";
import { generateChatCompletionWithModel, getDefaultAIModel } from "@/lib/ai";
import {
  discoverMoviesByGenres,
  discoverTVByGenres,
  getGenreIdsByNames,
  getKnownGenreNames,
  getMovieRecommendations,
  getMovieReviews,
  getTrendingMovies,
  getTrendingTV,
  getTVRecommendations,
  getTVReviews,
  searchMulti,
  searchMultiple,
  TMDBMovie,
  TMDBMovieReview,
} from "@/lib/tmdb";
import {
  getUserAvoidPreferences,
  mergeAvoidPreferences,
} from "@/lib/taste-preferences";
import { getUserTastePayload } from "@/lib/user-movies";

export const runtime = "nodejs";

type DiscoverMode = "smart" | "movie" | "mood" | "genre" | "series";
type MediaPreference = "all" | "movie" | "tv";

interface DiscoverIntent {
  genres: string[];
  mediaType: MediaPreference;
  searchTerms: string[];
  referenceTitles: string[];
  moodTags: string[];
  themeTags: string[];
  characterTags: string[];
  avoidTags: string[];
  explanation: string;
}

interface UserTastePayload {
  likedMovieIds: number[];
  dislikedMovieIds: number[];
  savedMovieIds: number[];
  watchedMovieIds: number[];
  excludeKeys?: string[];
  likedGenres: string[];
  dislikedGenres: string[];
  avoidedGenres: string[];
  avoidTerms: string[];
  excludeMovieIds: number[];
}

interface ServerTasteContext {
  userId: string | null;
  taste: UserTastePayload | null;
}

interface RankedResult {
  key: string;
  relevanceScore: number;
  reason: string;
  caveat?: string;
  highlightedThemes?: string[];
}

interface EnrichedCandidate {
  movie: TMDBMovie;
  reviews: TMDBMovieReview[];
}

const MODEL = getDefaultAIModel();
const MAX_QUERY_LENGTH = 600;
const MAX_REFERENCE_TITLES = 6;
const MAX_SEARCH_TERMS = 8;
const MAX_CANDIDATES_FOR_RANKING = 28;
const MAX_REVIEWS_PER_CANDIDATE = 6;
const RATE_LIMIT_WINDOW_MS = 60_000;
const ANONYMOUS_RATE_LIMIT = 8;
const AUTHENTICATED_RATE_LIMIT = 20;

const rateLimitBuckets = new Map<string, { count: number; resetAt: number }>();

function normalize(value: string) {
  return value
    .toLowerCase()
    .replace(/^the\s+/, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function titleMatchScore(query: string, movie: TMDBMovie) {
  const normalizedQuery = normalize(query);
  const title = normalize(movie.title);
  const originalTitle = normalize(movie.original_title || movie.title);

  if (!normalizedQuery) return 0;
  if (title === normalizedQuery || originalTitle === normalizedQuery) return 100;
  if (title.startsWith(normalizedQuery)) return 84;
  if (title.includes(normalizedQuery) && normalizedQuery.length >= 4) return 70;

  return 0;
}

function safeParseJSON<T>(raw: string): T {
  const cleaned = raw
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  try {
    return JSON.parse(cleaned) as T;
  } catch {
    const match = cleaned.match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]) as T;
    throw new Error("Could not parse AI response");
  }
}

function dedupeMovies(movies: TMDBMovie[]) {
  const seen = new Set<string>();
  return movies.filter((movie) => {
    const key = `${movie.mediaType || "movie"}:${movie.id}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function decorateMovie(
  movie: TMDBMovie,
  index: number,
  feedReason: string,
  matchReason?: string,
  highlightedThemes?: string[],
  matchCaveat?: string
) {
  return {
    ...movie,
    relevanceScore: Math.max(96 - index * 2, 45),
    feedReason,
    matchReason,
    matchCaveat,
    highlightedThemes,
    reviews: [],
    hasContentAnalysis: Boolean(matchReason),
  };
}

function qualityPriorityScore(movie: TMDBMovie) {
  const rating = Number.isFinite(movie.vote_average) ? movie.vote_average : 0;
  const votes = Number.isFinite(movie.vote_count) ? movie.vote_count : 0;
  const confidence = Math.min(Math.log10(votes + 1) / 3, 1);
  const ratingScore = rating * 10 * confidence;
  const supportScore = Math.min(Math.log10(votes + 1) * 6, 18);
  const popularityScore = Math.min(movie.popularity / 12, 12);

  return Math.min(ratingScore + supportScore + popularityScore, 100);
}

function ratingTieBreaker(movie: TMDBMovie) {
  return qualityPriorityScore(movie) + Math.min(movie.popularity / 10, 10);
}

function sanitizeList(values: unknown, limit: number) {
  if (!Array.isArray(values)) return [];
  return values
    .filter((value): value is string => typeof value === "string")
    .map((value) => value.trim())
    .filter(Boolean)
    .slice(0, limit);
}

function emptyTaste(): UserTastePayload {
  return {
    likedMovieIds: [],
    dislikedMovieIds: [],
    savedMovieIds: [],
    watchedMovieIds: [],
    excludeKeys: [],
    likedGenres: [],
    dislikedGenres: [],
    avoidedGenres: [],
    avoidTerms: [],
    excludeMovieIds: [],
  };
}

function getRequestAvoids(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  return mergeAvoidPreferences({
    genres: searchParams.getAll("avoidGenre"),
    terms: searchParams.getAll("avoidTerm"),
  });
}

function withAvoids(
  taste: UserTastePayload | null,
  avoids: { genres: string[]; terms: string[] }
): UserTastePayload | null {
  if (!taste && avoids.genres.length === 0 && avoids.terms.length === 0) {
    return null;
  }

  const nextTaste = taste || emptyTaste();
  const merged = mergeAvoidPreferences(
    {
      genres: nextTaste.avoidedGenres,
      terms: nextTaste.avoidTerms,
    },
    avoids
  );

  return {
    ...nextTaste,
    avoidedGenres: merged.genres,
    avoidTerms: merged.terms,
  };
}

function movieText(movie: TMDBMovie) {
  return [
    movie.title,
    movie.original_title,
    movie.overview,
    movie.genres.join(" "),
  ]
    .join(" ")
    .toLowerCase();
}

function matchesAvoidPreferences(
  movie: TMDBMovie,
  taste: UserTastePayload | null
) {
  if (!taste) return false;

  const avoidedGenres = new Set(taste.avoidedGenres || []);
  if (movie.genres.some((genre) => avoidedGenres.has(genre))) return true;

  const terms = (taste.avoidTerms || [])
    .map((term) => term.toLowerCase().trim())
    .filter((term) => term.length >= 2);
  if (terms.length === 0) return false;

  const searchable = movieText(movie);
  return terms.some((term) => searchable.includes(term));
}

function normalizeMediaPreference(value: unknown, mode: DiscoverMode): MediaPreference {
  if (mode === "series") return "tv";
  if (mode === "movie") return "movie";
  if (value === "movie" || value === "tv" || value === "all") return value;
  return "all";
}

function enrichIntentWithHeuristics(query: string, intent: DiscoverIntent) {
  const next: DiscoverIntent = {
    ...intent,
    genres: [...intent.genres],
    searchTerms: [...intent.searchTerms],
    referenceTitles: [...intent.referenceTitles],
    moodTags: [...intent.moodTags],
    themeTags: [...intent.themeTags],
    characterTags: [...intent.characterTags],
    avoidTags: [...intent.avoidTags],
  };

  const addUnique = (target: string[], ...values: string[]) => {
    for (const value of values) {
      if (!target.some((item) => item.toLowerCase() === value.toLowerCase())) {
        target.push(value);
      }
    }
  };

  if (/mind.?bending|plot twist|twist ending|psychological/i.test(query)) {
    addUnique(next.genres, "Science Fiction", "Mystery", "Thriller");
    addUnique(next.referenceTitles, "Inception", "Memento", "The Prestige", "Shutter Island", "Dark");
    addUnique(next.searchTerms, "psychological thriller", "mystery thriller", "time travel");
    addUnique(next.themeTags, "twists", "unreliable reality", "mystery");
  }

  if (/smart|genius|intelligent|brilliant|main character|protagonist/i.test(query)) {
    addUnique(next.referenceTitles, "Sherlock", "A Beautiful Mind", "The Imitation Game", "Mr. Robot");
    addUnique(next.searchTerms, "genius protagonist", "detective mystery", "strategic thriller");
    addUnique(next.characterTags, "smart lead", "genius protagonist");
  }

  if (/dark comedy|satire|rich people|wealthy|elite/i.test(query)) {
    addUnique(next.genres, "Comedy", "Drama");
    addUnique(next.referenceTitles, "The Menu", "Succession", "Triangle of Sadness", "Parasite");
    addUnique(next.searchTerms, "dark comedy", "satire", "class conflict");
    addUnique(next.themeTags, "wealth", "satire", "class conflict");
  }

  return {
    ...next,
    genres: Array.from(new Set(next.genres)).slice(0, 5),
    searchTerms: Array.from(new Set(next.searchTerms)).slice(0, MAX_SEARCH_TERMS),
    referenceTitles: Array.from(new Set(next.referenceTitles)).slice(
      0,
      MAX_REFERENCE_TITLES
    ),
    moodTags: Array.from(new Set(next.moodTags)).slice(0, 6),
    themeTags: Array.from(new Set(next.themeTags)).slice(0, 8),
    characterTags: Array.from(new Set(next.characterTags)).slice(0, 6),
    avoidTags: Array.from(new Set(next.avoidTags)).slice(0, 6),
  };
}

function buildIntentPrompt(query: string, mode: DiscoverMode, taste: UserTastePayload | null) {
  const knownGenres = getKnownGenreNames().join(", ");
  const tasteText = taste
    ? `\nUser taste signals from server-side interactions:
- liked genres: ${taste.likedGenres.slice(0, 8).join(", ") || "none yet"}
- disliked genres: ${taste.dislikedGenres.slice(0, 8).join(", ") || "none yet"}
- avoided genres: ${taste.avoidedGenres.slice(0, 8).join(", ") || "none set"}
- avoided titles/themes: ${taste.avoidTerms.slice(0, 8).join(", ") || "none set"}
Use these as soft preferences only. The user's current request is more important.\n`
    : "";

  return `You are FlickBuddy's movie and TV discovery planner.

User request: "${query}"
Discovery mode: ${mode}
Known TMDB genres: ${knownGenres}
${tasteText}
Analyze the request and convert it into structured search intent. Do not recommend titles here. Return genres and retrieval hints that can fetch real TMDB movies or series.

Respond with ONLY valid JSON:
{
  "genres": ["TMDB genre name"],
  "mediaType": "all | movie | tv",
  "searchTerms": ["terms TMDB text search can match"],
  "referenceTitles": ["known movie or series title if mentioned or strongly implied"],
  "moodTags": ["mood words"],
  "themeTags": ["story/theme words"],
  "characterTags": ["character traits"],
  "avoidTags": ["things the user wants to avoid"],
  "explanation": "one concise sentence explaining the interpreted request"
}

Rules:
- genres must use names from the Known TMDB genres list when possible.
- Use 2-5 genres, 3-8 searchTerms, and at most 6 referenceTitles.
- If the user asks for shows/series, mediaType must be "tv".
- If the user asks for movies only, mediaType must be "movie".
- If uncertain, use mediaType "all".
- Do not invent fictional titles.`;
}

async function getDiscoverIntent(
  query: string,
  mode: DiscoverMode,
  taste: UserTastePayload | null
): Promise<DiscoverIntent> {
  const fallback: DiscoverIntent = {
    genres: [],
    mediaType: mode === "series" ? "tv" : "all",
    searchTerms: [query],
    referenceTitles: [],
    moodTags: [],
    themeTags: [],
    characterTags: [],
    avoidTags: [],
    explanation: `Matched "${query}".`,
  };

  try {
    const raw = await generateChatCompletionWithModel(
      buildIntentPrompt(query, mode, taste),
      MODEL,
      0.2,
      1600
    );
    const parsed = safeParseJSON<Partial<DiscoverIntent>>(raw);
    const intent: DiscoverIntent = {
      genres: sanitizeList(parsed.genres, 5),
      mediaType: normalizeMediaPreference(parsed.mediaType, mode),
      searchTerms: sanitizeList(parsed.searchTerms, MAX_SEARCH_TERMS),
      referenceTitles: sanitizeList(parsed.referenceTitles, MAX_REFERENCE_TITLES),
      moodTags: sanitizeList(parsed.moodTags, 6),
      themeTags: sanitizeList(parsed.themeTags, 8),
      characterTags: sanitizeList(parsed.characterTags, 6),
      avoidTags: sanitizeList(parsed.avoidTags, 6),
      explanation:
        typeof parsed.explanation === "string" && parsed.explanation.trim()
          ? parsed.explanation.trim()
          : fallback.explanation,
    };

    if (!intent.searchTerms.length) intent.searchTerms = [query];
    return enrichIntentWithHeuristics(query, intent);
  } catch {
    return enrichIntentWithHeuristics(query, fallback);
  }
}

function matchesMediaPreference(movie: TMDBMovie, preference: MediaPreference) {
  if (preference === "all") return true;
  return (movie.mediaType || "movie") === preference;
}

function preScoreCandidate(
  movie: TMDBMovie,
  intent: DiscoverIntent,
  taste: UserTastePayload | null,
  titleKeys: Set<string>
) {
  let score = 0;
  const genres = new Set(movie.genres.map((genre) => genre.toLowerCase()));
  const rating = movie.vote_average || 0;
  const votes = movie.vote_count || 0;

  score += Math.min(movie.popularity / 24, 14);
  score += qualityPriorityScore(movie) * 0.34;

  if (rating >= 8 && votes >= 100) score += 14;
  else if (rating >= 7.2 && votes >= 80) score += 9;
  else if (rating < 6 && votes >= 50) score -= 16;
  if (votes < 20) score -= 10;

  for (const genre of intent.genres) {
    if (genres.has(genre.toLowerCase())) score += 12;
  }

  if (titleKeys.has(`${movie.mediaType || "movie"}:${movie.id}`)) score += 20;

  if (taste) {
    for (const genre of taste.likedGenres) {
      if (genres.has(genre.toLowerCase())) score += 4;
    }
    for (const genre of taste.dislikedGenres) {
      if (genres.has(genre.toLowerCase())) score -= 10;
    }
    for (const genre of taste.avoidedGenres) {
      if (genres.has(genre.toLowerCase())) score -= 34;
    }
    if (matchesAvoidPreferences(movie, taste)) {
      score -= 60;
    }
    const key = `${movie.mediaType || "movie"}:${movie.id}`;
    if (taste.dislikedMovieIds.includes(movie.id) || taste.excludeKeys?.includes(key)) {
      score -= 50;
    }
    if (taste.excludeMovieIds.includes(movie.id) || taste.excludeKeys?.includes(key)) {
      score -= 18;
    }
  }

  return score;
}

async function getGenreCandidates(intent: DiscoverIntent) {
  const genreIds = getGenreIdsByNames(intent.genres);
  if (genreIds.length === 0) return [];

  const movieRequests =
    intent.mediaType === "tv"
      ? []
      : [
          discoverMoviesByGenres(genreIds, 1, "popularity.desc", 80),
          discoverMoviesByGenres(genreIds, 1, "vote_average.desc", 180),
          discoverMoviesByGenres(genreIds, 2, "popularity.desc", 80),
        ];
  const tvRequests =
    intent.mediaType === "movie"
      ? []
      : [
          discoverTVByGenres(genreIds, 1, "popularity.desc", 80),
          discoverTVByGenres(genreIds, 1, "vote_average.desc", 120),
          discoverTVByGenres(genreIds, 2, "popularity.desc", 80),
        ];

  const batches = await Promise.all([...movieRequests, ...tvRequests]);
  return batches.flat();
}

async function getReferenceCandidates(intent: DiscoverIntent, query: string) {
  const terms = Array.from(
    new Set([query, ...intent.referenceTitles, ...intent.searchTerms])
  ).slice(0, MAX_SEARCH_TERMS + 2);
  const directMatches = await searchMultiple(terms);
  const referenceMatches = intent.referenceTitles.length
    ? await searchMultiple(intent.referenceTitles)
    : [];

  const recommendationBatches = await Promise.all(
    referenceMatches.slice(0, MAX_REFERENCE_TITLES).map((movie) =>
      movie.mediaType === "tv"
        ? getTVRecommendations(movie.id)
        : getMovieRecommendations(movie.id)
    )
  );

  return {
    directMatches,
    referenceMatches,
    referenceRecommendations: recommendationBatches.flat(),
  };
}

async function enrichCandidatesWithReviews(
  candidates: TMDBMovie[]
): Promise<EnrichedCandidate[]> {
  return Promise.all(
    candidates.map(async (movie) => {
      const reviews =
        movie.mediaType === "tv"
          ? await getTVReviews(movie.id)
          : await getMovieReviews(movie.id);
      return {
        movie,
        reviews: reviews.slice(0, MAX_REVIEWS_PER_CANDIDATE),
      };
    })
  );
}

function truncate(value: string, maxLength: number) {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength).trim()}...`;
}

function buildRankPrompt({
  query,
  intent,
  candidates,
  taste,
}: {
  query: string;
  intent: DiscoverIntent;
  candidates: EnrichedCandidate[];
  taste: UserTastePayload | null;
}) {
  const candidateList = candidates.map(({ movie, reviews }) => ({
    key: `${movie.mediaType || "movie"}:${movie.id}`,
    title: movie.title,
    mediaType: movie.mediaType || "movie",
    overview: truncate(movie.overview || "", 700),
    release_date: movie.release_date,
    vote_average: movie.vote_average,
    vote_count: movie.vote_count,
    popularity: movie.popularity,
    genres: movie.genres,
    reviewSnippets: reviews.map((review) => ({
      rating: review.author_details?.rating ?? null,
      content: truncate(review.content.replace(/\s+/g, " "), 420),
    })),
  }));

  const tasteText = taste
    ? `\nPersonalization signals:
- liked genres: ${taste.likedGenres.slice(0, 8).join(", ") || "none yet"}
- disliked genres: ${taste.dislikedGenres.slice(0, 8).join(", ") || "none yet"}
- avoided genres: ${taste.avoidedGenres.slice(0, 8).join(", ") || "none set"}
- avoided titles/themes: ${taste.avoidTerms.slice(0, 8).join(", ") || "none set"}
Do not recommend exact titles the user already liked, saved, disliked, or watched unless they are exact title matches.`
    : "";

  return `You are FlickBuddy's final recommendation ranker.

User request: "${query}"
Structured intent:
${JSON.stringify(intent, null, 2)}
${tasteText}

Candidate movies and series from TMDB, including synopsis and up to ${MAX_REVIEWS_PER_CANDIDATE} review snippets each:
${JSON.stringify(candidateList, null, 2)}

Choose the best 12-20 candidates that genuinely fit the request. Use the synopsis, genres, ratings, vote counts, popularity, and reviews. Prioritize well-rated movies/series with enough votes when relevance is similar. Penalize candidates that only match a word but not the meaning, very low-rated candidates, and titles with thin vote support unless they are an unusually strong fit. Do not invent titles and only use candidate keys.

Respond with ONLY valid JSON:
{
  "ranked": [
    {
      "key": "movie:123",
      "relevanceScore": 95,
      "reason": "specific reason this fits the user's request",
      "caveat": "optional short caveat, or empty string",
      "highlightedThemes": ["theme1", "theme2", "theme3"]
    }
  ]
}`;
}

async function rankCandidates(
  query: string,
  intent: DiscoverIntent,
  candidates: EnrichedCandidate[],
  taste: UserTastePayload | null
) {
  if (candidates.length === 0) return [];

  try {
    const raw = await generateChatCompletionWithModel(
      buildRankPrompt({ query, intent, candidates, taste }),
      MODEL,
      0.2,
      3600
    );
    const parsed = safeParseJSON<{ ranked: RankedResult[] }>(raw);
    const candidateMap = new Map(
      candidates.map(({ movie }) => [`${movie.mediaType || "movie"}:${movie.id}`, movie])
    );
    const usedKeys = new Set<string>();

    return (parsed.ranked || [])
      .filter((item) => {
        if (!candidateMap.has(item.key) || usedKeys.has(item.key)) return false;
        usedKeys.add(item.key);
        return true;
      })
      .slice(0, 20)
      .map((item, index) => {
        const movie = candidateMap.get(item.key)!;
        const qualityScore = qualityPriorityScore(movie);
        const aiScore = Math.min(Math.max(Math.round(item.relevanceScore), 1), 100);
        return {
          ...decorateMovie(
            movie,
            index,
            intent.explanation,
            item.reason,
            item.highlightedThemes || [],
            item.caveat || undefined
          ),
          relevanceScore: Math.round(aiScore * 0.72 + qualityScore * 0.28),
        };
      })
      .sort(
        (a, b) =>
          b.relevanceScore - a.relevanceScore ||
          ratingTieBreaker(b) - ratingTieBreaker(a)
      );
  } catch {
    return candidates
      .map(({ movie }) => movie)
      .sort((a, b) => ratingTieBreaker(b) - ratingTieBreaker(a))
      .slice(0, 20)
      .map((movie, index) =>
        decorateMovie(
          movie,
          index,
          intent.explanation,
          `Matches ${intent.genres.concat(intent.themeTags).slice(0, 3).join(", ") || "the request"} based on its TMDB synopsis and genres.`
        )
      );
  }
}

function getRateLimitKey(request: NextRequest, userId: string | null) {
  if (userId) return `user:${userId}`;
  const forwardedFor = request.headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  const realIp = request.headers.get("x-real-ip")?.trim();
  return `ip:${forwardedFor || realIp || "anonymous"}`;
}

function checkRateLimit(key: string, limit: number) {
  const now = Date.now();
  const current = rateLimitBuckets.get(key);

  if (!current || current.resetAt <= now) {
    rateLimitBuckets.set(key, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS });
    return { allowed: true, retryAfter: 0 };
  }

  if (current.count >= limit) {
    return {
      allowed: false,
      retryAfter: Math.ceil((current.resetAt - now) / 1000),
    };
  }

  current.count += 1;
  return { allowed: true, retryAfter: 0 };
}

async function getServerTaste(request: NextRequest): Promise<ServerTasteContext> {
  await ensureBackendReady();
  const session = await auth.api.getSession({ headers: request.headers });
  const requestAvoids = getRequestAvoids(request);

  if (!session) {
    return {
      userId: null,
      taste: withAvoids(null, requestAvoids),
    };
  }

  const serverAvoids = getUserAvoidPreferences(session.user.id);
  const avoids = mergeAvoidPreferences(serverAvoids, requestAvoids);

  return {
    userId: session.user.id,
    taste: withAvoids(getUserTastePayload(session.user.id), avoids),
  };
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const query = (searchParams.get("query")?.trim() || "").slice(0, MAX_QUERY_LENGTH);
  const mode = ((searchParams.get("mode") || "smart") as DiscoverMode) || "smart";
  const { userId, taste } = await getServerTaste(request);
  const rateLimit = checkRateLimit(
    getRateLimitKey(request, userId),
    userId ? AUTHENTICATED_RATE_LIMIT : ANONYMOUS_RATE_LIMIT
  );

  if (!rateLimit.allowed) {
    return NextResponse.json(
      { error: "Too many discovery requests. Please wait a moment." },
      {
        status: 429,
        headers: { "Retry-After": String(rateLimit.retryAfter) },
      }
    );
  }

  if (!query) {
    const trending = mode === "series" ? await getTrendingTV() : await getTrendingMovies();
    const results = trending
      .filter((movie) => movie.poster_path && !movie.adult)
      .filter((movie) => {
        const key = `${movie.mediaType || "movie"}:${movie.id}`;
        return (
          !taste?.excludeMovieIds.includes(movie.id) &&
          !taste?.excludeKeys?.includes(key) &&
          !matchesAvoidPreferences(movie, taste)
        );
      })
      .slice(0, 30)
      .map((movie, index) =>
        decorateMovie(movie, index, "Trending in the FlickBuddy discovery pool.")
      );

    return NextResponse.json({
      mode: "trending",
      interpretedQuery: null,
      titleMatches: [],
      discoveryMatches: results,
      results,
    });
  }

  const betaLimit = consumeBetaLimit({
    request,
    userId,
    action: "ai_discover",
  });

  if (!betaLimit.allowed) {
    if (!userId) {
      return NextResponse.json(
        {
          error:
            "Create a free account to keep using AI discovery.",
          code: "AUTH_REQUIRED",
        },
        {
          status: 401,
          headers: betaLimitHeaders(betaLimit),
        }
      );
    }

    return NextResponse.json(
      {
        error:
          "You have reached today's beta AI discovery limit. Try again tomorrow.",
      },
      {
        status: 429,
        headers: betaLimitHeaders(betaLimit),
      }
    );
  }

  const intent = await getDiscoverIntent(query, mode, taste);
  const [rawTitleMatches, genreCandidates, referenceCandidates] = await Promise.all([
    searchMulti(query),
    getGenreCandidates(intent),
    getReferenceCandidates(intent, query),
  ]);

  const titleMatches = rawTitleMatches
    .filter((movie) => movie.poster_path && !movie.adult)
    .filter((movie) => matchesMediaPreference(movie, intent.mediaType))
    .filter((movie) => !matchesAvoidPreferences(movie, taste))
    .map((movie) => ({
      movie,
      score: titleMatchScore(query, movie),
    }))
    .filter((item) => item.score >= 70)
    .sort(
      (a, b) =>
        b.score - a.score || ratingTieBreaker(b.movie) - ratingTieBreaker(a.movie)
    )
    .slice(0, 8)
    .map((item, index) =>
      decorateMovie(
        item.movie,
        index,
        `Strong ${item.movie.mediaType === "tv" ? "series" : "movie"} title match for "${query}".`
      )
    );

  const titleKeys = new Set(
    titleMatches.map((movie) => `${movie.mediaType || "movie"}:${movie.id}`)
  );

  const candidates = dedupeMovies([
    ...genreCandidates,
    ...referenceCandidates.directMatches,
    ...referenceCandidates.referenceRecommendations,
  ])
    .filter((movie) => movie.poster_path && !movie.adult && movie.overview)
    .filter((movie) => matchesMediaPreference(movie, intent.mediaType))
    .filter((movie) => !titleKeys.has(`${movie.mediaType || "movie"}:${movie.id}`))
    .filter((movie) => {
      const key = `${movie.mediaType || "movie"}:${movie.id}`;
      return (
        !taste?.dislikedMovieIds.includes(movie.id) &&
        !taste?.excludeKeys?.includes(key) &&
        !matchesAvoidPreferences(movie, taste)
      );
    })
    .filter((movie) => movie.vote_count >= 20 || movie.popularity >= 12)
    .map((movie) => ({
      movie,
      score: preScoreCandidate(movie, intent, taste, titleKeys),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, MAX_CANDIDATES_FOR_RANKING)
    .map((item) => item.movie);

  const enrichedCandidates = await enrichCandidatesWithReviews(candidates);
  const rankedDiscoveryMatches = await rankCandidates(
    query,
    intent,
    enrichedCandidates,
    taste
  );

  const responseMode =
    titleMatches.length > 0 && rankedDiscoveryMatches.length > 0
      ? "ambiguous"
      : "semantic";

  return NextResponse.json({
    mode: responseMode,
    interpretedQuery: intent,
    titleMatches,
    discoveryMatches: rankedDiscoveryMatches,
    results: [...titleMatches, ...rankedDiscoveryMatches],
  });
}
