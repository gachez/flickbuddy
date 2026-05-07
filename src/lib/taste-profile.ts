import "server-only";

import { generateChatCompletionWithModel, getDefaultAIModel } from "@/lib/ai";
import { db, ensureAppTables } from "@/lib/db";
import { getUserMoviesByAction } from "@/lib/user-movies";
import { Movie } from "@/types/movie";

const MODEL = getDefaultAIModel();
const PROFILE_TTL_MS = 1000 * 60 * 60 * 6;

export interface AITasteProfile {
  summary: string;
  likedThemes: string[];
  dislikedThemes: string[];
  referenceTitles: string[];
  recommendedQueries: string[];
  genreWeights: Record<string, number>;
  updatedAt: string;
}

interface TasteProfileRow {
  user_id: string;
  summary: string;
  liked_themes: string;
  disliked_themes: string;
  reference_titles: string;
  recommended_queries: string;
  genre_weights: string;
  updated_at: string;
}

ensureAppTables();

function safeParse<T>(value: string, fallback: T): T {
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

function stringList(value: unknown, maxItems: number, maxLength = 80) {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim().slice(0, maxLength))
    .filter(Boolean)
    .slice(0, maxItems);
}

function genreWeights(value: unknown) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};

  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .map(([genre, weight]) => [genre.trim().slice(0, 80), Number(weight)])
      .filter(([genre, weight]) => genre && Number.isFinite(weight))
      .map(([genre, weight]) => [
        genre,
        Math.max(-10, Math.min(10, Number(weight))),
      ])
      .slice(0, 24)
  );
}

function rowToProfile(row: TasteProfileRow): AITasteProfile {
  return {
    summary: row.summary,
    likedThemes: safeParse<string[]>(row.liked_themes, []),
    dislikedThemes: safeParse<string[]>(row.disliked_themes, []),
    referenceTitles: safeParse<string[]>(row.reference_titles, []),
    recommendedQueries: safeParse<string[]>(row.recommended_queries, []),
    genreWeights: safeParse<Record<string, number>>(row.genre_weights, {}),
    updatedAt: row.updated_at,
  };
}

export function getStoredTasteProfile(userId: string) {
  const row = db
    .prepare(
      `
        SELECT user_id, summary, liked_themes, disliked_themes, reference_titles,
          recommended_queries, genre_weights, updated_at
        FROM user_taste_profile
        WHERE user_id = ?
      `
    )
    .get(userId) as TasteProfileRow | undefined;

  return row ? rowToProfile(row) : null;
}

export function isTasteProfileFresh(profile: AITasteProfile | null) {
  if (!profile) return false;
  const updated = new Date(profile.updatedAt).getTime();
  return Number.isFinite(updated) && Date.now() - updated < PROFILE_TTL_MS;
}

function compactMovie(movie: Movie) {
  return {
    title: movie.title,
    type: movie.mediaType === "tv" ? "series" : "movie",
    year: movie.release_date ? new Date(movie.release_date).getFullYear() : null,
    genres: movie.genres?.slice(0, 5) || [],
    overview: movie.overview?.slice(0, 320) || "",
  };
}

function buildProfilePrompt({
  liked,
  saved,
  disliked,
}: {
  liked: Movie[];
  saved: Movie[];
  disliked: Movie[];
}) {
  return `You are FlickBuddy's taste profiler. Analyze this user's movie and series behavior and produce grounded discovery guidance for TMDB searches.

Liked:
${JSON.stringify(liked.slice(0, 24).map(compactMovie), null, 2)}

Saved:
${JSON.stringify(saved.slice(0, 24).map(compactMovie), null, 2)}

Passed or disliked:
${JSON.stringify(disliked.slice(0, 18).map(compactMovie), null, 2)}

Return ONLY valid JSON:
{
  "summary": "one sentence taste profile",
  "likedThemes": ["theme"],
  "dislikedThemes": ["theme"],
  "referenceTitles": ["existing movie or series title useful as a discovery reference"],
  "recommendedQueries": ["short TMDB search phrase or title-like query"],
  "genreWeights": { "Drama": 4, "Comedy": -2 }
}

Rules:
- Use only themes supported by the behavior above.
- recommendedQueries should be practical TMDB search terms, not long prose.
- Include 4-8 likedThemes, 0-5 dislikedThemes, 3-8 referenceTitles, 4-10 recommendedQueries.
- genreWeights range from -10 to 10.`;
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
    if (!match) throw new Error("AI taste profile response was not JSON");
    return JSON.parse(match[0]) as Record<string, unknown>;
  }
}

export async function refreshTasteProfile(userId: string) {
  const liked = getUserMoviesByAction(userId, "like");
  const saved = getUserMoviesByAction(userId, "save");
  const disliked = getUserMoviesByAction(userId, "dislike");

  if (liked.length + saved.length < 2) {
    return getStoredTasteProfile(userId);
  }

  const raw = await generateChatCompletionWithModel(
    buildProfilePrompt({ liked, saved, disliked }),
    MODEL,
    0.2,
    1800
  );
  const parsed = parseModelJSON(raw);

  const profile = {
    summary:
      typeof parsed.summary === "string"
        ? parsed.summary.trim().slice(0, 280)
        : "A developing FlickBuddy taste profile.",
    likedThemes: stringList(parsed.likedThemes, 8),
    dislikedThemes: stringList(parsed.dislikedThemes, 5),
    referenceTitles: stringList(parsed.referenceTitles, 8, 120),
    recommendedQueries: stringList(parsed.recommendedQueries, 10, 120),
    genreWeights: genreWeights(parsed.genreWeights),
  };

  db.prepare(
    `
      INSERT INTO user_taste_profile (
        user_id,
        summary,
        liked_themes,
        disliked_themes,
        reference_titles,
        recommended_queries,
        genre_weights,
        updated_at
      )
      VALUES (
        @userId,
        @summary,
        @likedThemes,
        @dislikedThemes,
        @referenceTitles,
        @recommendedQueries,
        @genreWeights,
        datetime('now')
      )
      ON CONFLICT(user_id)
      DO UPDATE SET
        summary = excluded.summary,
        liked_themes = excluded.liked_themes,
        disliked_themes = excluded.disliked_themes,
        reference_titles = excluded.reference_titles,
        recommended_queries = excluded.recommended_queries,
        genre_weights = excluded.genre_weights,
        updated_at = datetime('now')
    `
  ).run({
    userId,
    summary: profile.summary,
    likedThemes: JSON.stringify(profile.likedThemes),
    dislikedThemes: JSON.stringify(profile.dislikedThemes),
    referenceTitles: JSON.stringify(profile.referenceTitles),
    recommendedQueries: JSON.stringify(profile.recommendedQueries),
    genreWeights: JSON.stringify(profile.genreWeights),
  });

  return getStoredTasteProfile(userId);
}
