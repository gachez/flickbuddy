import { NextRequest, NextResponse } from "next/server";
import { getRequestSession } from "@/lib/api-session";
import { betaLimitHeaders, consumeBetaLimit } from "@/lib/beta-limits";
import { generateChatCompletionWithModel, getDefaultAIModel } from "@/lib/ai";
import { searchMultiple, TMDBMovie } from "@/lib/tmdb";
import { getPostHogClient } from "@/lib/posthog-server";

const MODEL = getDefaultAIModel();

// --- PROMPT BUILDERS ---

function buildIntentPrompt(query: string): string {
  return `You are FlickBuddy, an expert movie recommendation AI. Your job is to understand what movies a user wants to watch.

User query: "${query}"

Extract search parameters to find matching movies on TMDB. Think about:
- Specific films, franchises, or directors they mention
- The genre, tone, mood, or era they want
- Similar well-known films that would be good search terms
- If it's a follow-up to the conversation, account for previous context

Respond with ONLY valid JSON (no markdown, no code blocks):
{
  "searchTerms": ["term1", "term2", "term3"],
  "explanation": "brief explanation of what they want"
}

Generate 2-4 diverse search terms that will surface relevant results on TMDB.`;
}

function buildRankPrompt(query: string, movies: TMDBMovie[]): string {
  const movieList = movies.map((m) => ({
    id: m.id,
    title: m.title,
    overview: m.overview,
    release_date: m.release_date,
    vote_average: m.vote_average,
    genres: m.genres,
  }));

  return `You are FlickBuddy, an expert movie curator. The user asked: "${query}"

Here are candidate movies from TMDB:
${JSON.stringify(movieList, null, 2)}

Select the 8-12 best matching movies and rank them by relevance to the user's request. For each:
- relevanceScore: integer 0–100 (how well it matches what they want)
- matchReason: 1–2 sentences explaining why this film fits their request
- highlightedThemes: array of 3–5 keywords/themes relevant to the user's request

Only include movies with relevanceScore above 50. Sort by relevanceScore descending.

Respond with ONLY valid JSON (no markdown, no code blocks):
{
  "recommendations": [
    {
      "id": 123,
      "relevanceScore": 95,
      "matchReason": "...",
      "highlightedThemes": ["theme1", "theme2", "theme3"]
    }
  ]
}`;
}

// --- JSON PARSING WITH FALLBACK ---

function safeParseJSON<T>(raw: string): T {
  // Strip markdown code fences if present
  const cleaned = raw
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  try {
    return JSON.parse(cleaned) as T;
  } catch {
    // Try to extract JSON object from response
    const match = cleaned.match(/\{[\s\S]*\}/);
    if (match) {
      return JSON.parse(match[0]) as T;
    }
    throw new Error(`Failed to parse AI response as JSON: ${raw.slice(0, 200)}`);
  }
}

// --- ROUTE HANDLER ---

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get("query")?.trim() || "";

  if (!query) {
    return NextResponse.json({ error: "query is required" }, { status: 400 });
  }

  const session = await getRequestSession(request);
  const betaLimit = consumeBetaLimit({
    request,
    userId: session?.user.id,
    action: "legacy_recommendations",
  });

  const distinctId = session?.user.id ?? `anon_${request.headers.get("x-flickbuddy-client-id") ?? "unknown"}`;
  getPostHogClient().capture({
    distinctId,
    event: "ai_recommendations_requested",
    properties: {
      query,
      authenticated: !!session?.user.id,
    },
  });

  if (!betaLimit.allowed) {
    return NextResponse.json(
      {
        error:
          "You have reached today's beta AI recommendation limit. Try discovery or come back tomorrow.",
      },
      {
        status: 429,
        headers: betaLimitHeaders(betaLimit),
      }
    );
  }

  try {
    // --- Pass 1: Extract search intent ---
    const intentRaw = await generateChatCompletionWithModel(
      buildIntentPrompt(query),
      MODEL,
    );

    let intent: { searchTerms: string[]; explanation: string };
    try {
      intent = safeParseJSON<{ searchTerms: string[]; explanation: string }>(
        intentRaw
      );
    } catch {
      // If AI response fails to parse, fall back to raw query as search term
      intent = { searchTerms: [query], explanation: query };
    }

    // Ensure we always have at least one search term
    if (!intent.searchTerms?.length) {
      intent.searchTerms = [query];
    }

    // --- TMDB search ---
    const movies = await searchMultiple(intent.searchTerms);

    if (movies.length === 0) {
      return NextResponse.json({ results: [] });
    }

    // --- Pass 2: Rank and enrich ---
    const rankRaw = await generateChatCompletionWithModel(
      buildRankPrompt(query, movies),
      MODEL,
      0.3,
      2000
    );

    let ranked: {
      recommendations: {
        id: number;
        relevanceScore: number;
        matchReason: string;
        highlightedThemes: string[];
      }[];
    };

    try {
      ranked = safeParseJSON<{
        recommendations: {
          id: number;
          relevanceScore: number;
          matchReason: string;
          highlightedThemes: string[];
        }[];
      }>(rankRaw);
    } catch {
      // Fall back: return top movies from TMDB with no AI enrichment
      ranked = {
        recommendations: movies.slice(0, 10).map((m, i) => ({
          id: m.id,
          relevanceScore: 80 - i * 5,
          matchReason: m.overview?.slice(0, 120) || "",
          highlightedThemes: m.genres.slice(0, 3),
        })),
      };
    }

    // --- Merge TMDB data with AI enrichment ---
    const movieMap = new Map(movies.map((m) => [m.id, m]));

    const results = ranked.recommendations
      .filter((r) => movieMap.has(r.id))
      .map((r) => {
        const movie = movieMap.get(r.id)!;
        return {
          ...movie,
          relevanceScore: r.relevanceScore,
          matchReason: r.matchReason,
          highlightedThemes: r.highlightedThemes,
          hasContentAnalysis: true,
        };
      });

    return NextResponse.json({ results });
  } catch (error) {
    console.error("Recommendations error:", error);
    return NextResponse.json(
      { error: "Failed to generate recommendations" },
      { status: 500 }
    );
  }
}
