# Contributing

Thanks for improving FlickBuddy. Keep changes focused, documented, and easy to
review.

## Local Setup

1. Install dependencies:

```bash
npm install
```

2. Copy the environment template:

```bash
cp .env.local.example .env.local
```

3. Fill in at least `TMDB_API_KEY`, `BETTER_AUTH_SECRET`, `AI_PROVIDER`, and the
API key for your chosen AI provider.

4. Start the app:

```bash
npm run dev
```

## Pull Requests

- Keep provider-specific AI logic inside `src/lib/ai.ts`.
- Do not expose API keys through `NEXT_PUBLIC_*` variables.
- Add or update docs when adding environment variables.
- Run `npm run build` before opening a PR when your change touches runtime code.
- Avoid committing generated databases, local env files, build output, or OS files.

## Adding an AI Provider

Add a provider implementation to the provider registry in `src/lib/ai.ts`. A
provider should accept normalized chat messages and return plain text. Provider
keys must come from server-side environment variables or a server-only encrypted
credential store, never from client code.
