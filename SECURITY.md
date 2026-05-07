# Security Policy

## Supported Versions

Security fixes target the current `main` branch.

## Reporting a Vulnerability

Do not open public issues for suspected vulnerabilities involving secrets,
authentication, authorization, API-key handling, or user data. Use a private
GitHub security advisory if available, or contact the maintainers privately.

## API Keys

FlickBuddy routes all AI and TMDB calls through server-side code. Provider keys
must remain in `.env.local`, deployment secrets, or a server-only encrypted
credential store.

Never put AI provider keys, TMDB keys, database URLs, OAuth secrets, or auth
secrets in `NEXT_PUBLIC_*` variables. Anything with that prefix is visible in
the browser bundle.

Before publishing a fork or release, run a secret scan and rotate any key that
was ever committed to Git history.
