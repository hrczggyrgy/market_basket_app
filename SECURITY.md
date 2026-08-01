# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.3.x   | ✅ Yes              |
| 1.0.x   | ⚠️ Critical fixes only |
| < 1.0   | ❌ No               |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, report them privately:

1. Go to the repository **Security** tab → **Report a vulnerability** (GitHub private advisory).
2. Or email the maintainer directly (see the repository profile).

You can expect an acknowledgement within **48 hours** and a resolution timeline within **7 days** for critical issues.

## Scope

- Secrets / credentials accidentally committed to the repository
- Dependency vulnerabilities with a published CVE
- Data injection or path traversal via the CSV upload endpoint

## Out of Scope

- Vulnerabilities in third-party libraries that have no available patch
- Issues reproducible only with unrealistic data volumes or attack scenarios

## Disclosure Policy

We follow a **coordinated disclosure** model. Once a fix is released, the advisory will be made public with full credit to the reporter.
