# Branch protection on `main`

**Status: not enforced server-side.** Recorded here rather than left implicit,
because an unenforced control that everyone assumes is enforced is worse than a
known gap.

---

## What was attempted

Both GitHub APIs, requiring the `battery` and `integration` checks and blocking
force-push and deletion on the default branch:

```bash
gh api repos/<owner>/ORATORIA-ENGINE/rulesets --method POST ...
gh api repos/<owner>/ORATORIA-ENGINE/branches/main/protection --method PUT ...
```

Both return the same answer:

```
403 Upgrade to GitHub Pro or make this repository public to enable this feature.
```

Branch protection on a **private** repository is a paid feature. This is an
account-plan limit, not a permissions or configuration problem: the token holds
the `repo` scope and the same call succeeds on a public repository.

## The three ways out

| Option                         | Cost                  | What it gives up                                                                                                   |
| ------------------------------ | --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **GitHub Pro**                 | a few dollars a month | Nothing. Full server-side protection on a private repo.                                                            |
| **Make the repository public** | free                  | The corpus work, consent policy and unpublished evaluation protocol become readable before the thesis is defended. |
| **Local hook only**            | free                  | Server-side enforcement. Runs on one machine and is bypassable with `--no-verify`.                                 |

The recommendation is Pro. The repository holds an unpublished evaluation
protocol and a consent policy for a study that has not run; §14.4 freezes the
research questions and the held-out set _before_ final analysis, and publishing
the protocol early makes that freeze unverifiable to a reader who could have
seen it change.

## What is in place meanwhile

CI runs on every push and every pull request, and both jobs must be green:

- `battery` — ruff, mypy `--strict`, six import contracts, the full suite with
  warnings as errors, and an 88% branch-coverage floor on domain and
  application
- `integration` — the migration round-trip and the 16 tests against
  PostgreSQL, Redis and MinIO

A local pre-push hook runs the `battery` gates before a push to `main`:

```bash
./scripts/install-hooks.sh
```

It is opt-in, it skips the integration suite (a hook that fails because Docker
is not running teaches people to pass `--no-verify`), and it is a reminder with
teeth rather than enforcement. **CI is the real gate; nothing currently stops a
push that ignores it.**

## When Pro is enabled

```bash
gh api repos/<owner>/ORATORIA-ENGINE/rulesets --method POST --input - <<'JSON'
{
  "name": "main protection",
  "target": "branch",
  "enforcement": "active",
  "conditions": { "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] } },
  "rules": [
    { "type": "deletion" },
    { "type": "non_fast_forward" },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "required_status_checks": [
          { "context": "battery" },
          { "context": "integration" }
        ]
      }
    }
  ]
}
JSON
```

Then update the Status line at the top of this file.
