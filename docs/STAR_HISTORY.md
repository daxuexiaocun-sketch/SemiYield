# GitHub star-history maintenance

The previous workflow ran `python scripts/update_star_history.py` without installing this `src`
layout package. On a clean interpreter that raises `ModuleNotFoundError: semiyield`.
The workflow now installs uv, synchronizes the locked project, and runs the script through uv.

The repository identifier comes from `GITHUB_REPOSITORY`; no owner is hard-coded. Scheduled/manual
runs and changes to the star script, workflow or dependency definitions update the default branch.
A concurrency group serializes updates. Assets are staged before comparing the staged diff, so a
first run that creates untracked files is committed as well as updates to existing files.
Only the two star-history assets are committed. This workflow needs repository contents write
permission and a branch policy that permits its bot commit; push failures remain visible in Actions.

The script fetches timestamped current stargazers with pagination. The chart's x axis uses elapsed
calendar days, including gaps without new stars. Zero stars produces a zero line; missing data shows
a pending message. Network/API failures preserve existing assets and fail the job with diagnostics.
An authorization error is not interpreted as a zero-star repository.

GitHub documents access restrictions for stargazer listing. Check token/repository access and rate
limits if the workflow receives 401/403/404; do not grant broad tokens merely to hide a failed job.
See [GitHub's starring API documentation](https://docs.github.com/en/rest/activity/starring).

The curve reconstructs timestamps of **currently remaining stargazers**. It cannot recover users
who removed their stars, and therefore is not a complete historical net-star ledger.

Local read-only entry-point check:

```bash
uv sync --locked
uv run python -I scripts/update_star_history.py --help
```

After publishing these changes, inspect a manual Actions run to confirm actual API and branch
permissions. Local tests mock responses and do not prove remote write access.
