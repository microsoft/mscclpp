---
name: trigger-fork-pr-ci
description: Trigger Azure Pipelines CI (default mscclpp-ut) for a fork pull request by creating a temporary pr-<N> branch at the PR head commit, queueing the pipeline run, canceling superseded PR merge runs, and removing the pr-<N> branch after completion. Use when asked to trigger mscclpp-ut or other CI pipelines for a fork or blocked pull request.
argument-hint: "<pr_number> [pipeline_name]"
---

# Trigger Fork PR CI

Queue an Azure Pipelines validation run on a temporary upstream branch `pr-<pr_number>` pointing to the pull request's exact head commit so GitHub Check Runs report directly back to the pull request, then clean up the temporary branch once the run completes.

## Inputs

- `pr_number`: Pull request number (infer from the active session's tracked PR if omitted).
- `pipeline_name`: Optional pipeline definition name. Defaults to `mscclpp-ut`.
- `cleanup <pr_number>`: Allowed for open, closed, or merged PRs. Resolve `<owner>/<repo>` and `<project>` (using Step 2 without requiring the PR to be open), skip Steps 1 and 3-5, exhaustively list all builds on `refs/heads/pr-<pr_number>` across every pipeline definition (`action: "list"`, `project: "<project>"`, `branchName: "refs/heads/pr-<pr_number>"`, `queryOrder: "QueueTimeDescending"`, paging `continuationToken` until `null`, omitting `definitions`, `top`, and `repositoryId`), and delete `refs/heads/pr-<pr_number>` only if the complete listing succeeds and every returned build has `status == "Completed"` (leave the branch intact if listing is incomplete or any build is unfinished; treat `404` on delete as already clean).

## Execution Steps

1. **Record immutable PR SHAs and verify modified paths (trigger mode only)**
   - Resolve the target repository (`<owner>/<repo>`) from the current workspace (`gh repo view --json nameWithOwner --jq .nameWithOwner`).
   - Query the pull request in a single call:
     ```bash
     gh api repos/<owner>/<repo>/pulls/<pr_number> --jq '{number, state, base_sha: .base.sha, head_sha: .head.sha}'
     ```
   - Require `state == "open"` (apply this open-state check only when triggering CI, never during `cleanup <pr_number>`) and record the immutable 40-character `BASE_SHA` and `HEAD_SHA`.
   - Obtain the complete, untruncated file list between `BASE_SHA` and `HEAD_SHA` (never use `gh pr diff`, which reads the mutable PR head):
     - In a local Git checkout, fetch both immutable commits and diff them directly:
       ```bash
       git fetch --no-tags origin <BASE_SHA> <HEAD_SHA>
       git --no-pager diff --name-only <BASE_SHA>...<HEAD_SHA>
       ```
     - If no local Git checkout is available, query `gh api "repos/<owner>/<repo>/compare/<BASE_SHA>...<HEAD_SHA>"`; because this REST endpoint caps `files` at 300 entries, if `files` is missing, incomplete, or reaches 300 entries, require explicit user confirmation (or fetch into a local checkout) before proceeding.
   - **Pre-trigger confirmation (bound to `HEAD_SHA`)**: If any changed path touches CI, build, container, or deployment inputs (`.azure-pipelines/`, `.github/`, `cmake/`, `docker/`, `test/deploy/`, `tools/`, any root or nested `**/CMakeLists.txt`, `*.cmake`, `**/Makefile`, `pyproject.toml`, `setup.py`, `setup.cfg`, `python/requirements*.txt`, or `*.sh` / `*.sh.in`), inspect the exact diff for `<BASE_SHA>...<HEAD_SHA>` first and ask the user for explicit confirmation bound to `HEAD_SHA` before creating a branch or queueing a build. Never reuse confirmation granted for a different `HEAD_SHA`.

2. **Resolve the Azure DevOps project and pipeline ID dynamically**
   - Load the Azure DevOps pipeline tools via `tool_search_tool` if deferred (`azure-devops-pipelines_definition`, `azure-devops-pipelines_build`, `azure-devops-pipelines_write`).
   - Query `gh pr view <pr_number> --repo <owner>/<repo> --json statusCheckRollup` (works for open, closed, or merged PRs) and extract the Azure DevOps `<project>` identifier from an existing pipeline `detailsUrl` (or fall back to CI badge URLs in `README.md` / `azure-devops-core_list_projects`).
   - For trigger requests, look up `<pipeline_id>` via `azure-devops-pipelines_definition` (`action: "list"`, `project: "<project>"`, `name: "<pipeline_name>"`, `repositoryId: "<owner>/<repo>"`, `repositoryType: "GitHub"`).

3. **Create or update `refs/heads/pr-<pr_number>` at `HEAD_SHA`**
   - Inspect `gh api repos/<owner>/<repo>/git/ref/heads/pr-<pr_number>` first:
     - If `404` (branch does not exist), create it at `HEAD_SHA` without creating a merge commit or rebase:
       ```bash
       gh api -X POST repos/<owner>/<repo>/git/refs -f ref="refs/heads/pr-<pr_number>" -f sha="<HEAD_SHA>"
       ```
     - If it already points to `HEAD_SHA`, keep it as-is.
     - If it points to a different SHA, exhaustively list builds on `refs/heads/pr-<pr_number>` (`queryOrder: "QueueTimeDescending"`, paging `continuationToken` until `null` without `definitions`, `top`, or `repositoryId`); leave the branch untouched if listing fails, cancel unfinished superseded builds for `<pipeline_id>`, ask the user whether to cancel or wait if another pipeline definition still has an unfinished build (`status != "Completed"`) on the older SHA, and then force-update:
       ```bash
       gh api -X PATCH repos/<owner>/<repo>/git/refs/heads/pr-<pr_number> -f sha="<HEAD_SHA>" -F force=true
       ```

4. **Queue the pipeline and cancel superseded merge runs**
   - Queue the build via `azure-devops-pipelines_write`:
     - `action`: `"run_pipeline"`
     - `project`: `"<project>"`
     - `pipelineId`: `<pipeline_id>`
     - `resources`: `{ "repositories": { "self": { "refName": "refs/heads/pr-<pr_number>", "version": "<HEAD_SHA>" } } }`
   - Exhaustively list runs on `refs/pull/<pr_number>/merge` for `<pipeline_id>` via `azure-devops-pipelines_build` (`action: "list"`, `project: "<project>"`, `definitions: [<pipeline_id>]`, `branchName: "refs/pull/<pr_number>/merge"`, `queryOrder: "QueueTimeDescending"`, paging `continuationToken` until `null` without `repositoryId`).
   - Cancel any merge run with `status` in `["NotStarted", "InProgress"]` via `azure-devops-pipelines_write` (`action: "update_build_stage"`, `project: "<project>"`, `buildId: <old_build_id>`, `stageName: "__default"`, `status: "Cancel"`).

5. **Verify GitHub Check Runs**
   - Wait ~10 seconds and re-check `gh pr view <pr_number> --repo <owner>/<repo> --json statusCheckRollup` to confirm the check runs now reference the new `buildId`.

6. **Clean up `pr-<pr_number>` after completion**
   - Keep `refs/heads/pr-<pr_number>` until `<buildId>` and all other builds on `refs/heads/pr-<pr_number>` across every pipeline definition have `status == "Completed"`, so concurrent pipeline runs on the same PR head can still fetch the ref when queued matrix jobs start.
   - If `save_session_automation` is available and `get_session_automation` shows no active conflicting schedule, register a 15-minute check (`interval: "minutes"`, `every_minutes: 15`) whose prompt:
     1. Queries the tracked build separately via `azure-devops-pipelines_build` (`action: "list"`, `project: "<project>"`, `buildIds: [<buildId>]`). If this call fails or the build's `status != "Completed"`, waits for the next check.
     2. Once `<buildId>` has `status == "Completed"`, exhaustively lists all builds on `refs/heads/pr-<pr_number>` in separate paged calls via `azure-devops-pipelines_build` (`action: "list"`, `project: "<project>"`, `branchName: "refs/heads/pr-<pr_number>"`, `queryOrder: "QueueTimeDescending"`, following `continuationToken` until `null` and omitting `buildIds`, `definitions`, `top`, and `repositoryId`). If any page fails or any returned build has `status != "Completed"`, leaves `refs/heads/pr-<pr_number>` intact and waits for the next check.
     3. Once the complete branch-wide listing confirms zero unfinished builds remain, checks `gh api repos/<owner>/<repo>/git/ref/heads/pr-<pr_number>`: if it still points to `<HEAD_SHA>`, deletes it via `gh api -X DELETE repos/<owner>/<repo>/git/refs/heads/pr-<pr_number>`; if it already returned `404` or moved to a newer SHA, leaves it untouched. Finally, clears the schedule via `save_session_automation(clear=true)` and reports the final result of `<buildId>` (noting if a newer SHA now owns `pr-<pr_number>`).
   - Otherwise — if `save_session_automation` is unavailable or `get_session_automation` already has a conflicting schedule (do **not** overwrite an existing schedule) — remind the user to run `/trigger-fork-pr-ci cleanup <pr_number>` once all builds on `refs/heads/pr-<pr_number>` finish.
