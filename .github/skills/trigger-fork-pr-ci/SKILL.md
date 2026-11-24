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
- `cleanup <pr_number>`: Resolve `<owner>/<repo>` and `<project>` (Steps 1-2), skip Steps 3-5, check `azure-devops-pipelines_build` (`action: "list"`, `project: "<project>"`, `branchName: "refs/heads/pr-<pr_number>"`, `top: 20`) across all pipeline definitions, and delete `refs/heads/pr-<pr_number>` only after confirming zero builds on that branch are `NotStarted` or `InProgress`.

## Execution Steps

1. **Inspect PR metadata and verify modified paths**
   - Resolve the target repository (`<owner>/<repo>`) from the current workspace (`gh repo view --json nameWithOwner --jq .nameWithOwner`).
   - Query the pull request:
     ```bash
     gh pr view <pr_number> --repo <owner>/<repo> --json number,state,headRefOid,statusCheckRollup
     ```
   - Require `state == "OPEN"` and record `HEAD_SHA` (`headRefOid`).
   - Inspect changed paths:
     ```bash
     gh pr diff <pr_number> --repo <owner>/<repo> --name-only
     ```
   - **Pre-trigger confirmation**: If the diff touches CI, build, container, or deployment inputs (`.azure-pipelines/`, `.github/`, `cmake/`, `docker/`, `test/deploy/`, `tools/`, any root or nested `**/CMakeLists.txt`, `*.cmake`, `pyproject.toml`, `setup.py`, `python/requirements*.txt`, or `*.sh` / `*.sh.in`), review those exact diffs first and ask the user for explicit confirmation before creating a branch or queueing a build.

2. **Resolve the Azure DevOps project and pipeline ID dynamically**
   - Load the Azure DevOps pipeline tools via `tool_search_tool` if deferred (`azure-devops-pipelines_definition`, `azure-devops-pipelines_build`, `azure-devops-pipelines_write`).
   - Extract the Azure DevOps `<project>` identifier from an existing pipeline `detailsUrl` in `statusCheckRollup` (or from CI badge URLs in `README.md` if no check run exists yet).
   - Look up `<pipeline_id>` via `azure-devops-pipelines_definition` (`action: "list"`, `project: "<project>"`, `name: "<pipeline_name>"`, `repositoryId: "<owner>/<repo>"`, `repositoryType: "GitHub"`).

3. **Create or update `refs/heads/pr-<pr_number>` at `HEAD_SHA`**
   - Point `refs/heads/pr-<pr_number>` directly to `HEAD_SHA` without creating a merge commit or rebase so Check Runs attach to the PR head:
     ```bash
     gh api -X POST repos/<owner>/<repo>/git/refs -f ref="refs/heads/pr-<pr_number>" -f sha="<HEAD_SHA>" \
       || gh api -X PATCH repos/<owner>/<repo>/git/refs/heads/pr-<pr_number> -f sha="<HEAD_SHA>" -F force=true
     ```

4. **Queue the pipeline and cancel superseded merge runs**
   - Queue the build via `azure-devops-pipelines_write`:
     - `action`: `"run_pipeline"`
     - `project`: `"<project>"`
     - `pipelineId`: `<pipeline_id>`
     - `resources`: `{ "repositories": { "self": { "refName": "refs/heads/pr-<pr_number>", "version": "<HEAD_SHA>" } } }`
   - List recent runs on `refs/pull/<pr_number>/merge` via `azure-devops-pipelines_build` (`action: "list"`, `project: "<project>"`, `definitions: [<pipeline_id>]`, `branchName: "refs/pull/<pr_number>/merge"`, `top: 5`).
   - Cancel any `NotStarted` or `InProgress` merge run via `azure-devops-pipelines_write` (`action: "update_build_stage"`, `project: "<project>"`, `buildId: <old_build_id>`, `stageName: "__default"`, `status: "Cancel"`).

5. **Verify GitHub Check Runs**
   - Wait ~10 seconds and re-check `gh pr view <pr_number> --repo <owner>/<repo> --json statusCheckRollup` to confirm the check runs now reference the new `buildId`.

6. **Clean up `pr-<pr_number>` after completion**
   - Keep `refs/heads/pr-<pr_number>` until `<buildId>` and all other builds on `refs/heads/pr-<pr_number>` across every pipeline definition reach `Completed`, so concurrent pipeline runs on the same PR head can still fetch the ref when queued matrix jobs start.
   - If `save_session_automation` is available and `get_session_automation` shows no active conflicting schedule, register a 15-minute check (`interval: "minutes"`, `every_minutes: 15`) whose prompt:
     1. Queries `azure-devops-pipelines_build` for both `buildIds: [<buildId>]` and `branchName: "refs/heads/pr-<pr_number>"` (`action: "list"`, `project: "<project>"`, `top: 20`, omitting `definitions` so all pipelines are checked).
     2. Waits if `<buildId>` or any build on `refs/heads/pr-<pr_number>` has `status` in `["NotStarted", "InProgress"]`.
     3. Once `<buildId>` is `Completed` and no unfinished builds remain on `refs/heads/pr-<pr_number>` across any pipeline definition, verifies `refs/heads/pr-<pr_number>` still points to `<HEAD_SHA>`, deletes it with `gh api -X DELETE repos/<owner>/<repo>/git/refs/heads/pr-<pr_number>`, clears the schedule via `save_session_automation(clear=true)`, and reports the final build result.
   - Otherwise — if `save_session_automation` is unavailable or `get_session_automation` already has a conflicting schedule (do **not** overwrite an existing schedule) — remind the user to run `/trigger-fork-pr-ci cleanup <pr_number>` once all builds on `refs/heads/pr-<pr_number>` finish.
