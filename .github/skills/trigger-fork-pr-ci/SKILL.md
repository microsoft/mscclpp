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
- `cleanup <pr_number>`: Skip Steps 1-5 and delete `refs/heads/pr-<pr_number>` after confirming no build on that branch is `NotStarted` or `InProgress`.

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
   - **Pre-trigger confirmation**: If the diff touches CI, build, or deployment files (`.azure-pipelines/`, `.github/`, `test/deploy/`, `tools/`, `CMakeLists.txt`, or `setup.py`), review those changes first and ask the user for explicit confirmation before creating a branch or queueing a build.

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
   - Keep `refs/heads/pr-<pr_number>` until the build reaches `Completed` so all matrix jobs can fetch the ref when starting.
   - If `save_session_automation` is available and `get_session_automation` shows no conflicting schedule, register a 15-minute check (`interval: "minutes"`, `every_minutes: 15`) whose prompt:
     1. Queries `azure-devops-pipelines_build` (`action: "list"`, `project: "<project>"`, `buildIds: [<buildId>]`).
     2. Waits if `status` is `NotStarted` or `InProgress`.
     3. Once `status` is `Completed`, verifies `refs/heads/pr-<pr_number>` still points to `<HEAD_SHA>`, deletes it with `gh api -X DELETE repos/<owner>/<repo>/git/refs/heads/pr-<pr_number>`, clears the schedule via `save_session_automation(clear=true)`, and reports the final build result.
   - If `save_session_automation` is unavailable, remind the user to run `/trigger-fork-pr-ci cleanup <pr_number>` once the build finishes.
