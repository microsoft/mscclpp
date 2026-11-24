---
name: trigger-fork-pr-ci
description: Trigger Azure Pipelines CI (default mscclpp-ut) for a fork pull request by creating an isolated temporary pr-<N>-<sha>-<nonce> branch at the PR head commit, queueing the pipeline run, canceling active PR merge runs, and removing the temporary branch after completion. Use when asked to trigger mscclpp-ut or other CI pipelines for a fork pull request.
argument-hint: "<pr_number> [pipeline_name]"
---

# Trigger Fork PR CI

Queue an Azure Pipelines validation run for a fork pull request on an isolated, single-use upstream branch `pr-<pr_number>-<short_sha>-<nonce>` pointing to the PR's exact head commit so GitHub Check Runs report directly back to that commit, then clean up the temporary branch once the run completes.

## Prerequisites and User Responsibility

- **Fork PRs only**: Only run this workflow on cross-repository (fork) pull requests. Same-repository pull requests should use standard PR triggers or `/azp run`.
- **Local Git checkout required**: Run from a local Git checkout of the target repository; stop and ask the user to open a repository workspace if none is available.
- **Reviewer responsibility**: The user invoking this skill is responsible for reviewing the fork pull request's changes at `HEAD_SHA` before running upstream CI, and for running manual cleanup (`/trigger-fork-pr-ci cleanup <pr_number>`) if automated cleanup cannot run or encounters an ambiguous state.

## Inputs

- `pr_number`: Pull request number (infer from the active session's tracked PR if omitted).
- `pipeline_name`: Optional pipeline definition name. Defaults to `mscclpp-ut`.
- `cleanup <pr_number>`: Allowed for open, closed, or merged PRs. Resolve `<owner>/<repo>` and `<project>` (Step 2, without requiring the PR to be open), list refs via `gh api --paginate "repos/<owner>/<repo>/git/matching-refs/heads/pr-<pr_number>"`, and filter strictly to exact `refs/heads/pr-<pr_number>` or prefix `refs/heads/pr-<pr_number>-`. For each matching `<branch_ref>`, exhaustively list all builds across every pipeline definition (`action: "list"`, `project: "<project>"`, `branchName: "<branch_ref>"`, `queryOrder: "QueueTimeDescending"`, paging `continuationToken` until `null` while omitting `definitions`, `top`, and `repositoryId`), and delete `<branch_ref>` only when the complete listing succeeds, returns at least one build, and every returned build has `status == "Completed"` (report any zero-build or ambiguous refs to the user to confirm deletion).

## Execution Steps

1. **Verify fork PR metadata and modified paths (trigger mode only)**
   - Resolve the target repository (`<owner>/<repo>`) from the local Git checkout (`gh repo view --json nameWithOwner --jq .nameWithOwner`).
   - Query the pull request in a single call:
     ```bash
     gh api repos/<owner>/<repo>/pulls/<pr_number> --jq '{number, state, is_fork: (.head.repo != null and .head.repo.full_name != .base.repo.full_name), base_sha: .base.sha, head_sha: .head.sha}'
     ```
   - Require `state == "open"` and `is_fork == true` (if `is_fork` is `false`, stop and tell the user to trigger CI directly on the PR without creating a temporary branch), and record the immutable 40-character `BASE_SHA` and `HEAD_SHA`.
   - Fetch both immutable commits into the local checkout and list changed paths with rename detection disabled:
     ```bash
     git fetch --no-tags origin <BASE_SHA> <HEAD_SHA>
     git --no-pager diff --no-renames --name-only <BASE_SHA>...<HEAD_SHA>
     ```
   - **Pre-trigger confirmation (bound to `HEAD_SHA`)**: If any changed path touches CI, build, container, or deployment inputs (`.azure-pipelines/`, `.github/`, `.devcontainer/`, `cmake/`, `docker/`, `test/deploy/`, `tools/`, any root or nested `**/CMakeLists.txt`, `*.cmake`, `**/Makefile`, `pyproject.toml`, `setup.py`, `setup.cfg`, `python/requirements*.txt`, or `*.sh` / `*.sh.in`), inspect `git --no-pager diff --no-renames <BASE_SHA>...<HEAD_SHA>` first and ask the user for explicit confirmation bound to `HEAD_SHA` before creating a branch or queueing a build. Never reuse confirmation granted for a different `HEAD_SHA`.

2. **Resolve the Azure DevOps project and pipeline ID dynamically**
   - Load the Azure DevOps pipeline tools via `tool_search_tool` if deferred (`azure-devops-pipelines_definition`, `azure-devops-pipelines_build`, `azure-devops-pipelines_write`).
   - Query `gh pr view <pr_number> --repo <owner>/<repo> --json statusCheckRollup` (works for open, closed, or merged PRs) and extract the Azure DevOps `<project>` identifier from an existing pipeline `detailsUrl` (or fall back to CI badge URLs in `README.md` / `azure-devops-core_list_projects`).
   - For trigger requests, look up `<pipeline_id>` via `azure-devops-pipelines_definition` (`action: "list"`, `project: "<project>"`, `name: "<pipeline_name>"`, `repositoryId: "<owner>/<repo>"`, `repositoryType: "GitHub"`).

3. **Create an isolated temporary branch and queue the pipeline**
   - Generate a fresh, single-use branch name `RUN_BRANCH="pr-<pr_number>-<HEAD_SHA_8>-<6_hex_nonce>"` (never reuse or force-update an existing branch) pointing directly to `HEAD_SHA` without creating a merge commit or rebase:
     ```bash
     gh api -X POST repos/<owner>/<repo>/git/refs -f ref="refs/heads/<RUN_BRANCH>" -f sha="<HEAD_SHA>"
     ```
   - Queue the build via `azure-devops-pipelines_write`:
     - `action`: `"run_pipeline"`
     - `project`: `"<project>"`
     - `pipelineId`: `<pipeline_id>`
     - `resources`: `{ "repositories": { "self": { "refName": "refs/heads/<RUN_BRANCH>", "version": "<HEAD_SHA>" } } }`
   - If `run_pipeline` errors or times out after the ref was created, check `azure-devops-pipelines_build` (`action: "list"`, `project: "<project>"`, `branchName: "refs/heads/<RUN_BRANCH>"`): if a build was created, adopt its `buildId` and continue; delete `refs/heads/<RUN_BRANCH>` immediately only when that listing succeeds and returns zero builds (otherwise leave the ref intact and ask the user to reconcile via `cleanup <pr_number>`).

4. **Cancel active fork merge runs**
   - `refs/pull/<pr_number>/merge` runs are automatically queued by GitHub PR webhooks for fork PRs, where they fail due to fork restrictions and hold CI runners.
   - Exhaustively list runs on `refs/pull/<pr_number>/merge` for `<pipeline_id>` via `azure-devops-pipelines_build` (`action: "list"`, `project: "<project>"`, `definitions: [<pipeline_id>]`, `branchName: "refs/pull/<pr_number>/merge"`, `queryOrder: "QueueTimeDescending"`, paging `continuationToken` until `null` without `repositoryId`), and intentionally cancel every `NotStarted` or `InProgress` merge run regardless of commit SHA via `azure-devops-pipelines_write` (`action: "update_build_stage"`, `project: "<project>"`, `buildId: <old_build_id>`, `stageName: "__default"`, `status: "Cancel"`; if a pipeline defines explicit `stages:` in `.azure-pipelines/*.yml` instead of an implicit `__default` stage, cancel each declared stage name—or use build-level `az pipelines build cancel --build-id <old_build_id>` when Azure CLI is authenticated).

5. **Verify GitHub Check Runs**
   - Wait ~10 seconds and check `gh api "repos/<owner>/<repo>/commits/<HEAD_SHA>/check-runs?per_page=100"` to confirm the pipeline's check runs on `HEAD_SHA` reference the new `buildId`.

6. **Clean up `<RUN_BRANCH>` after completion**
   - Keep `refs/heads/<RUN_BRANCH>` until `<buildId>` reaches `Completed` so all queued matrix jobs can fetch the ref when starting.
   - If `save_session_automation` is available and `get_session_automation` shows no active conflicting schedule, register a 15-minute check (`interval: "minutes"`, `every_minutes: 15`) whose prompt:
     1. Queries `azure-devops-pipelines_build` (`action: "list"`, `project: "<project>"`, `buildIds: [<buildId>]`). If this call fails or `status != "Completed"`, waits for the next check.
     2. Once `status == "Completed"`, deletes the dedicated `refs/heads/<RUN_BRANCH>` via `gh api -X DELETE repos/<owner>/<repo>/git/refs/heads/<RUN_BRANCH>` (treating `404` as already clean), clears the schedule via `save_session_automation(clear=true)`, and reports the final result of `<buildId>`.
   - Otherwise — if `save_session_automation` is unavailable or `get_session_automation` already has a conflicting schedule (do **not** overwrite an existing schedule) — remind the user to run `/trigger-fork-pr-ci cleanup <pr_number>` once the build finishes.
