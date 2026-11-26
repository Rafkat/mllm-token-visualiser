# mllm-token-visualiser — Claude Code Project Instructions

Repository: `Rafkat/mllm-token-visualiser`  
Primary execution: VPS, Linux user `claude`  
Development model: feature branch -> tests -> push -> pull request -> human review

**Instruction scope:** This file governs only `mllm-token-visualiser`. Read and comply with `~/.claude/CLAUDE.md` for global rules and the project registry. The other project has independent instructions and is out of scope for Token Visualiser tasks.

## 1. Mission

Maintain and extend `mllm-token-visualiser`, a tool for inspecting and estimating token use and related resource characteristics in multimodal language models. Prioritize accurate model-specific accounting, useful visualizations, robust APIs/CLI, reproducible results, and clear documentation.

Potential research areas include multimodal tokenization for images, video, audio, streaming inputs, and KV-cache/memory estimates. These are areas to explore, not assertions that every feature is implemented. Determine supported models and current behavior from the code and tests rather than assuming.

The principal aim is a reliable developer tool, not expensive model training.

## 2. Scope, architecture, and quality

- First inspect `README.md`, `pyproject.toml`, package entry points, adapters, existing tests, and the actual repository structure.
- Keep compatibility with existing documented interfaces unless a breaking change is justified and approved.
- Use small, explicit, model-specific fixtures and tests when possible; do not assume a model's token count from generic heuristics when processor details differ.
- Keep token counting, metadata interpretation, visualization, and memory estimation conceptually separate where practical.
- Clearly label exact counts vs estimates and document assumptions such as resolution, crop/patch strategy, frame sampling, special tokens, sequence length, and KV-cache configuration.
- Never claim support for a model, processor version, or modality until it has been verified appropriately.
- Add targeted regression tests for bugs and new adapters/features. Prefer existing tooling and pinned dependencies.
- Avoid installing heavyweight model stacks or downloading weights for code paths that can be tested with synthetic inputs or lightweight mocks.

## 3. Runtime and compute policy

### Default: VPS only

- Run ordinary research, development, tests, static checks, and documentation on the VPS.
- VPS resources are limited (~2 GiB RAM, ~15 GB disk): check `free -h` / `df -h` before dependency installs or larger runs.
- Use lightweight `uv`/Python environments according to the repository's existing tooling. Avoid resolving or downloading large optional ML dependencies by default.
- Do not launch full-model inference, training, or substantial downloads on the VPS.

### Exception: RTX 4070

- Remote SSH alias from VPS: `home4070` (remote user `claude-runner`).
- GPU: NVIDIA RTX 4070, 12 GiB-class VRAM; system RAM approximately 32 GiB.
- Use only when a task truly requires more RAM, CUDA, model weights, or GPU tests that are impractical on the VPS; state the reason first and obtain approval before expensive or disruptive work.
- Do not assume this project's checkout or environment exists there. Establish a dedicated working directory under `~/workspace/projects/mllm-token-visualiser` only when authorized and necessary, without modifying any other project checkout.
- Keep optional model downloads and generated artifacts in approved local experiment directories; do not use the protected dataset NFS tree.
- Avoid simultaneous GPU workloads that interfere with other ongoing experiments. Check resources and active processes before use.
- Do not use RTX 3060 for this repository unless explicitly authorized.

## 4. Git and pull-request workflow

Default branch: `main` (verify against GitHub if repository settings differ). Work in the existing `Rafkat/mllm-token-visualiser` repository; do not create a separate agent fork/repository unless directed.

For every substantive task:

1. Confirm the intended repository and clean/safe worktree; preserve uncommitted user changes.
2. Fetch `origin` and branch from current `origin/main` (or the verified default branch).
3. Create a descriptive branch such as `agent/fix-video-token-count` or `agent/add-adapter-tests`.
4. Implement a focused change with appropriate docs and tests.
5. Run relevant tests/static checks; note any that could not run due to resource limits.
6. Commit with an accurate message. Push **only the feature branch** to this repository.
7. Open a PR against `main` with a clear summary, rationale, validation evidence, limitations, and any follow-up work. Use `gh pr create` if authenticated.
8. Provide the PR URL to the user. Never self-approve or merge PRs, even if technically permitted.

Do not push directly to `main`, force-push, delete remote branches without approval, or change branch protections/rulesets. Do not modify GitHub Actions workflows or repository permissions without discussing impact first.

**Permissions:** Git push and GitHub API PR creation are distinct capabilities. The user must configure repository-scoped authorization for the VPS. Never attempt to copy image-generation deploy keys or acquire unrelated credentials.

## 5. Development and validation

- Reuse existing test and lint commands discovered from the repository; do not invent commands or assume optional packages are installed.
- Prefer CPU-only unit tests, mocked model metadata and processor outputs, deterministic synthetic multimodal examples, and small sample files.
- When reporting memory footprints or token counts, record model/version, inputs, method, and assumptions.
- Test edge cases: empty inputs, invalid or missing modality information, variable image/video resolutions, special token handling, and version-dependent behavior when relevant.
- Be cautious about large allocations: estimate required RAM/disk before starting.
- Preserve documented CLI/API outputs unless intentional changes are adequately tested and described.

## 6. Documentation, research, and communication

- Keep README/usage examples accurate and aligned with code.
- For changes that affect user-visible behavior, explain the result with reproducible commands or a small example.
- Cite primary model documentation, processor source, or tests when discussing model-specific behavior; note uncertain details.
- Keep discussions concise and research notes actionable; avoid large speculative roadmaps when a focused patch is sufficient.
- At the end of a task report: branch, commit, PR URL (if created), tests and results, remaining limitations, and any resource-related blockers.

## 7. Boundaries

- The VPS also hosts `Rafkat/clothes_and_models_generation_agent`, but this is only cross-project awareness, not a task dependency or permission.
- Do not access or modify `clothes_and_models_generation_agent`, shared image datasets, or other repositories when working on this project.
- Do not copy code, models, datasets, credentials, or experiment outputs between repositories without explicit authorization.
- Do not transfer private datasets or personal images from GPU nodes to this repository or VPS.
- No automatic merges, privileged commands, destructive cleanup, external releases, or uncontrolled resource-heavy work.
- Follow global VPS instructions in `~/.claude/CLAUDE.md` as well as this file.

