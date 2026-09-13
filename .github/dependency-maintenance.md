# Dependency maintenance

Renovate dashboard: https://github.com/dhimmel/openskistats/issues/52

- Keep `website/package.json` engine pins identical to the Node and pnpm pins in `pyproject.toml`.
  Pixi installs these tools from conda-forge for `linux-64` and `osx-64`; npm engine updates alone do not establish eligibility in conda-forge.
- Keep React, React DOM, and their type packages compatible and update them together.
- Refresh Python transitive dependencies with `pixi update` and frontend transitive dependencies with `pnpm --dir website update`, preserving exact direct pins and package-manager release-age checks.
  Follow `AGENTS.md` installation restrictions; `--no-install` for Pixi and `--lockfile-only --ignore-scripts` for pnpm allow lockfile work without installation.
- Validate Python with `pixi run pytest` and `pixi run mypy openskistats`.
  After installing the frontend lockfile, run `pixi run frontend test`, `pixi run frontend typecheck`, and `pixi run frontend build`.

## Deferred toolchain updates

- 2026-09-13: Renovate selects Node 26.8.2 and pnpm 12.4.1 for npm engine updates, but its Pixi update selects Node 26.8.1 and pnpm 12.4.0.
  Keep both manifests on the Pixi-selected versions until Renovate selects the newer conda-forge versions and Pixi resolves them on both supported platforms.
  Evidence: https://github.com/dhimmel/openskistats/issues/52 and https://github.com/dhimmel/openskistats/pull/98.
