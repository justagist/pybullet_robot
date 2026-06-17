# CHANGELOG.md

## [0.2.1] - 2026-06-17

### Changed

- Fixed minor bugs and documentation typos.
- Update ci action versions.

# CHANGELOG.md

## [0.2.0] - 2026-06-17

### Changed

- Modernised packaging: full project metadata (keywords, classifiers, project URLs) and `[project.optional-dependencies]` (`test`, `dev`) in `pyproject.toml`; explicit `setuptools` package discovery.
- Releases now publish to PyPI via tag-triggered GitHub Actions using PyPI trusted publishing (OIDC) instead of a stored API token.

### Added

- CI workflow running an install + import smoke test across Python 3.10–3.12 and a build/metadata check.
- `.devcontainer` for a reproducible development environment.

### Removed

- Removed `pixi` configuration and `pixi.lock`; development now uses a plain `pip install -e ".[dev]"` workflow.

## [0.1.4] - 2024-08-07

### Fixes

- pypi installation conflict in numpy versions for pyrcf and pybullet_robot pkgs

## [0.1.3] - 2024-08-07

### Adds

- support using pre-loaded robot to create bulletrobot class
- utility tool for retrieving mjcf files from robot_descriptions.py
- (experimental): support for mjcf files in bulletrobot

### Fixes

- pypi installation does not work with conda dependencies; use pypi dependencies instead

## [0.1.2] - 2024-08-06

### Features

- add minimal example on robot loading and writing custom controllers
- migrate to pixi install

## [0.1.1]

### Features

- Release to pypi
- Bug fixes

## [0.1.0]

### Features

- Initial working version of robot interface and IK interface
