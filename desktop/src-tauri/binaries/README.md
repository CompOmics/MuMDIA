# Bundled binaries

Filled in by `release.yml` at build time, and deliberately empty in the repository:
these are build artifacts of another workspace and are megabytes each.

At bundle time this directory holds

- `mumdia` (`mumdia.exe` on Windows), the search engine the application spawns, built
  from the same checkout so a released application and its engine cannot disagree;
- `uv` (`uv.exe`), the installer used to create the managed Python environment
  without conda.

The directory is declared as a Tauri `resource`, and a resource glob that matches
nothing fails the build. This file is what keeps it matching while the directory is
otherwise empty, so a developer running `cargo tauri build` locally gets a working
bundle rather than a packaging error.

`src/engine.rs` and `src/components.rs` look here, and beside the executable, when
resolving those two programs.
