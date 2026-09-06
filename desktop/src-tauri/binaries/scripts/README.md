# Bundled Python workers

Filled in by `release.yml` at build time from the repository's top-level `scripts/`,
and deliberately empty here for the same reason as the parent directory: these are
copies, and a second copy in the repository would drift from the original.

## Why the workers live beside the engine and not somewhere tidier

The engine resolves its `predict_frag.sidecar_script_dir` itself. The shipped
default is the relative string `"scripts"`, and `python::resolve_script_dir` tries,
in order, the directory of the configuration file, then the directory of the engine
binary, and only then the working directory. In a bundle there is no configuration
file beside the workers and the working directory is wherever the user happened to
launch the application from, so the engine's own directory is the only stable
answer. The engine is staged into `binaries/`, so the workers go into
`binaries/scripts/`.

Without this the bundled application had no sidecars at all: DeepLC, mokapot and the
neural rescorer would each fail at the point of use, which is hours into a search
rather than at startup.

`src/engine.rs::scripts_dir` looks here too, for the DIA-NN library conversion,
which calls two of these workers directly rather than through the engine.

As with the parent directory, a Tauri resource glob that matches nothing fails the
build. This file keeps `binaries/scripts/*` matching while the directory is
otherwise empty.
