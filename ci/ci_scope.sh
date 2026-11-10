#!/usr/bin/env bash
# Does this CI job have anything to check in this change?
#
#   ci/ci_scope.sh <job>
#
# Writes `run=true` or `run=false` to $GITHUB_OUTPUT (stdout when unset) and says why.
# A job with nothing to check skips its STEPS, not itself, and stays green in seconds:
# eight jobs are required status checks on `main`, and a matrix job skipped by a
# job-level `if:` never reports under its matrix name, which would leave a pull request
# waiting for a check that cannot arrive.
#
# Conservative by construction. Every changed file falls into one class, and a path this
# script does not recognise is `engine`, which runs every engine job on every platform.
# A new directory or a new kind of file therefore costs a full run until someone
# classifies it; it cannot make a check skip silently.
#
# A scheduled or manually dispatched run always does everything, as does any event whose
# base cannot be established. A pull request is diffed as its merge commit against the
# first parent, i.e. the base branch it merges into, so the job checks out with
# `fetch-depth: 2`. A push is diffed against the commit the branch pointed to before it
# (PUSH_BEFORE, from `github.event.before`), fetched if the clone lacks it.
#
# Run locally to see what a branch would run, or what a list of paths would:
#   GITHUB_EVENT_NAME=push PUSH_BEFORE=$(git merge-base origin/main HEAD) \
#       RUNNER_OS=Windows bash ci/ci_scope.sh build-test
#   CI_SCOPE_FILES=$'docs/04_convert.md\nscripts/mbr_worker.py' bash ci/ci_scope.sh smoke
set -euo pipefail

job="${1:?usage: ci/ci_scope.sh <job>}"
out="${GITHUB_OUTPUT:-/dev/stdout}"
event="${GITHUB_EVENT_NAME:-}"
os="${RUNNER_OS:-Linux}"

decide() {
    echo "run=$1" >> "$out"
    if [ "$1" = true ]; then
        echo "$job ($os): runs ($2)"
    else
        # An annotation, so a skip shows on the run summary and not only in a log.
        echo "::notice title=$job ($os) has nothing to check::$2"
    fi
    exit 0
}

if [ -n "${CI_SCOPE_FILES:-}" ]; then
    event=files
fi

case "$event" in
    files)
        ;;
    pull_request)
        base=$(git rev-parse -q --verify 'HEAD^1') \
            || decide true "no base commit in the clone (fetch-depth must be 2)"
        ;;
    push)
        base="${PUSH_BEFORE:-}"
        case "$base" in
            "" | 0000000000000000000000000000000000000000)
                decide true "push without a previous commit" ;;
        esac
        if ! git cat-file -e "$base^{commit}" 2>/dev/null; then
            git fetch -q --no-tags --depth=1 origin "$base" \
                || decide true "previous commit $base could not be fetched"
        fi
        ;;
    *)
        decide true "a ${event:-local} run checks everything"
        ;;
esac

if [ "$event" = files ]; then
    files="$CI_SCOPE_FILES"
else
    files=$(git diff --name-only "$base" HEAD) || decide true "git diff failed"
fi
[ -n "$files" ] || decide true "empty diff"

# The first matching pattern wins; in `case`, `*` also matches `/`.
classify() {
    case "$1" in
        # The CI definition and this script: everything.
        .github/workflows/ci.yml | ci/ci_scope.sh) echo all ;;
        # Generated from `--help`; the smoke job checks it against the binary.
        docs/23_cli_reference.md) echo cli_doc ;;
        # Before the Markdown rule: a README under rust/ can be compiled in as crate
        # documentation, and the installer bundles desktop/src-tauri/binaries/*.
        rust/*) echo engine ;;
        desktop/*) echo desktop ;;
        docs/* | *.md | LICENSE | CITATION.cff | .gitignore) echo docs ;;
        .github/dependabot.yml | .github/ISSUE_TEMPLATE/* | .github/CODEOWNERS) echo docs ;;
        # Read only by the Python jobs, which run on every change.
        .github/workflows/release.yml | .github/workflows/docker.yml) echo python ;;
        tests/python/* | bench/* | Dockerfile | .dockerignore | sbom.cdx.json) echo python ;;
        env/docker-*.yml | env/mumdia-*.yml) echo python ;;
        ci/check_console_envs.py | ci/check_doc_refs.py | ci/check_workflows.py) echo python ;;
        ci/gen_config_reference.py | ci/gen_third_party_licenses.py | ci/gen_sbom.py) echo python ;;
        ci/_lockfile_versions.py) echo python ;;
        # The engine's tests include_str! some workers, so a script change still builds
        # and tests the engine once, on Linux.
        scripts/*) echo scripts ;;
        # What the desktop application compiles in or is checked against.
        env/console-* | configs/config-schema.json) echo desktop ;;
        # The Linux desktop job's frontend checks, which the installer does not run.
        ci/check_desktop_ui.py | ci/test_desktop_ui.mjs) echo desktop_ui ;;
        ci/check_msi_runtime.ps1 | ci/find_vc_merge_module.ps1) echo msi ;;
        # Everything else is the engine: configs/examples, docker/*.json, test_data/,
        # the smoke scripts, ci/run_name_cases.json, .gitattributes, and any path this
        # script has not been taught about.
        *) echo engine ;;
    esac
}

seen=" "
while IFS= read -r f; do
    [ -n "$f" ] || continue
    c=$(classify "$f")
    case "$seen" in *" $c "*) ;; *) seen="$seen$c " ;; esac
done <<EOF
$files
EOF

has() {
    local c
    for c in "$@"; do
        case "$seen" in *" $c "*) return 0 ;; esac
    done
    return 1
}

why="changed: $(printf '%s\n' $seen | sort | tr '\n' ' ' | sed 's/ *$//')"

has all && decide true "$why"

case "$job" in
    # fmt, clippy and rustdoc of the engine workspace.
    lint)
        has engine && decide true "$why" ;;
    # Linux also covers the workers the engine compiles in, and gives a desktop change
    # one engine run, which costs no wall time beside the desktop jobs.
    build-test)
        has engine && decide true "$why"
        [ "$os" = Linux ] && has scripts desktop && decide true "$why" ;;
    smoke)
        has engine && decide true "$why"
        [ "$os" = Linux ] && has scripts desktop cli_doc && decide true "$why" ;;
    # Compares the two smoke legs, so it follows the Windows one.
    smoke-cross-platform)
        has engine && decide true "$why" ;;
    # The Linux desktop job compiles in engine sources (enzymes, modifications, example
    # configs, the Python-resolution code), and from a checkout the application finds
    # its workers in scripts/.
    desktop)
        has engine desktop desktop_ui scripts && decide true "$why" ;;
    # The installer: desktop/ and what the application compiles in, the two installer
    # scripts, and the workflow (the `all` class above).
    desktop-windows)
        has desktop msi && decide true "$why" ;;
    *)
        echo "::error::ci/ci_scope.sh: unknown job '$job'" >&2
        exit 2 ;;
esac

decide false "$why"
