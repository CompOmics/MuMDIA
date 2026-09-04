# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities through GitHub's private vulnerability
reporting: open <https://github.com/CompOmics/MuMDIA/security/advisories/new>, or
go to the repository's **Security** tab and choose **Report a vulnerability**.
This creates a private advisory visible only to the maintainers.

Please do not open a public issue for a suspected vulnerability, and do not
include a working exploit in the initial report. A description of the class of
problem, the affected component, and how you observed it is enough to start.

We aim to acknowledge a report within five working days. MuMDIA is a research
tool maintained by an academic group, so remediation time depends on severity and
on whether the problem sits in MuMDIA or in a dependency.

## Supported versions

Only the latest release receives fixes. There are no long-term support branches.

## Threat model

MuMDIA is a local command-line data-analysis tool. It reads mass spectrometry
data and spectral libraries, runs local computation, writes result files, and
launches Python interpreters that the user names in the configuration. It is not
a network service: it opens no listening socket and, apart from the Python
packages an environment pulls at install time, it makes no network requests of
its own.

The security-relevant surfaces are therefore:

- **Untrusted input files.** mzML, FASTA, and Parquet inputs are parsed by
  MuMDIA and its dependencies. A malformed or hostile file that causes memory
  unsafety, a crash exploitable beyond denial of service, or execution of
  attacker-controlled code is in scope. A malformed file that simply produces an
  error, or a wrong scientific result, is a correctness bug: report it as a normal
  issue.
- **Configuration as code.** A MuMDIA configuration names Python interpreter
  paths and a sidecar script directory, and the engine executes them. A
  configuration file is therefore as trusted as a shell script: treat one from an
  untrusted source the same way. Findings that depend on the user having supplied
  a hostile configuration are not vulnerabilities in MuMDIA.
- **Dependencies.** Vulnerabilities in Rust crates or Python packages we pin. If
  the upstream project has an advisory, we would rather have a pull request
  bumping the pin than a private report.
- **The published Docker image.** `ghcr.io/compomics/mumdia`. Report base-image
  or bundled-dependency issues here.

Out of scope: results that are scientifically wrong, denial of service from
oversized inputs (the tool is expected to be resource-hungry and is run
deliberately by its user), and anything requiring local privileges the user
already has.

## Credentials

MuMDIA needs no credentials or tokens to run. If you find any code path that
reads, stores, or transmits one, that is a bug worth reporting, because nothing
in the pipeline should need it.
