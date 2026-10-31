<#
.SYNOPSIS
  Print the path of the Visual C++ x64 runtime merge module in the newest Visual Studio
  installation on this machine.

.DESCRIPTION
  The desktop MSI merges Microsoft_VC14x_CRT_x64.msm (desktop/src-tauri/wix/
  vc-runtime.wxs). The module is part of Visual Studio's "C++ Redistributable MSMs"
  component (Microsoft.VisualStudio.Component.VC.Redist.MSM), which the GitHub Windows
  runners install. vswhere finds the installations that have it; each names its current
  redistributable version in VC\Auxiliary\Build\Microsoft.VCRedistVersion.default.txt,
  and only that version's directory is used, because an installation can also keep much
  older ones beside it (a 14.16 module, for one, which is too old for PyTorch).

  Fails, rather than printing nothing, when no installation has the module.
#>
param([string]$Arch = "x64")

$ErrorActionPreference = "Stop"

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Write-Error "vswhere.exe not found at ${vswhere}: Visual Studio is not installed"
    exit 1
}
$roots = & $vswhere -all -prerelease -products * `
    -requires Microsoft.VisualStudio.Component.VC.Redist.MSM -property installationPath

$found = @()
foreach ($root in @($roots)) {
    if ([string]::IsNullOrWhiteSpace($root)) { continue }
    $txt = Join-Path $root "VC\Auxiliary\Build\Microsoft.VCRedistVersion.default.txt"
    if (-not (Test-Path $txt)) { continue }
    $ver = (Get-Content $txt -TotalCount 1).Trim()
    $dir = Join-Path $root "VC\Redist\MSVC\$ver\MergeModules"
    $msm = Get-ChildItem -Path $dir -Filter "Microsoft_VC*_CRT_$Arch.msm" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($msm) {
        $found += [pscustomobject]@{ Version = [version]$ver; Path = $msm.FullName }
    }
}

if ($found.Count -eq 0) {
    Write-Error ("no Visual Studio installation here has the Visual C++ $Arch merge module " +
        "(component Microsoft.VisualStudio.Component.VC.Redist.MSM)")
    exit 1
}
($found | Sort-Object Version -Descending | Select-Object -First 1).Path
