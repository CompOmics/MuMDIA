<#
.SYNOPSIS
  Fail unless an MSI carries the Microsoft Visual C++ x64 runtime that PyTorch needs.

.DESCRIPTION
  The desktop installer merges Microsoft's VC++ CRT merge module (desktop/src-tauri/wix/
  vc-runtime.wxs). This script opens the built package with the Windows Installer
  automation interface, read-only, and checks what a merge that silently did nothing
  would not have:

  - the module is recorded in the package's ModuleSignature table;
  - vcruntime140.dll, vcruntime140_1.dll and msvcp140.dll are in the File table, at a
    version of at least -MinVersion (PyTorch's Windows wheels are built with a toolset
    whose STL needs 14.40 or newer; an older runtime is the WinError 1114 on c10.dll);
  - their components install into the 64-bit system directory (System64Folder), where
    Windows looks for the runtime, rather than beside the application.

  It prints what it found, so a CI log is the evidence. Also accepts a .msm, to look at
  the module before it is merged.

.EXAMPLE
  pwsh ci/check_msi_runtime.ps1 -Msi desktop/dist/MuMDIA_0.9.0_x64_en-US.msi
#>
param(
    [Parameter(Mandatory = $true)][string]$Msi,
    [string]$MinVersion = "14.40"
)

$ErrorActionPreference = "Stop"

function Invoke-Com($obj, [string]$name, [string]$kind, [object[]]$argv) {
    $flags = [System.Reflection.BindingFlags]::$kind
    return $obj.GetType().InvokeMember($name, $flags, $null, $obj, $argv)
}

function Get-Rows($db, [string]$sql) {
    $view = Invoke-Com $db "OpenView" "InvokeMethod" @($sql)
    Invoke-Com $view "Execute" "InvokeMethod" $null | Out-Null
    $rows = @()
    while ($true) {
        $rec = Invoke-Com $view "Fetch" "InvokeMethod" $null
        if ($null -eq $rec) { break }
        $n = [int](Invoke-Com $rec "FieldCount" "GetProperty" $null)
        $vals = @()
        for ($k = 1; $k -le $n; $k++) {
            $vals += [string](Invoke-Com $rec "StringData" "GetProperty" @([int]$k))
        }
        $rows += , $vals
    }
    Invoke-Com $view "Close" "InvokeMethod" $null | Out-Null
    return , $rows
}

function Test-Table($db, [string]$table) {
    $rows = Get-Rows $db "SELECT ``Name`` FROM ``_Tables`` WHERE ``Name`` = '$table'"
    return $rows.Count -gt 0
}

$path = (Resolve-Path $Msi).Path
$installer = New-Object -ComObject WindowsInstaller.Installer
# 0 = msiOpenDatabaseModeReadOnly
$db = Invoke-Com $installer "OpenDatabase" "InvokeMethod" @($path, 0)
Write-Output "package: $path"

$failures = @()

if (Test-Table $db "ModuleSignature") {
    $mods = Get-Rows $db "SELECT ``ModuleID``, ``Language``, ``Version`` FROM ``ModuleSignature``"
    Write-Output "merge modules (ModuleSignature):"
    foreach ($m in $mods) { Write-Output ("  {0}  language {1}  version {2}" -f $m[0], $m[1], $m[2]) }
    if (-not ($mods | Where-Object { $_[0] -match "CRT" })) {
        $failures += "no CRT merge module in ModuleSignature"
    }
} else {
    $failures += "the package has no ModuleSignature table: nothing was merged"
}

$files = Get-Rows $db "SELECT ``File``, ``FileName``, ``Version``, ``Component_`` FROM ``File``"
$components = @{}
foreach ($c in (Get-Rows $db "SELECT ``Component``, ``Directory_`` FROM ``Component``")) {
    $components[$c[0]] = $c[1]
}
$min = [version]$MinVersion
Write-Output "runtime files (File table):"
foreach ($want in @("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")) {
    # FileName is "SHORT~1.DLL|long name.dll" when the long name is not 8.3.
    $hit = $files | Where-Object { (($_[1] -split "\|")[-1]) -ieq $want } | Select-Object -First 1
    if (-not $hit) { $failures += "$want is not in the File table"; continue }
    $dir = $components[$hit[3]]
    Write-Output ("  {0}  version {1}  component {2}  directory {3}" -f $want, $hit[2], $hit[3], $dir)
    if ([string]::IsNullOrEmpty($hit[2]) -or ([version]$hit[2]) -lt $min) {
        $failures += "$want is version '$($hit[2])', older than $MinVersion"
    }
    if ($dir -notlike "System64Folder*") {
        $failures += "$want installs into '$dir', not the 64-bit system directory"
    }
}

if ($failures.Count -gt 0) {
    foreach ($f in $failures) { Write-Output "::error::$f" }
    exit 1
}
Write-Output "the package carries the Visual C++ x64 runtime ($MinVersion or newer)"
# An explicit exit code: the callers test $LASTEXITCODE, which is $null in a step that ran no
# native command before this script, and `$null -ne 0` is true.
exit 0
