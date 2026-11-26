<#
.SYNOPSIS
  Fail unless an MSI carries the installer's choice of where the analysis components go.

.DESCRIPTION
  The desktop installer links desktop/src-tauri/wix/components-scope.wxs, which lets the
  administrator choose between each user's own folder (the default) and the program folder
  for the analysis components, as a page in the interactive installer and as the property
  COMPONENTS_SCOPE for a silent one. This script opens the built package read-only with the
  Windows Installer automation interface and checks each piece a fragment that was not linked,
  or was linked into the wrong place, would lack:

  - COMPONENTS_SCOPE is a secure property with no default of its own; INSTALL_COMPONENTS
    defaults to 1;
  - the remembered choice is read from HKLM\Software\CompOmics\MuMDIA, value ComponentsScope,
    in the 64-bit view, and the custom actions that restore it and set the default "user"
    run after AppSearch and before LaunchConditions in both sequences;
  - a launch condition refuses any other value;
  - the same registry value is written with [COMPONENTS_SCOPE] by a 64-bit component;
  - the page MumdiaScopeDlg exists with the two choices, bound to its own MumdiaScopeChoice
    (a radio button group needs a Property table default, ICE34), which is set from
    COMPONENTS_SCOPE before the page shows and copied back on Next; and the InstallDirDlg
    Next and VerifyReadyDlg Back buttons lead to it with a later ordering than their other
    NewDialog events, which is what makes it the one that is shown;
  - Finish starts the application with --install-components-on-start when asked to.

  It prints what it found, so a CI log is the evidence.

.EXAMPLE
  pwsh ci/check_msi_scope.ps1 -Msi desktop/target/release/bundle/msi/MuMDIA_0.15.1_x64_en-US.msi
#>
param(
    [Parameter(Mandatory = $true)][string]$Msi
)

$ErrorActionPreference = "Stop"

function Invoke-Com($obj, [string]$name, [string]$kind, [object[]]$argv) {
    $flags = [System.Reflection.BindingFlags]::$kind
    return $obj.GetType().InvokeMember($name, $flags, $null, $obj, $argv)
}

# The tables a query named that the package does not have: a package without the fragment
# lacks some of them altogether, which is a failure to report, not an exception.
$script:absent = @()

function Get-Rows($db, [string]$sql) {
    if ($sql -match 'FROM `(\w+)`' -and $Matches[1] -ne "_Tables") {
        $table = $Matches[1]
        if ((Get-Rows $db "SELECT ``Name`` FROM ``_Tables`` WHERE ``Name`` = '$table'").Count -eq 0) {
            if ($script:absent -notcontains $table) { $script:absent += $table }
            return , @()
        }
    }
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

$path = (Resolve-Path $Msi).Path
$installer = New-Object -ComObject WindowsInstaller.Installer
# 0 = msiOpenDatabaseModeReadOnly
$db = Invoke-Com $installer "OpenDatabase" "InvokeMethod" @($path, 0)
Write-Output "package: $path"

$failures = @()
$key = "Software\CompOmics\MuMDIA"
$valueName = "ComponentsScope"

# --- properties ---------------------------------------------------------------------------
$props = @{}
foreach ($r in (Get-Rows $db "SELECT ``Property``, ``Value`` FROM ``Property``")) { $props[$r[0]] = $r[1] }
if ($props.ContainsKey("COMPONENTS_SCOPE")) {
    $failures += "COMPONENTS_SCOPE has a default value ('$($props["COMPONENTS_SCOPE"])'), which would override the remembered choice"
}
if ($props["INSTALL_COMPONENTS"] -ne "1") {
    $failures += "INSTALL_COMPONENTS does not default to 1"
}
if ($props["MumdiaScopeChoice"] -ne "user") {
    $failures += "the page's MumdiaScopeChoice does not default to user"
}
$secure = ($props["SecureCustomProperties"] -split ";")
foreach ($p in @("COMPONENTS_SCOPE", "MUMDIA_REMEMBERED_SCOPE", "INSTALL_COMPONENTS")) {
    if ($secure -notcontains $p) { $failures += "$p is not a secure property" }
}
Write-Output "properties: INSTALL_COMPONENTS=$($props["INSTALL_COMPONENTS"]), secure: $(($secure | Where-Object { $_ -match 'SCOPE|COMPONENTS' }) -join ', ')"

# --- the remembered choice ----------------------------------------------------------------
$locators = Get-Rows $db "SELECT ``Signature_``, ``Root``, ``Key``, ``Name``, ``Type`` FROM ``RegLocator``"
$loc = $locators | Where-Object { $_[2] -eq $key -and $_[3] -eq $valueName } | Select-Object -First 1
if (-not $loc) {
    $failures += "no RegLocator reads HKLM\$key\$valueName"
} else {
    Write-Output ("search: {0} root {1} {2}\{3} type {4}" -f $loc[0], $loc[1], $loc[2], $loc[3], $loc[4])
    # Root 2 = HKEY_LOCAL_MACHINE; Type bit 16 = msidbLocatorType64bit.
    if ($loc[1] -ne "2") { $failures += "the remembered choice is not read from HKEY_LOCAL_MACHINE" }
    if ((([int]$loc[4]) -band 16) -eq 0) { $failures += "the remembered choice is not read from the 64-bit registry view" }
    $app = Get-Rows $db "SELECT ``Property``, ``Signature_`` FROM ``AppSearch``"
    if (-not ($app | Where-Object { $_[0] -eq "MUMDIA_REMEMBERED_SCOPE" -and $_[1] -eq $loc[0] })) {
        $failures += "AppSearch does not fill MUMDIA_REMEMBERED_SCOPE from that search"
    }
}

$actions = @{}
foreach ($r in (Get-Rows $db "SELECT ``Action``, ``Type``, ``Source``, ``Target`` FROM ``CustomAction``")) { $actions[$r[0]] = $r }
foreach ($a in @(
        @("MumdiaScopeRemembered", "COMPONENTS_SCOPE", "[MUMDIA_REMEMBERED_SCOPE]"),
        @("MumdiaScopeDefault", "COMPONENTS_SCOPE", "user"),
        @("MumdiaScopeMachine", "COMPONENTS_SCOPE", "machine"),
        @("MumdiaScopeUser", "COMPONENTS_SCOPE", "user"))) {
    $row = $actions[$a[0]]
    if (-not $row) { $failures += "custom action $($a[0]) is missing"; continue }
    # 51 = msidbCustomActionTypeTextData + msidbCustomActionTypeProperty: a property assignment.
    if ((([int]$row[1]) -band 63) -ne 51) { $failures += "custom action $($a[0]) is not a property assignment" }
    if ($row[2] -ne $a[1] -or $row[3] -ne $a[2]) {
        $failures += "custom action $($a[0]) sets $($row[2]) to '$($row[3])', not $($a[1]) to '$($a[2])'"
    }
}
foreach ($table in @("InstallUISequence", "InstallExecuteSequence")) {
    $seq = @{}
    foreach ($r in (Get-Rows $db "SELECT ``Action``, ``Sequence`` FROM ``$table``")) { $seq[$r[0]] = [int]$r[1] }
    $order = @("AppSearch", "MumdiaScopeRemembered", "MumdiaScopeDefault", "MumdiaScopeMachine", "MumdiaScopeUser", "LaunchConditions")
    $missing = $order | Where-Object { -not $seq.ContainsKey($_) }
    if ($missing) { $failures += "$table lacks $($missing -join ', ')"; continue }
    $nums = $order | ForEach-Object { $seq[$_] }
    Write-Output ("{0}: {1}" -f $table, (($order | ForEach-Object { "$_ $($seq[$_])" }) -join ", "))
    for ($i = 1; $i -lt $nums.Count; $i++) {
        if ($nums[$i] -le $nums[$i - 1]) { $failures += "$table runs $($order[$i]) before $($order[$i - 1])" }
    }
}

# The page's copy is set after the normalised choice, in the interface sequence.
$toPage = $actions["MumdiaScopeToPage"]
if (-not $toPage -or $toPage[2] -ne "MumdiaScopeChoice" -or $toPage[3] -ne "[COMPONENTS_SCOPE]") {
    $failures += "MumdiaScopeToPage does not set MumdiaScopeChoice from [COMPONENTS_SCOPE]"
} else {
    $ui = @{}
    foreach ($r in (Get-Rows $db "SELECT ``Action``, ``Sequence`` FROM ``InstallUISequence``")) { $ui[$r[0]] = [int]$r[1] }
    if (-not $ui.ContainsKey("MumdiaScopeToPage") -or $ui["MumdiaScopeToPage"] -le $ui["MumdiaScopeUser"]) {
        $failures += "InstallUISequence does not set the page's copy after the choice is normalised"
    }
}

$conds = Get-Rows $db "SELECT ``Condition``, ``Description`` FROM ``LaunchCondition``"
$cond = $conds | Where-Object { $_[0] -match "COMPONENTS_SCOPE" } | Select-Object -First 1
if (-not $cond) { $failures += "no launch condition checks COMPONENTS_SCOPE" }
else { Write-Output "launch condition: $($cond[0])" }

# --- the value the application reads -----------------------------------------------------
$regs = Get-Rows $db "SELECT ``Registry``, ``Root``, ``Key``, ``Name``, ``Value``, ``Component_`` FROM ``Registry``"
$reg = $regs | Where-Object { $_[2] -eq $key -and $_[3] -eq $valueName } | Select-Object -First 1
if (-not $reg) {
    $failures += "no Registry row writes HKLM\$key\$valueName"
} else {
    Write-Output ("registry: root {0} {1}\{2} = {3} (component {4})" -f $reg[1], $reg[2], $reg[3], $reg[4], $reg[5])
    if ($reg[1] -ne "2") { $failures += "the choice is not written to HKEY_LOCAL_MACHINE" }
    if ($reg[4] -ne "[COMPONENTS_SCOPE]") { $failures += "the choice is written as '$($reg[4])', not [COMPONENTS_SCOPE]" }
    $comp = Get-Rows $db "SELECT ``Component``, ``Attributes`` FROM ``Component`` WHERE ``Component`` = '$($reg[5])'"
    # 256 = msidbComponentAttributes64bit: the 64-bit registry view, where the application reads.
    if (-not $comp -or ((([int]$comp[0][1]) -band 256) -eq 0)) {
        $failures += "the component writing the choice is not a 64-bit component"
    }
    $fc = Get-Rows $db "SELECT ``Feature_``, ``Component_`` FROM ``FeatureComponents`` WHERE ``Component_`` = '$($reg[5])'"
    if (-not $fc) { $failures += "the component writing the choice is in no feature, so it is never installed" }
}

# --- the page and the way to it ------------------------------------------------------------
$dialogs = Get-Rows $db "SELECT ``Dialog`` FROM ``Dialog`` WHERE ``Dialog`` = 'MumdiaScopeDlg'"
if (-not $dialogs) { $failures += "the page MumdiaScopeDlg is not in the package" }
$radios = Get-Rows $db "SELECT ``Property``, ``Value`` FROM ``RadioButton`` WHERE ``Property`` = 'MumdiaScopeChoice'"
$values = ($radios | ForEach-Object { $_[1] }) -join ","
Write-Output "choices: $values"
if ($values -ne "user,machine") { $failures += "the page offers '$values', not user,machine" }

$events = Get-Rows $db "SELECT ``Dialog_``, ``Control_``, ``Event``, ``Argument``, ``Ordering`` FROM ``ControlEvent``"
function Test-Leads($dialog, $control, $target) {
    # @(): a single matching row would otherwise be unrolled into its strings.
    $mine = @($events | Where-Object { $_[0] -eq $dialog -and $_[1] -eq $control -and $_[2] -eq "NewDialog" -and $_[3] -eq $target })
    if ($mine.Count -eq 0) { return "$dialog $control does not lead to $target" }
    $others = @($events | Where-Object { $_[0] -eq $dialog -and $_[1] -eq $control -and $_[2] -eq "NewDialog" -and $_[3] -ne $target })
    $top = ($others | ForEach-Object { [int]$_[4] } | Measure-Object -Maximum).Maximum
    # Write-Host: output written here would be returned with the verdict.
    Write-Host ("{0} {1}: NewDialog {2} at {3}, others at most {4}" -f $dialog, $control, $target, $mine[0][4], $top)
    if ([int]$mine[0][4] -le [int]$top) { return "$dialog $control leads to $target too early to be the dialog shown" }
    return $null
}
foreach ($leg in @(@("InstallDirDlg", "Next", "MumdiaScopeDlg"), @("VerifyReadyDlg", "Back", "MumdiaScopeDlg"))) {
    $f = Test-Leads $leg[0] $leg[1] $leg[2]
    if ($f) { $failures += $f }
}
$launch = $actions["MumdiaOpenForComponents"]
if (-not $launch -or $launch[2] -ne "Path" -or $launch[3] -ne "--install-components-on-start") {
    $failures += "MumdiaOpenForComponents does not start the application with --install-components-on-start"
}
if (-not ($events | Where-Object { $_[0] -eq "ExitDialog" -and $_[1] -eq "Finish" -and $_[2] -eq "DoAction" -and $_[3] -eq "MumdiaOpenForComponents" })) {
    $failures += "Finish does not run MumdiaOpenForComponents"
}
$copy = @($events | Where-Object { $_[0] -eq "MumdiaScopeDlg" -and $_[1] -eq "Next" -and $_[2] -eq "[COMPONENTS_SCOPE]" -and $_[3] -eq "[MumdiaScopeChoice]" })
$leave = @($events | Where-Object { $_[0] -eq "MumdiaScopeDlg" -and $_[1] -eq "Next" -and $_[2] -eq "NewDialog" -and $_[3] -eq "VerifyReadyDlg" })
if ($copy.Count -eq 0) {
    $failures += "Next on the page does not copy the choice into COMPONENTS_SCOPE"
} elseif ($leave.Count -eq 0 -or [int]$copy[0][4] -ge [int]$leave[0][4]) {
    $failures += "Next on the page leaves before it copies the choice into COMPONENTS_SCOPE"
}
if (-not ($events | Where-Object { $_[0] -eq "MumdiaScopeDlg" -and $_[1] -eq "Next" -and $_[2] -eq "[LAUNCHAPPARGS]" -and $_[3] -eq "--install-components-on-start" })) {
    $failures += "the page does not pass --install-components-on-start to the template's own launch"
}

foreach ($t in $script:absent) { $failures += "the package has no $t table" }
if ($failures.Count -gt 0) {
    foreach ($f in $failures) { Write-Output "::error::$f" }
    exit 1
}
Write-Output "the package carries the components scope choice"
exit 0
