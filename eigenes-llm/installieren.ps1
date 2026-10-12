# Einrichtung unter Windows: virtuelle Umgebung + PyTorch mit CUDA + Pakete.
# Start per Doppelklick auf 1_installieren.bat (oder: powershell -ExecutionPolicy Bypass -File installieren.ps1)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

function Info($text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Warn($text) { Write-Host "!!  $text" -ForegroundColor Yellow }

# 1. Python finden
$python = $null
$pyExe = $null
$pyPre = @()
foreach ($cand in @("py -3.13", "py -3.12", "py -3", "python")) {
    $parts = $cand.Split(" ")
    $exe = $parts[0]
    $pre = @($parts | Select-Object -Skip 1)
    try {
        $ver = & $exe @pre -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $ver) { $python = $cand; $pyExe = $exe; $pyPre = $pre; break }
    } catch { }
}
if (-not $python) {
    Warn "Python wurde nicht gefunden. Bitte von https://www.python.org/downloads/ installieren"
    Warn "und beim Installieren 'Add python.exe to PATH' anhaken."
    exit 1
}
Info "Python gefunden: $python (Version $ver)"

# 2. Virtuelle Umgebung
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Info "Lege virtuelle Umgebung .venv an ..."
    & $pyExe @pyPre -m venv .venv
}
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip --quiet

# 3. PyTorch mit CUDA (neueste passende Variante zuerst)
$hasCuda = $false
try { $hasCuda = (& $py -c "import torch; print(torch.cuda.is_available())" 2>$null) -eq "True" } catch { }
if (-not $hasCuda) {
    foreach ($cu in @("cu130", "cu128", "cu126")) {
        Info "Installiere PyTorch mit CUDA ($cu) - das sind ca. 2-3 GB, bitte warten ..."
        & $py -m pip install --upgrade torch --index-url "https://download.pytorch.org/whl/$cu"
        if ($LASTEXITCODE -eq 0) {
            $ok = & $py -c "import torch; print(torch.cuda.is_available())" 2>$null
            if ($ok -eq "True") { $hasCuda = $true; break }
            Warn "PyTorch ($cu) installiert, findet aber keine Grafikkarte - versuche aeltere Variante."
        }
    }
}
if (-not $hasCuda) {
    Warn "PyTorch findet keine NVIDIA-Grafikkarte."
    Warn "Meist hilft: neuesten NVIDIA-Treiber installieren (NVIDIA App) und dieses Skript erneut starten."
    Warn "Python 3.14 ist eventuell zu neu fuer PyTorch - dann Python 3.13 installieren und erneut starten."
}

# 4. Restliche Pakete
Info "Installiere restliche Pakete ..."
& $py -m pip install -r requirements.txt --quiet
if ($LASTEXITCODE -ne 0) { Warn "Installation der Pakete fehlgeschlagen."; exit 1 }

# 5. Check
Info "Systemcheck:"
& $py -m llm info
Write-Host ""
Info "Fertig! Naechster Schritt: 2_daten_vorbereiten.bat"
