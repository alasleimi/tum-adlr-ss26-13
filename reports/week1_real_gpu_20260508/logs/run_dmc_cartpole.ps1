Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root

$ProgressLog = "reports/week1_real_gpu_20260508/logs/dmc_cartpole_25k_progress.log"
$Stamp = Get-Date -Format o
Add-Content -LiteralPath $ProgressLog -Value "$Stamp START dmc_cartpole_25k seed 0"
& python -m last_nine_rl.train `
    --config configs/week1_cartpole_swingup.json `
    --seed 0 `
    --device cuda `
    --total-steps 25000 `
    --eval-episodes 10 `
    --run-dir "runs/week1_real_gpu_20260508/dmc_cartpole_25k/seed0" `
    --overwrite *>&1 |
    Tee-Object -FilePath "reports/week1_real_gpu_20260508/logs/dmc_cartpole_25k_seed0.log"
$Exit = $LASTEXITCODE
$Stamp = Get-Date -Format o
Add-Content -LiteralPath $ProgressLog -Value "$Stamp END dmc_cartpole_25k seed 0 exit $Exit"
exit $Exit
