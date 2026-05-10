Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $Root

$ProgressLog = "reports/week1_real_gpu_20260509/logs/dmc_cartpole_100k_progress.log"
foreach ($Seed in 0..2) {
    $Stamp = Get-Date -Format o
    Add-Content -LiteralPath $ProgressLog -Value "$Stamp START dmc_cartpole_100k seed $Seed"
    & python -m last_nine_rl.train `
        --config configs/week1_cartpole_swingup.json `
        --seed $Seed `
        --device cuda `
        --total-steps 100000 `
        --eval-every-steps 25000 `
        --eval-episodes 5 `
        --run-dir "runs/week1_real_gpu_20260509/dmc_cartpole_100k/seed$Seed" `
        --overwrite *>&1 |
        Tee-Object -FilePath "reports/week1_real_gpu_20260509/logs/dmc_cartpole_100k_seed$Seed.log"
    $Exit = $LASTEXITCODE
    $Stamp = Get-Date -Format o
    Add-Content -LiteralPath $ProgressLog -Value "$Stamp END dmc_cartpole_100k seed $Seed exit $Exit"
    if ($Exit -ne 0) {
        exit $Exit
    }
}
