$ErrorActionPreference = "Stop"

Set-Location "C:\Users\Ala\Desktop\Project 15"
$env:PYTHONPATH = "src"

$RunRoot = "runs\pendulum_investigation_20260509"
$ReportRoot = "reports\pendulum_investigation_20260509"
$LogDir = Join-Path $ReportRoot "logs"
$Condition = "pendulum_250k_utd2_buffer500k"
$ConditionRunRoot = Join-Path $RunRoot $Condition
$ConditionReportRoot = Join-Path $ReportRoot $Condition
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ConditionReportRoot | Out-Null

function Invoke-CheckedPython {
    param(
        [string[]] $Arguments,
        [string] $LogPath
    )
    $timestamp = Get-Date -Format o
    "[$timestamp] python $($Arguments -join ' ')" | Tee-Object -FilePath $LogPath -Append
    & python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "python exited with code $LASTEXITCODE for: $($Arguments -join ' ')"
    }
}

function Invoke-CheckedPythonJson {
    param(
        [string[]] $Arguments,
        [string] $LogPath,
        [string] $JsonPath
    )
    $timestamp = Get-Date -Format o
    "[$timestamp] python $($Arguments -join ' ')" | Tee-Object -FilePath $LogPath -Append
    $Output = & python @Arguments 2>&1
    $Output | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "python exited with code $LASTEXITCODE for: $($Arguments -join ' ')"
    }
    $Output | Set-Content -Path $JsonPath -Encoding UTF8
}

foreach ($Seed in @(2, 3, 4)) {
    $RunDir = Join-Path $ConditionRunRoot "seed$Seed"
    $LogPath = Join-Path $LogDir "$Condition`_seed$Seed.log"
    Invoke-CheckedPython -LogPath $LogPath -Arguments @(
        "-m", "last_nine_rl.train",
        "--config", "configs\week1_pendulum.json",
        "--seed", "$Seed",
        "--run-dir", "$RunDir",
        "--total-steps", "250000",
        "--buffer-size", "500000",
        "--updates-per-step", "2",
        "--eval-every-steps", "100000",
        "--eval-episodes", "20",
        "--log-interval", "5000",
        "--replay-inspection-interval", "25000",
        "--diagnostics-interval", "25000",
        "--device", "cuda",
        "--save-replay",
        "--overwrite"
    )
}

Invoke-CheckedPythonJson `
    -LogPath (Join-Path $LogDir "$Condition`_aggregate.log") `
    -JsonPath (Join-Path $ConditionReportRoot "aggregate.json") `
    -Arguments @(
    "-m", "last_nine_rl.aggregate",
    "--runs", "$ConditionRunRoot",
    "--thresholds", "-250", "-200", "-150", "-100"
)

Invoke-CheckedPython -LogPath (Join-Path $LogDir "$Condition`_compare.log") -Arguments @(
    "-m", "last_nine_rl.compare",
    "--runs", "$ConditionRunRoot",
    "--out", (Join-Path $ConditionReportRoot "compare")
)

Invoke-CheckedPython -LogPath (Join-Path $LogDir "$Condition`_posthoc_1000eps.log") -Arguments @(
    "-m", "last_nine_rl.posthoc_eval",
    "--runs", "$ConditionRunRoot",
    "--out", (Join-Path $ConditionReportRoot "posthoc_1000eps"),
    "--episodes", "1000",
    "--seed-base", "200000",
    "--device", "cpu"
)

Invoke-CheckedPython -LogPath (Join-Path $LogDir "$Condition`_grid_reset_support.log") -Arguments @(
    "-m", "last_nine_rl.pendulum_grid",
    "--runs", "$ConditionRunRoot",
    "--out", (Join-Path $ConditionReportRoot "grid_reset_support_61x41"),
    "--theta-bins", "61",
    "--velocity-bins", "41",
    "--velocity-limit", "1.0",
    "--device", "cpu"
)

Invoke-CheckedPython -LogPath (Join-Path $LogDir "$Condition`_relative_success.log") -Arguments @(
    "-m", "last_nine_rl.pendulum_relative",
    "--condition-label", "Pendulum SAC 250k UTD2",
    "--sac-rollouts", (Join-Path $ConditionReportRoot "grid_reset_support_61x41\pendulum_grid_rollouts.csv"),
    "--dp-grid", "reports\pendulum_investigation_20260509\pendulum_dp_100k_reset_support_241x161x81\pendulum_dp_grid.csv",
    "--controller-grid", "reports\pendulum_investigation_20260509\pendulum_controller_reset_support_61x41\controller_grid.csv",
    "--out", "reports\pendulum_investigation_20260509\relative_success_250k_utd2",
    "--epsilon-return", "5.0"
)

Invoke-CheckedPython -LogPath (Join-Path $LogDir "$Condition`_replay_diagnostics.log") -Arguments @(
    "-m", "last_nine_rl.replay_diagnostics_report",
    "--condition", "100k_utd1=runs\week1_real_gpu_20260509\pendulum_100k",
    "--condition", "500k_utd1=runs\pendulum_investigation_20260509\pendulum_500k_utd1_buffer500k",
    "--condition", "250k_utd2=runs\pendulum_investigation_20260509\pendulum_250k_utd2_buffer500k",
    "--out", "reports\pendulum_investigation_20260509\replay_diagnostics_comparison_250k_utd2"
)

"[$(Get-Date -Format o)] completed $Condition resume and reports" |
    Tee-Object -FilePath (Join-Path $LogDir "$Condition`_resume.done.log") -Append
