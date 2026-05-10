$ErrorActionPreference = "Stop"

Set-Location "C:\Users\Ala\Desktop\Project 15"
$env:PYTHONPATH = "src"

$RunRoot = "runs\pendulum_investigation_20260509"
$ReportRoot = "reports\pendulum_investigation_20260509"
$LogDir = Join-Path $ReportRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

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

function Invoke-TrainCondition {
    param(
        [string] $Condition,
        [int[]] $Seeds,
        [int] $TotalSteps,
        [int] $UpdatesPerStep,
        [int] $BufferSize
    )
    foreach ($Seed in $Seeds) {
        $RunDir = Join-Path $RunRoot "$Condition\seed$Seed"
        $LogPath = Join-Path $LogDir "$Condition`_seed$Seed.log"
        Invoke-CheckedPython -LogPath $LogPath -Arguments @(
            "-m", "last_nine_rl.train",
            "--config", "configs\week1_pendulum.json",
            "--seed", "$Seed",
            "--run-dir", "$RunDir",
            "--total-steps", "$TotalSteps",
            "--buffer-size", "$BufferSize",
            "--updates-per-step", "$UpdatesPerStep",
            "--eval-every-steps", "100000",
            "--eval-episodes", "20",
            "--log-interval", "5000",
            "--replay-inspection-interval", "25000",
            "--diagnostics-interval", "25000",
            "--device", "cuda",
            "--overwrite"
        )
    }
}

function Invoke-ConditionReports {
    param([string] $Condition)

    $ConditionRunRoot = Join-Path $RunRoot $Condition
    $ConditionReportRoot = Join-Path $ReportRoot $Condition
    New-Item -ItemType Directory -Force -Path $ConditionReportRoot | Out-Null

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
}

$conditions = @(
    @{
        Name = "pendulum_500k_utd1_buffer500k"
        Seeds = @(0, 1, 2, 3, 4)
        TotalSteps = 500000
        UpdatesPerStep = 1
        BufferSize = 500000
    },
    @{
        Name = "pendulum_250k_utd2_buffer500k"
        Seeds = @(0, 1, 2, 3, 4)
        TotalSteps = 250000
        UpdatesPerStep = 2
        BufferSize = 500000
    },
    @{
        Name = "pendulum_500k_utd2_buffer500k"
        Seeds = @(0, 1, 2)
        TotalSteps = 500000
        UpdatesPerStep = 2
        BufferSize = 500000
    }
)

foreach ($condition in $conditions) {
    Invoke-TrainCondition `
        -Condition $condition.Name `
        -Seeds $condition.Seeds `
        -TotalSteps $condition.TotalSteps `
        -UpdatesPerStep $condition.UpdatesPerStep `
        -BufferSize $condition.BufferSize
    Invoke-ConditionReports -Condition $condition.Name
}

"[$(Get-Date -Format o)] completed all Pendulum investigation conditions" | Tee-Object -FilePath (Join-Path $LogDir "run_pendulum_investigation.done.log") -Append
