# Unified verification sweep for all 10 @aartii.py ML & Python projects
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyML = Join-Path $root ".venv-ml\Scripts\python.exe"
$py2  = Join-Path $root ".venv-ch2\Scripts\python.exe"
$py3  = Join-Path $root ".venv-ch3\Scripts\python.exe"
$env:CUDA_VISIBLE_DEVICES = ""

$jobs = @(
  @{ id="01"; dir="01_instagram_two_tower";       py=$pyML; req="reports\benchmark.json" },
  @{ id="02"; dir="02_tiny_model_lora_banking77"; py=$pyML; req="reports\eval_results.json" },
  @{ id="03"; dir="03_agent_harness";             py=$py2;  req="traces\eval_summary.json" },
  @{ id="04"; dir="04_quantize_your_own_llm";     py=$pyML; req="reports\quant_benchmark.json" },
  @{ id="05"; dir="05_neural_pong_world_model";   py=$pyML; req="reports\eval_rollout.json" },
  @{ id="06"; dir="06_personal_rag_assistant";    py=$py3;  req="reports\rag_eval.json" },
  @{ id="07"; dir="07_ai_research_agent";         py=$py2;  req="reports\brief.md" },
  @{ id="08"; dir="08_discord_bot";               py=$py3;  req="Dockerfile" },
  @{ id="09"; dir="09_twenty_one_game";           py=$py3;  req="reports\simulation_1000.json" },
  @{ id="10"; dir="10_multi_agent_system";        py=$py2;  req="reports\execution_trace.json" }
)

$allPass = $true
$totalPassed = 0
foreach ($j in $jobs) {
  $p = Join-Path $root $j.dir
  $reqOk = Test-Path (Join-Path $p $j.req)
  Push-Location $p
  $env:PYTHONPATH = $p
  $out = & $j.py -m pytest -o addopts="" -W ignore tests 2>&1
  $rc = $LASTEXITCODE
  Pop-Location
  $lineObj = $out | Select-String -Pattern "\d+ passed|\d+ failed|\d+ error" | Select-Object -Last 1
  $summary = if ($lineObj) { $lineObj.ToString().Trim() } else { ($out | Select-Object -Last 2) -join " | " }
  if ($summary -match "(\d+) passed") { $totalPassed += [int]$Matches[1] }
  if ($rc -ne 0 -or -not $reqOk) { $allPass = $false }
  Write-Output ("[{0}] {1,-30} | rc={2} | req={3} | {4}" -f $j.id, $j.dir, $rc, $reqOk, $summary)
}
Write-Output "=== ALL_PASS: $allPass | TOTAL_PASSED: $totalPassed ==="
if (-not $allPass) { exit 1 }
