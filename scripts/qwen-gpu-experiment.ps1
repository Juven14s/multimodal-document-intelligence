[CmdletBinding()]
param(
    [string] $Manifest = 'data\processed\sroie-full\sroie.jsonl',
    [string] $Examples = 'data\processed\sroie-full\fewshot.jsonl',
    [string] $OutputDir = 'artifacts\qwen-experiment',
    [string] $ModelId = 'Qwen/Qwen3-VL-2B-Instruct',
    [ValidateRange(1, 1000000)]
    [int] $Limit = 10,
    [ValidateRange(1, 32)]
    [int] $FewShotCount = 2,
    [ValidateRange(64, 8192)]
    [int] $MaxNewTokens = 1536,
    [switch] $Overwrite,
    [switch] $AllowCpu
)

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Environnement Python introuvable : $python"
}

$arguments = @(
    '-m', 'document_intelligence.evaluation.qwen_experiment',
    (Join-Path $projectRoot $Manifest),
    '--examples', (Join-Path $projectRoot $Examples),
    '--output-dir', (Join-Path $projectRoot $OutputDir),
    '--model-id', $ModelId,
    '--limit', "$Limit",
    '--few-shot-count', "$FewShotCount",
    '--max-new-tokens', "$MaxNewTokens"
)
if ($Overwrite) {
    $arguments += '--overwrite'
}
if ($AllowCpu) {
    $arguments += '--allow-cpu'
}

& $python @arguments
exit $LASTEXITCODE
