[CmdletBinding()]
param(
    [string] $Manifest = 'data\processed\sroie-smoke\sroie.jsonl',
    [ValidatePattern('^[a-zA-Z0-9._-]+$')]
    [string] $OutputName = 'sroie-ocr-smoke',
    [ValidateSet('train', 'validation', 'test')]
    [string] $Split = 'test',
    [ValidateSet('ocr_rules', 'vlm_zero_shot', 'vlm_few_shot')]
    [string] $Method = 'ocr_rules',
    [ValidateSet('json_schema', 'compact')]
    [string] $PromptStyle = 'json_schema',
    [string] $ModelId = 'Qwen/Qwen3-VL-2B-Instruct',
    [string] $Examples,
    [ValidateRange(1, 32)]
    [int] $FewShotCount = 2,
    [ValidateRange(1, 1000000)]
    [int] $Limit = 10,
    [ValidateRange(64, 8192)]
    [int] $MaxNewTokens = 1536,
    [switch] $Overwrite
)

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$dataRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'data'))
$artifactsRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'artifacts'))
$manifestPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $Manifest))
if (-not $manifestPath.StartsWith(
    $dataRoot + [IO.Path]::DirectorySeparatorChar,
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw "Le manifeste doit se trouver sous $dataRoot"
}
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Manifeste introuvable : $manifestPath"
}

New-Item -ItemType Directory -Force -Path $artifactsRoot | Out-Null
$outputRoot = Join-Path $artifactsRoot "benchmarks\$OutputName"
if ($Method -ne 'ocr_rules') {
    $python = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Environnement Python introuvable : $python"
    }
    $nativeArguments = @(
        '-m', 'document_intelligence.evaluation.benchmark',
        $manifestPath,
        '--method', $Method,
        '--prompt-style', $PromptStyle,
        '--model-id', $ModelId,
        '--split', $Split,
        '--limit', "$Limit",
        '--max-new-tokens', "$MaxNewTokens",
        '--output-dir', $outputRoot
    )
    if ($Overwrite) {
        $nativeArguments += '--overwrite'
    }
    if ($Method -eq 'vlm_few_shot') {
        if (-not $Examples) {
            throw 'La méthode vlm_few_shot exige -Examples avec un catalogue train.'
        }
        $examplesPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $Examples))
        if (-not (Test-Path -LiteralPath $examplesPath -PathType Leaf)) {
            throw "Catalogue few-shot introuvable : $examplesPath"
        }
        $nativeArguments += @('--examples', $examplesPath, '--few-shot-count', "$FewShotCount")
    }
    & $python @nativeArguments
    exit $LASTEXITCODE
}

$relativeManifest = [IO.Path]::GetRelativePath($dataRoot, $manifestPath).Replace('\', '/')
$dockerArguments = @(
    'compose', 'run', '--rm', '--no-deps',
    '-e', 'PYTHONPATH=/app/src',
    '-v', "${projectRoot}/src:/app/src:ro",
    '-v', "${dataRoot}:/app/data:ro",
    '-v', "${artifactsRoot}:/app/host-artifacts",
    'api', 'python', '-m', 'document_intelligence.evaluation.benchmark',
    "/app/data/$relativeManifest",
    '--method', $Method,
    '--prompt-style', $PromptStyle,
    '--model-id', $ModelId,
    '--split', $Split,
    '--limit', "$Limit",
    '--max-new-tokens', "$MaxNewTokens",
    '--output-dir', "/app/host-artifacts/benchmarks/$OutputName"
)
if ($Overwrite) {
    $dockerArguments += '--overwrite'
}

& (Join-Path $PSScriptRoot 'docker.ps1') @dockerArguments
exit $LASTEXITCODE
