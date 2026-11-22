[CmdletBinding()]
param(
    [string] $Manifest = 'data\processed\sroie-full\sroie.jsonl',
    [string] $DataRoot = 'data\processed\sroie-full',
    [string] $OutputDir = 'data\processed\sroie-full\sft',
    [ValidateSet('compact', 'json_schema')]
    [string] $PromptStyle = 'compact'
)

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Environnement Python introuvable : $python"
}

& $python -m document_intelligence.finetuning.dataset `
    (Join-Path $projectRoot $Manifest) `
    --data-root (Join-Path $projectRoot $DataRoot) `
    --output-dir (Join-Path $projectRoot $OutputDir) `
    --prompt-style $PromptStyle
exit $LASTEXITCODE
