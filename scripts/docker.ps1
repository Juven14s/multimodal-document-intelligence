$DockerArgs = $args

$resolved = Get-Command docker -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1
$candidates = @(
    if ($resolved) { $resolved.Source }
    if ($env:LOCALAPPDATA) {
        Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
    }
    if ($env:ProgramFiles) {
        Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe'
    }
)
$inaccessible = @()
$docker = $null
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    try {
        if (Test-Path -LiteralPath $candidate -PathType Leaf -ErrorAction Stop) {
            $docker = $candidate
            break
        }
    } catch {
        $inaccessible += $candidate
    }
}

if (-not $docker) {
    if ($inaccessible.Count -gt 0) {
        throw "Accès au client Docker refusé par cette session. Chemins non vérifiables : $($inaccessible -join ', ')"
    }
    throw 'Client Docker introuvable dans le PATH et les emplacements standards de Docker Desktop.'
}

& $docker @DockerArgs
exit $LASTEXITCODE
