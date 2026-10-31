param(
    [int]$Port = 8766,
    [string]$Bind = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$PublicDir = Join-Path $PSScriptRoot "public"
if (-not (Test-Path (Join-Path $PublicDir "index.html"))) {
    throw "public/index.html was not found. Pull the latest snapshot first."
}

$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
$Python3 = Get-Command python3 -ErrorAction SilentlyContinue
$Python = Get-Command python -ErrorAction SilentlyContinue
if ($PyLauncher) {
    & py -3 -m http.server $Port --bind $Bind --directory $PublicDir
} elseif ($Python3) {
    & python3 -m http.server $Port --bind $Bind --directory $PublicDir
} elseif ($Python) {
    & python -m http.server $Port --bind $Bind --directory $PublicDir
} else {
    throw "Python 3 was not found. Install Python 3 and try again."
}
