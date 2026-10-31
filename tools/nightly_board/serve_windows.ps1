param(
    [int]$Port = 8766,
    [string]$Bind = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$PublicDir = Join-Path $PSScriptRoot "public"
if (-not (Test-Path (Join-Path $PublicDir "index.html"))) {
    throw "public/index.html 不存在，请先从数据分支拉取最新快照。"
}

$Python = Get-Command py -ErrorAction SilentlyContinue
if ($Python) {
    & py -3 -m http.server $Port --bind $Bind --directory $PublicDir
} else {
    & python -m http.server $Port --bind $Bind --directory $PublicDir
}
