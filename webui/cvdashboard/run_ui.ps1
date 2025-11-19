<#
Run the static UI locally from the `webui/cvdashboard` folder.

Usage (from repo root):
  .\webui\cvdashboard\run_ui.ps1

This will serve files at http://127.0.0.1:8000
#>
param()

Set-Location -Path "$PSScriptRoot"

Write-Host "Serving UI at http://127.0.0.1:8000 ..."
py -3 -m http.server 8000
