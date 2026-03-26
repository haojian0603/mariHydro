#!/usr/bin/env pwsh
#
# Blocking guard for IO pipeline shutdown truthfulness.
# Ensures flush/wait/shutdown semantics do not collapse errors into bools,
# ignored results, or warning-only pseudo-success paths.
#

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Errors = @()

function Add-Failure {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
    $script:Errors += $Message
}

function Check-PatternAbsent {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required pipeline file: $RelativePath"
        return
    }

    $matches = Select-String -Path $absolutePath -Pattern $Pattern -CaseSensitive
    if ($matches) {
        Add-Failure $Message
        $matches | Select-Object -First 5 | ForEach-Object {
            Write-Host ("  " + $_.Path + ":" + $_.LineNumber + ": " + $_.Line.Trim()) -ForegroundColor Red
        }
    } else {
        Write-Host "[OK] $Message" -ForegroundColor Green
    }
}

function Check-PatternPresent {
    param(
        [string]$RelativePath,
        [string]$Pattern,
        [string]$Message
    )

    $absolutePath = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $absolutePath)) {
        Add-Failure "missing required pipeline file: $RelativePath"
        return
    }

    $matches = Select-String -Path $absolutePath -Pattern $Pattern -CaseSensitive
    if ($matches) {
        Write-Host "[OK] $Message" -ForegroundColor Green
    } else {
        Add-Failure $Message
    }
}

Push-Location $ProjectRoot
try {
    Write-Host "=== Checking IO pipeline shutdown contracts ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"
    Write-Host ""

    $pipelinePath = "crates/mh_io/src/pipeline.rs"

    Check-PatternAbsent `
        -RelativePath $pipelinePath `
        -Pattern 'pub fn wait_for_completion\(&self, timeout: Duration\) -> bool' `
        -Message "pipeline wait_for_completion must not expose bool timeout semantics"

    Check-PatternAbsent `
        -RelativePath $pipelinePath `
        -Pattern 'pub fn shutdown\(&mut self\)\s*\{' `
        -Message "pipeline shutdown must not hide failure behind a unit return type"

    Check-PatternAbsent `
        -RelativePath $pipelinePath `
        -Pattern 'pub fn shutdown_graceful\(&mut self, timeout: Duration\)\s*\{' `
        -Message "pipeline graceful shutdown must not hide failure behind a unit return type"

    Check-PatternAbsent `
        -RelativePath $pipelinePath `
        -Pattern 'pub fn shutdown_immediate\(&mut self\)\s*\{' `
        -Message "pipeline immediate shutdown must not hide failure behind a unit return type"

    Check-PatternAbsent `
        -RelativePath $pipelinePath `
        -Pattern 'let _ = self\.flush\(\)|let _ = self\.wait_for_completion\(timeout\)|let _ = self\.sender\.send\(OutputRequest::Shutdown\)|if let Ok\(mut stats\) = self\.stats\.lock\(\)' `
        -Message "pipeline shutdown path must not ignore flush, wait, send, or stats failures"

    Check-PatternAbsent `
        -RelativePath $pipelinePath `
        -Pattern '写入超时警告|PipelineError::Timeout\(Duration::from_millis\(write_timeout_ms\)\)\s*\.into_io_error\("process_request"\)' `
        -Message "pipeline write timeout must not degrade to a warning-only pseudo-success"

    Check-PatternPresent `
        -RelativePath $pipelinePath `
        -Pattern 'pub fn wait_for_completion\(&self, timeout: Duration\) -> crate::error::IoResult<\(\)>' `
        -Message "pipeline wait_for_completion must expose explicit IoResult timeout semantics"

    Check-PatternPresent `
        -RelativePath $pipelinePath `
        -Pattern 'pub fn shutdown\(&mut self\) -> crate::error::IoResult<\(\)>|pub fn shutdown_graceful\(&mut self, timeout: Duration\) -> crate::error::IoResult<\(\)>|pub fn shutdown_immediate\(&mut self\) -> crate::error::IoResult<\(\)>' `
        -Message "pipeline shutdown APIs must return explicit IoResult values"

    Check-PatternPresent `
        -RelativePath $pipelinePath `
        -Pattern 'fn send_shutdown_request\(&self, stage: &str\) -> crate::error::IoResult<\(\)>|fn join_worker\(&mut self, stage: &str\) -> crate::error::IoResult<\(\)>' `
        -Message "pipeline shutdown implementation must use explicit send/join helpers"

    Check-PatternPresent `
        -RelativePath $pipelinePath `
        -Pattern 'pending requests did not drain within|test_wait_for_completion_timeout_is_explicit|test_pipeline_shutdown' `
        -Message "pipeline shutdown path must keep explicit timeout text and regression coverage"

    if ($Errors.Count -eq 0) {
        Write-Host ""
        Write-Host "[OK] IO pipeline shutdown contracts passed" -ForegroundColor Green
        exit 0
    }

    Write-Host ""
    Write-Host "[FAIL] IO pipeline shutdown contracts failed" -ForegroundColor Red
    Write-Host "Failed items: $($Errors -join ', ')" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
