#!/usr/bin/env pwsh
# MariHydro 架构验证脚本
# 用于验证层级依赖和代码规范

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " MariHydro 架构验证" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$errors = @()

# =============================================================================
# Phase 1: Layer 依赖验证
# =============================================================================

Write-Host "`n=== Phase 1: Layer 依赖验证 ===" -ForegroundColor Cyan

# Foundation 零依赖（不依赖任何 mh_* 模块）
Write-Host "`n检查 mh_foundation 零依赖..." -ForegroundColor Yellow
$foundationDeps = cargo tree -p mh_foundation --edges normal 2>&1 | Select-String "mh_" | Where-Object { $_ -notmatch "^mh_foundation" }
if ($foundationDeps) {
    Write-Host "❌ mh_foundation 存在内部依赖:" -ForegroundColor Red
    $foundationDeps | ForEach-Object { Write-Host "   $_" -ForegroundColor Red }
    $errors += "mh_foundation 不应依赖其他 mh_* 模块"
} else {
    Write-Host "✓ mh_foundation 零依赖" -ForegroundColor Green
}

# Runtime 只依赖 Foundation
Write-Host "`n检查 mh_runtime 依赖..." -ForegroundColor Yellow
$runtimeResult = cargo tree -p mh_runtime --depth 1 2>&1
if ($LASTEXITCODE -eq 0) {
    $runtimeDeps = $runtimeResult | Select-String "mh_" | Where-Object { $_ -notmatch "mh_foundation" -and $_ -notmatch "^mh_runtime" }
    if ($runtimeDeps) {
        Write-Host "❌ mh_runtime 存在非法依赖:" -ForegroundColor Red
        $runtimeDeps | ForEach-Object { Write-Host "   $_" -ForegroundColor Red }
        $errors += "mh_runtime 只能依赖 mh_foundation"
    } else {
        Write-Host "✓ mh_runtime 依赖正确" -ForegroundColor Green
    }
} else {
    Write-Host "⚠ mh_runtime 尚未创建或无法编译" -ForegroundColor Yellow
}

# Config 只依赖 Runtime
Write-Host "`n检查 mh_config 依赖..." -ForegroundColor Yellow
$configResult = cargo tree -p mh_config --depth 1 2>&1
if ($LASTEXITCODE -eq 0) {
    $configDeps = $configResult | Select-String "mh_" | Where-Object { $_ -notmatch "mh_runtime" -and $_ -notmatch "^mh_config" }
    if ($configDeps) {
        Write-Host "❌ mh_config 存在非法依赖:" -ForegroundColor Red
        $configDeps | ForEach-Object { Write-Host "   $_" -ForegroundColor Red }
        $errors += "mh_config 只能依赖 mh_runtime"
    } else {
        Write-Host "✓ mh_config 依赖正确" -ForegroundColor Green
    }
} else {
    Write-Host "⚠ mh_config 尚未创建或无法编译" -ForegroundColor Yellow
}

# =============================================================================
# Phase 2: Legacy 残留检测
# =============================================================================

Write-Host "`n=== Phase 2: Legacy 残留检测 ===" -ForegroundColor Cyan

# 检测类型别名 (pub type Xxx = XxxGeneric<f64>)
Write-Host "`n检查 Legacy 类型别名..." -ForegroundColor Yellow
$typeAliases = rg "pub type \w+ = \w+Generic<f64>;" crates/ --type rust 2>&1
if ($typeAliases -and $typeAliases -notmatch "^$" -and $LASTEXITCODE -eq 0) {
    Write-Host "❌ 存在 Legacy 类型别名:" -ForegroundColor Red
    Write-Host $typeAliases -ForegroundColor Red
    $errors += "存在 Legacy 类型别名"
} else {
    Write-Host "✓ 无 Legacy 类型别名" -ForegroundColor Green
}

# 检查 mh_core 是否已删除
Write-Host "`n检查 mh_core 状态..." -ForegroundColor Yellow
if (Test-Path "crates/mh_core") {
    $coreFiles = Get-ChildItem -Path "crates/mh_core/src" -Filter "*.rs" -ErrorAction SilentlyContinue
    if ($coreFiles) {
        Write-Host "⚠ mh_core 仍存在，待迁移删除" -ForegroundColor Yellow
    } else {
        Write-Host "✓ mh_core 已清空或不存在" -ForegroundColor Green
    }
} else {
    Write-Host "✓ mh_core 已删除" -ForegroundColor Green
}

# =============================================================================
# Phase 3: Config 层无泛型验证
# =============================================================================

Write-Host "`n=== Phase 3: Config 层无泛型验证 ===" -ForegroundColor Cyan

Write-Host "`n检查 mh_config 无泛型..." -ForegroundColor Yellow
if (Test-Path "crates/mh_config/src") {
    $configGenerics = rg "<.*:.*Backend" crates/mh_config/src/ --type rust 2>&1
    if ($configGenerics -and $LASTEXITCODE -eq 0) {
        Write-Host "❌ mh_config 存在 Backend 泛型:" -ForegroundColor Red
        Write-Host $configGenerics -ForegroundColor Red
        $errors += "mh_config 不应包含 Backend 泛型"
    } else {
        Write-Host "✓ mh_config 无 Backend 泛型" -ForegroundColor Green
    }
} else {
    Write-Host "⚠ mh_config/src 尚未创建" -ForegroundColor Yellow
}

# =============================================================================
# Phase 4: Indices 无代际验证
# =============================================================================

Write-Host "`n=== Phase 4: Indices 无代际验证 ===" -ForegroundColor Cyan

Write-Host "`n检查 mh_runtime indices 无代际..." -ForegroundColor Yellow
if (Test-Path "crates/mh_runtime/src/indices.rs") {
    $indicesGen = rg "generation" crates/mh_runtime/src/indices.rs 2>&1
    if ($indicesGen -and $LASTEXITCODE -eq 0) {
        Write-Host "❌ indices.rs 包含 generation:" -ForegroundColor Red
        Write-Host $indicesGen -ForegroundColor Red
        $errors += "mh_runtime/indices.rs 不应包含代际字段"
    } else {
        Write-Host "✓ indices.rs 无代际字段" -ForegroundColor Green
    }
} else {
    Write-Host "⚠ mh_runtime/src/indices.rs 尚未创建" -ForegroundColor Yellow
}

# =============================================================================
# Phase 5: 编译验证
# =============================================================================

Write-Host "`n=== Phase 5: 编译验证 ===" -ForegroundColor Cyan

Write-Host "`n运行 cargo check..." -ForegroundColor Yellow
$checkResult = cargo check --workspace 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 编译失败" -ForegroundColor Red
    $errors += "cargo check 失败"
} else {
    Write-Host "✓ 编译通过" -ForegroundColor Green
}

# =============================================================================
# 结果汇总
# =============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
if ($errors.Count -eq 0) {
    Write-Host "✅ 架构验证通过！" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ 架构验证失败，发现 $($errors.Count) 个问题：" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
