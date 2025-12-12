#!/bin/bash
# marihydro/scripts/check_hardcoded_f64.sh
#
# CI 守护脚本：检测硬编码 f64 类型
#
# 本脚本扫描核心计算代码中的硬编码 f64，确保所有数值类型都通过 Scalar 泛型
# 或 Precision 枚举进行管理。
#
# 允许的例外：
# - mh_core/src/scalar.rs - Scalar trait 实现文件
# - mh_core/src/precision.rs - Precision 枚举定义
# - mh_mesh/** - 几何库（坐标存储需要 f64 精度）
# - mh_geo/** - 地理库（坐标转换需要 f64 精度）
# - *_test.rs, *_tests.rs - 测试文件中的常量
# - docs/*.md - 文档中的说明

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 定义需要严格扫描的核心目录（Layer 3 引擎层）
SCAN_DIRS=(
    "crates/mh_physics/src/engine"
    "crates/mh_physics/src/flux"
    "crates/mh_physics/src/boundary"
    "crates/mh_physics/src/numerics"
    "crates/mh_physics/src/sources"
)

# 定义排除的文件模式
EXCLUDE_PATTERNS=(
    "scalar.rs"          # Scalar trait 定义文件
    "precision.rs"       # Precision 枚举定义文件
    "_test.rs"           # 测试文件
    "_tests.rs"          # 测试文件
    "test_"              # 测试模块
)

echo "=== Checking for hardcoded f64 types ==="
echo "Project root: $PROJECT_ROOT"
echo ""

FOUND_ISSUES=0

for dir in "${SCAN_DIRS[@]}"; do
    full_dir="$PROJECT_ROOT/$dir"
    if [[ ! -d "$full_dir" ]]; then
        echo "Warning: Directory not found: $full_dir"
        continue
    fi

    # 查找所有 .rs 文件
    while IFS= read -r -d '' file; do
        # 检查是否在排除列表中
        skip=false
        for pattern in "${EXCLUDE_PATTERNS[@]}"; do
            if [[ "$file" == *"$pattern"* ]]; then
                skip=true
                break
            fi
        done
        [[ $skip == true ]] && continue

        # 搜索硬编码的 f64 模式
        # 匹配: `: f64`, `as f64`, `[f64;`, `Vec<f64>` 等
        matches=$(grep -n -E '(:\s*f64\b|as\s+f64\b|\[f64;|Vec<f64>)' "$file" 2>/dev/null || true)
        
        if [[ -n "$matches" ]]; then
            # 进一步过滤掉注释行
            while IFS= read -r line; do
                # 跳过注释（// 在 f64 之前）
                if [[ "$line" =~ // ]]; then
                    comment_pos=$(echo "$line" | grep -b -o "//" | head -1 | cut -d: -f1)
                    f64_pos=$(echo "$line" | grep -b -o "f64" | head -1 | cut -d: -f1)
                    if [[ -n "$comment_pos" ]] && [[ -n "$f64_pos" ]] && [[ "$comment_pos" -lt "$f64_pos" ]]; then
                        continue
                    fi
                fi
                
                echo "ISSUE: $file"
                echo "  $line"
                echo ""
                FOUND_ISSUES=$((FOUND_ISSUES + 1))
            done <<< "$matches"
        fi
    done < <(find "$full_dir" -name "*.rs" -type f -print0)
done

echo ""
echo "=== Summary ==="
if [[ $FOUND_ISSUES -eq 0 ]]; then
    echo "✅ No hardcoded f64 issues found in Layer 3 Engine!"
    exit 0
else
    echo "❌ Found $FOUND_ISSUES hardcoded f64 issues"
    echo "Please use Scalar<S> generic type or add // ALLOW_F64: <原因> comment"
    exit 1
fi