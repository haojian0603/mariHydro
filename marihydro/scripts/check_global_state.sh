#!/bin/bash
# marihydro/scripts/check_global_state.sh
#
# CI 守护脚本：检测全局可变状态
#
# 本脚本扫描代码中的全局可变状态（static mut, lazy_static, thread_local!）
# 以确保项目遵循无全局状态的设计原则。
#
# 允许的例外：
# - 日志初始化（log, tracing）
# - 测试桩（test fixtures）
# - 显式标注的 @global-state-ok 注释

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 定义扫描的核心目录
SCAN_DIRS=(
    "crates/mh_core/src"
    "crates/mh_physics/src"
    "crates/mh_mesh/src"
    "crates/mh_geo/src"
    "crates/mh_foundation/src"
    "crates/mh_workflow/src"
    "crates/mh_io/src"
    "crates/mh_terrain/src"
)

echo "=== Checking for global mutable state ==="
echo "Project root: $PROJECT_ROOT"
echo ""

FOUND_ISSUES=0

# 检测模式
PATTERNS=(
    "static\s+mut"           # static mut 变量
    "lazy_static!"           # lazy_static 宏
    "thread_local!"          # thread_local 宏
    "OnceCell::new\(\)"      # 全局 OnceCell
    "OnceLock::new\(\)"      # 全局 OnceLock
    "Lazy::new"              # once_cell::Lazy
)

# 合并为一个正则表达式
COMBINED_PATTERN=$(IFS="|"; echo "${PATTERNS[*]}")

for dir in "${SCAN_DIRS[@]}"; do
    full_dir="$PROJECT_ROOT/$dir"
    if [[ ! -d "$full_dir" ]]; then
        continue
    fi

    while IFS= read -r -d '' file; do
        # 跳过测试文件
        if [[ "$file" == *"_test.rs" ]] || [[ "$file" == *"/tests/"* ]]; then
            continue
        fi

        matches=$(grep -n -E "($COMBINED_PATTERN)" "$file" 2>/dev/null || true)
        
        if [[ -n "$matches" ]]; then
            while IFS= read -r line; do
                # 检查是否有例外标注
                if [[ "$line" == *"@global-state-ok"* ]]; then
                    continue
                fi
                # 跳过注释行
                trimmed=$(echo "$line" | sed 's/^[0-9]*://' | sed 's/^[[:space:]]*//')
                if [[ "$trimmed" == "//"* ]] || [[ "$trimmed" == "/*"* ]] || [[ "$trimmed" == "*"* ]]; then
                    continue
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
    echo "✅ No global mutable state issues found!"
    exit 0
else
    echo "❌ Found $FOUND_ISSUES global mutable state issues"
    echo "Consider using dependency injection or explicit parameter passing."
    echo "If intentional, add // @global-state-ok comment."
    exit 1
fi
