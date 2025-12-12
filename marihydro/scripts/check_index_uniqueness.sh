#!/bin/bash
# marihydro/scripts/check_index_uniqueness.sh
#
# CI 守护脚本：检测重复索引类型定义
#
# 本脚本确保 CellIndex, FaceIndex, NodeIndex 等索引类型仅在 mh_core 中定义，
# 防止在其他 crate 中出现重复定义。
#
# 统一定义位置：mh_core/src/indices.rs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Checking for duplicate index type definitions ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# 需要检查的索引类型
INDEX_TYPES=(
    "CellIndex"
    "FaceIndex"
    "NodeIndex"
    "VertexIndex"
    "HalfEdgeIndex"
    "BoundaryIndex"
)

# 唯一允许定义的文件
ALLOWED_FILE="crates/mh_core/src/indices.rs"

FOUND_ISSUES=0

for idx_type in "${INDEX_TYPES[@]}"; do
    echo "Checking: $idx_type"
    
    # 搜索 struct 定义模式: `pub struct CellIndex` 或 `struct CellIndex`
    # 排除 use 语句和类型别名
    matches=$(grep -r -n -E "^\s*(pub\s+)?struct\s+$idx_type\b" "$PROJECT_ROOT/crates" 2>/dev/null || true)
    
    if [[ -n "$matches" ]]; then
        while IFS= read -r line; do
            # 提取文件路径
            file=$(echo "$line" | cut -d: -f1)
            rel_file=${file#$PROJECT_ROOT/}
            
            # 检查是否是允许的定义位置
            if [[ "$rel_file" == "$ALLOWED_FILE" ]]; then
                echo "  ✓ Canonical definition in $rel_file"
            else
                echo "  ✗ Duplicate definition in $rel_file"
                echo "    $line"
                FOUND_ISSUES=$((FOUND_ISSUES + 1))
            fi
        done <<< "$matches"
    else
        echo "  Warning: No definition found for $idx_type"
    fi
    echo ""
done

# 额外检查 type alias 重定义
echo "Checking for conflicting type aliases..."
for idx_type in "${INDEX_TYPES[@]}"; do
    # 搜索 type alias: `type CellIndex = ...`
    aliases=$(grep -r -n -E "^\s*(pub\s+)?type\s+$idx_type\s*=" "$PROJECT_ROOT/crates" 2>/dev/null || true)
    
    if [[ -n "$aliases" ]]; then
        while IFS= read -r line; do
            file=$(echo "$line" | cut -d: -f1)
            rel_file=${file#$PROJECT_ROOT/}
            
            # 类型别名在迁移期间可能存在，但应该逐步移除
            if [[ "$rel_file" != "$ALLOWED_FILE" ]]; then
                echo "  Warning: Type alias for $idx_type in $rel_file"
                echo "    Consider migrating to mh_core::$idx_type"
            fi
        done <<< "$aliases"
    fi
done

echo ""
echo "=== Summary ==="
if [[ $FOUND_ISSUES -eq 0 ]]; then
    echo "✅ No duplicate index type definitions found!"
    exit 0
else
    echo "❌ Found $FOUND_ISSUES duplicate index type definitions"
    echo "Please remove duplicates and use mh_core::{CellIndex, FaceIndex, ...}"
    exit 1
fi
