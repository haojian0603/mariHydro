#!/usr/bin/env python3
"""
Rust代码收集器 - 将项目中的所有Rust代码收集到codelog文件夹中的txt文件
使用方法: python collect_code.py [路径] [输出文件名]
"""

import sys
from datetime import datetime
from pathlib import Path


class RustCodeCollector:
    """收集Rust代码并生成带文件树的文档"""

    def __init__(self, root_path="."):
        self.root_path = Path(root_path).resolve()
        if not self.root_path.exists():
            raise FileNotFoundError(f"路径不存在: {self.root_path}")

        self.rust_files = []
        self.file_tree = []
        # 定义输出文件夹名称
        self.output_dir_name = "codelog"
        # 设置输出目录为当前目录下的codelog文件夹
        self.output_dir = Path.cwd() / self.output_dir_name

    def gather_rust_files(self):
        """递归收集所有.rs文件"""
        print(f"🔍 正在扫描: {self.root_path}")

        # 优先处理src目录，然后处理其他rs文件
        src_path = self.root_path / "src"
        if src_path.exists():
            self._scan_directory(src_path)

        # 扫描根目录下的.rs文件（如build.rs, main.rs等）
        for rs_file in self.root_path.glob("*.rs"):
            # 再次确认不在 codelog 中（虽然 glob *.rs 通常只看当前层级，但在某些边缘情况下更安全）
            if self.output_dir_name not in rs_file.parts:
                self.rust_files.append(rs_file)

        # 扫描其他子目录
        for item in self.root_path.iterdir():
            if item.is_dir() and item.name not in [
                "target",
                ".git",
                "node_modules",
                "src",
                self.output_dir_name,  # <--- 修改点：排除 codelog 文件夹
            ]:
                self._scan_directory(item)

        # 排序：按路径字母顺序
        self.rust_files.sort(key=lambda p: str(p.relative_to(self.root_path)))
        print(f"📄 找到 {len(self.rust_files)} 个Rust文件")

    def _scan_directory(self, directory: Path):
        """扫描单个目录"""
        try:
            for path in directory.rglob("*.rs"):
                # <--- 修改点：排除 target 目录和 codelog 输出目录
                # 检查路径的所有部分，确保不包含被排除的目录名
                parts = path.parts
                if "target" not in parts and self.output_dir_name not in parts:
                    self.rust_files.append(path)
        except Exception as e:
            print(f"⚠️ 扫描 {directory} 时出错: {e}")

    def build_file_tree(self):
        """构建文件树结构"""
        if not self.rust_files:
            return "未找到Rust文件\n"

        tree_lines = ["文件树：", "=" * 50, ""]

        # 按相对路径组织
        files_by_dir = {}
        for file_path in self.rust_files:
            try:
                rel_path = file_path.relative_to(self.root_path)
            except ValueError:
                # 如果文件不在root_path下（极少数情况），使用文件名
                rel_path = Path(file_path.name)

            dir_name = rel_path.parent
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(rel_path.name)

        # 生成树形结构
        for dir_path in sorted(files_by_dir.keys()):
            # 根目录处理
            if str(dir_path) == ".":
                for filename in sorted(files_by_dir[dir_path]):
                    tree_lines.append(f"├── {filename}")
            else:
                # 子目录处理
                dir_parts = str(dir_path).split("/")
                for i, part in enumerate(dir_parts):
                    indent = "│   " * i + "├── "
                    pass

                # 简单起见，直接打印完整相对路径的目录头（稍微修改一下原逻辑以适应复杂层级）
                tree_lines.append(f"├── {dir_path}/")

                # 目录下的文件
                for filename in sorted(files_by_dir[dir_path]):
                    indent = "│   " + "    "  # 简单缩进
                    tree_lines.append(f"{indent}├── {filename}")

        tree_lines.extend(["", "=" * 50, ""])
        return "\n".join(tree_lines)

    def collect_to_file(self, output_filename=None):
        """将所有代码收集到codelog文件夹中的单个文件"""
        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)

        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"rust_code_collection_{timestamp}.txt"

        # 完整的输出路径
        output_path = self.output_dir / output_filename

        self.gather_rust_files()
        file_tree_str = self.build_file_tree()

        print(f"💾 正在写入: {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            # 写入头部信息
            f.write(f"""Rust代码收集报告
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
项目路径: {self.root_path}
排除目录: target, .git, node_modules, {self.output_dir_name}
{"=" * 80}

""")

            # 写入文件树
            f.write(file_tree_str)

            # 写入每个文件的代码
            for file_path in self.rust_files:
                try:
                    # 尝试获取相对路径，如果失败则使用绝对路径
                    try:
                        display_path = file_path.relative_to(self.root_path)
                    except ValueError:
                        display_path = file_path

                    content = file_path.read_text(encoding="utf-8")

                    # 写入文件头
                    f.write(f"# File: {display_path}\n\n")
                    f.write("```rust\n")

                    # 如果文件为空，添加提示
                    if not content.strip():
                        f.write("// 文件为空\n")
                    else:
                        # 移除末尾的换行，避免重复
                        content = content.rstrip()
                        f.write(content)

                    f.write("\n```\n\n")

                    print(f"  ✓ 已记录: {display_path}")

                except Exception as e:
                    print(f"  ✗ 读取失败 {file_path}: {e}")
                    f.write(f"# File: {file_path.name}\n\n")
                    f.write("```rust\n// 读取失败\n```\n\n")

        print(f"✅ 完成！共记录 {len(self.rust_files)} 个文件")
        print(f"📂 输出文件: {output_path}")
        return output_path


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="收集Rust项目中的所有代码到codelog文件夹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python collect_code.py                    # 扫描当前目录
  python collect_code.py /path/to/project   # 扫描指定路径
  python collect_code.py . my_code.txt      # 指定输出文件名
        """,
    )

    parser.add_argument(
        "path", nargs="?", default=".", help="要扫描的Rust项目路径（默认: 当前目录）"
    )
    parser.add_argument("output", nargs="?", help="输出文件名（默认: 自动生成）")

    args = parser.parse_args()

    try:
        collector = RustCodeCollector(args.path)
        collector.collect_to_file(args.output)
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
