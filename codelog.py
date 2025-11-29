#!/usr/bin/env python3
"""
代码收集器 - 将项目中的Rust代码和assets中的配置文件收集到codelog文件夹中的txt文件
使用方法: python collect_code.py [路径] [输出文件名]
"""

import sys
from datetime import datetime
from pathlib import Path


class CodeCollector:
    """收集Rust代码和配置文件并生成带文件树的文档"""

    # 文件扩展名到代码块语言的映射
    EXTENSION_LANG_MAP = {
        ".rs": "rust",
        ".json": "json",
        ".geo": "geo",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
    }

    def __init__(self, root_path="."):
        self.root_path = Path(root_path).resolve()
        if not self.root_path.exists():
            raise FileNotFoundError(f"路径不存在: {self.root_path}")

        self.collected_files = []
        self.file_tree = []
        self.output_dir_name = "codelog"
        self.output_dir = Path.cwd() / self.output_dir_name

    def gather_files(self):
        """递归收集所有目标文件"""
        print(f"🔍 正在扫描: {self.root_path}")

        # 1. 扫描src目录 (Rust源码)
        src_path = self.root_path / "src"
        if src_path.exists():
            self._scan_directory(src_path, extensions=[".rs"])

        # 2. 扫描assets目录 (JSON, GEO配置文件)
        assets_path = self.root_path / "assets"
        if assets_path.exists():
            self._scan_directory(
                assets_path, extensions=[".json", ".geo", ".toml", ".yaml", ".yml"]
            )
            print(f"📁 扫描assets目录")

        # 3. 扫描根目录下的特定文件
        for pattern in ["*.rs", "*.toml", "*.json"]:
            for file in self.root_path.glob(pattern):
                if self.output_dir_name not in file.parts and file.is_file():
                    if file not in self.collected_files:
                        self.collected_files.append(file)

        # 4. 扫描其他子目录的Rust文件
        excluded_dirs = {
            "target",
            ".git",
            "node_modules",
            "src",
            "assets",
            self.output_dir_name,
        }
        for item in self.root_path.iterdir():
            if item.is_dir() and item.name not in excluded_dirs:
                self._scan_directory(item, extensions=[".rs"])

        # 排序：按路径字母顺序
        self.collected_files.sort(key=lambda p: str(p.relative_to(self.root_path)))

        # 统计
        rust_count = sum(1 for f in self.collected_files if f.suffix == ".rs")
        json_count = sum(1 for f in self.collected_files if f.suffix == ".json")
        geo_count = sum(1 for f in self.collected_files if f.suffix == ".geo")
        other_count = len(self.collected_files) - rust_count - json_count - geo_count

        print(f"📄 找到 {len(self.collected_files)} 个文件:")
        print(f"   - Rust: {rust_count}")
        print(f"   - JSON: {json_count}")
        print(f"   - GEO:  {geo_count}")
        if other_count > 0:
            print(f"   - 其他: {other_count}")

    def _scan_directory(self, directory: Path, extensions: list[str]):
        """扫描单个目录中指定扩展名的文件"""
        try:
            for ext in extensions:
                for path in directory.rglob(f"*{ext}"):
                    parts = path.parts
                    # 排除 target 目录和 codelog 输出目录
                    if "target" not in parts and self.output_dir_name not in parts:
                        if path not in self.collected_files:
                            self.collected_files.append(path)
        except Exception as e:
            print(f"⚠️ 扫描 {directory} 时出错: {e}")

    def build_file_tree(self):
        """构建文件树结构"""
        if not self.collected_files:
            return "未找到文件\n"

        tree_lines = ["文件树：", "=" * 50, ""]

        # 按相对路径组织
        files_by_dir = {}
        for file_path in self.collected_files:
            try:
                rel_path = file_path.relative_to(self.root_path)
            except ValueError:
                rel_path = Path(file_path.name)

            dir_name = rel_path.parent
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(rel_path.name)

        # 生成树形结构
        for dir_path in sorted(files_by_dir.keys()):
            if str(dir_path) == ".":
                for filename in sorted(files_by_dir[dir_path]):
                    tree_lines.append(f"├── {filename}")
            else:
                tree_lines.append(f"├── {dir_path}/")
                for filename in sorted(files_by_dir[dir_path]):
                    indent = "│   " + "    "
                    tree_lines.append(f"{indent}├── {filename}")

        tree_lines.extend(["", "=" * 50, ""])
        return "\n".join(tree_lines)

    def _get_code_block_lang(self, file_path: Path) -> str:
        """根据文件扩展名获取代码块语言标识"""
        return self.EXTENSION_LANG_MAP.get(file_path.suffix.lower(), "text")

    def collect_to_file(self, output_filename=None):
        """将所有代码收集到codelog文件夹中的单个文件"""
        self.output_dir.mkdir(exist_ok=True)

        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"code_collection_{timestamp}.txt"

        output_path = self.output_dir / output_filename

        self.gather_files()
        file_tree_str = self.build_file_tree()

        print(f"💾 正在写入: {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            # 写入头部信息
            f.write(f"""代码收集报告
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
项目路径: {self.root_path}
扫描内容:
  - src/: *.rs
  - assets/: *.json, *.geo, *.toml, *.yaml
  - 根目录: *.rs, *.toml, *.json
排除目录: target, .git, node_modules, {self.output_dir_name}
{"=" * 80}

""")

            # 写入文件树
            f.write(file_tree_str)

            # 按类型分组写入
            rust_files = [p for p in self.collected_files if p.suffix == ".rs"]
            config_files = [p for p in self.collected_files if p.suffix != ".rs"]

            # 先写入Rust文件
            if rust_files:
                f.write("\n" + "=" * 80 + "\n")
                f.write("# Rust 源代码\n")
                f.write("=" * 80 + "\n\n")
                self._write_files(f, rust_files)

            # 再写入配置文件
            if config_files:
                f.write("\n" + "=" * 80 + "\n")
                f.write("# 配置文件 (JSON/GEO/TOML/YAML)\n")
                f.write("=" * 80 + "\n\n")
                self._write_files(f, config_files)

        print(f"✅ 完成！共记录 {len(self.collected_files)} 个文件")
        print(f"📂 输出文件: {output_path}")
        return output_path

    def _write_files(self, f, file_list: list[Path]):
        """写入文件列表到输出文件"""
        for file_path in file_list:
            try:
                try:
                    display_path = file_path.relative_to(self.root_path)
                except ValueError:
                    display_path = file_path

                content = file_path.read_text(encoding="utf-8")
                lang = self._get_code_block_lang(file_path)

                # 写入文件头
                f.write(f"# File: {display_path}\n\n")
                f.write(f"```{lang}\n")

                if not content.strip():
                    f.write("// 文件为空\n")
                else:
                    content = content.rstrip()
                    f.write(content)

                f.write("\n```\n\n")

                print(f"  ✓ 已记录: {display_path}")

            except UnicodeDecodeError:
                # 尝试其他编码
                try:
                    content = file_path.read_text(encoding="latin-1")
                    lang = self._get_code_block_lang(file_path)
                    f.write(f"# File: {file_path.name} (latin-1 编码)\n\n")
                    f.write(f"```{lang}\n{content.rstrip()}\n```\n\n")
                    print(f"  ✓ 已记录 (latin-1): {file_path.name}")
                except Exception as e:
                    print(f"  ✗ 读取失败 {file_path}: {e}")
                    f.write(
                        f"# File: {file_path.name}\n\n```text\n// 读取失败: {e}\n```\n\n"
                    )
            except Exception as e:
                print(f"  ✗ 读取失败 {file_path}: {e}")
                f.write(
                    f"# File: {file_path.name}\n\n```text\n// 读取失败: {e}\n```\n\n"
                )


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="收集Rust项目中的代码和配置文件到codelog文件夹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python collect_code.py                    # 扫描当前目录
  python collect_code.py /path/to/project   # 扫描指定路径
  python collect_code.py . my_code.txt      # 指定输出文件名

扫描规则:
  - src/         -> *.rs (Rust源码)
  - assets/      -> *.json, *.geo, *.toml, *.yaml (配置文件)
  - 根目录       -> *.rs, *.toml, *.json
  - 其他子目录   -> *.rs
        """,
    )

    parser.add_argument(
        "path", nargs="?", default=".", help="要扫描的项目路径（默认: 当前目录）"
    )
    parser.add_argument("output", nargs="?", help="输出文件名（默认: 自动生成）")

    args = parser.parse_args()

    try:
        collector = CodeCollector(args.path)
        collector.collect_to_file(args.output)
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
