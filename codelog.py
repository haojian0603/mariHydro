#!/usr/bin/env python3
"""
代码收集器 - 将指定路径下的所有文件收集到codelog文件夹中的txt文件
使用方法: 
python E:\Documents\mariHydro\codelog.py E:\Documents\mariHydro\marihydro\crates
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


class CodeCollector:
    """收集指定路径下的所有文件并生成带文件树的文档"""

    # 文件扩展名到代码块语言的映射
    EXTENSION_LANG_MAP = {
        ".rs": "rust",
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".json": "json",
        ".geo": "geo",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".wgsl": "wgsl",
        ".md": "markdown",
        ".txt": "text",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".fish": "fish",
        ".css": "css",
        ".scss": "scss",
        ".html": "html",
        ".xml": "xml",
        ".sql": "sql",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "cpp",
        ".hpp": "cpp",
        ".java": "java",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".ini": "ini",
        ".conf": "conf",
        ".config": "conf",
        ".dockerfile": "dockerfile",
    }

    def __init__(self, root_path=".", excluded_extensions=None, excluded_dirs=None):
        self.root_path = Path(root_path).resolve()
        if not self.root_path.exists():
            raise FileNotFoundError(f"路径不存在: {self.root_path}")

        # 默认排除的文件扩展名
        self.excluded_extensions = excluded_extensions or {
            ".pyc",
            ".pyo",
            ".so",
            ".dll",
            ".exe",
            ".bin",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".ico",
            ".svg",
            ".webp",
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".zip",
            ".tar",
            ".gz",
            ".rar",
            ".7z",
            ".lock",
            ".sqlite",
            ".db",
        }
        
        # 默认排除的目录
        self.excluded_dirs = excluded_dirs or {
            "target",
            ".git",
            "node_modules",
            "__pycache__",
            ".idea",
            ".vscode",
            "build",
            "dist",
            "codelog",
        }

        self.collected_files = []
        self.output_dir_name = "codelog"
        self.output_dir = Path.cwd() / self.output_dir_name

    def gather_files(self):
        """递归收集所有文件（应用排除规则）"""
        print(f"🔍 正在扫描: {self.root_path}")

        # 递归扫描所有文件
        try:
            for file_path in self.root_path.rglob("*"):
                if not file_path.is_file():
                    continue
                    
                # 检查是否在排除目录中
                parts = file_path.parts
                if any(excluded in parts for excluded in self.excluded_dirs):
                    continue
                    
                # 检查文件扩展名是否在排除列表中
                if file_path.suffix.lower() in self.excluded_extensions:
                    continue
                    
                # 排除输出目录本身
                if self.output_dir_name in parts:
                    continue
                    
                self.collected_files.append(file_path)
                
        except Exception as e:
            print(f"⚠️ 扫描时出错: {e}")

        # 排序：按路径字母顺序
        self.collected_files.sort(key=lambda p: str(p.relative_to(self.root_path)))

        # 统计
        total_count = len(self.collected_files)
        if total_count == 0:
            print("⚠️ 未找到任何文件")
            return

        # 按扩展名分组统计
        ext_stats = {}
        for f in self.collected_files:
            ext = f.suffix.lower() or "(无扩展名)"
            ext_stats[ext] = ext_stats.get(ext, 0) + 1

        print(f"📄 找到 {total_count} 个文件:")
        for ext, count in sorted(ext_stats.items()):
            print(f"   - {ext}: {count}")

    def build_file_tree(self) -> str:
        """构建 ASCII 风格的文件树结构"""
        if not self.collected_files:
            return "文件树：\n" + "=" * 50 + "\n.\n\n" + "=" * 50 + "\n"

        # 构建树结构
        tree = {}
        for file_path in self.collected_files:
            try:
                rel_path = file_path.relative_to(self.root_path)
            except ValueError:
                rel_path = Path(file_path.name)
            
            current = tree
            parts = rel_path.parts
            
            # 逐级构建目录树
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # 添加文件
            file_part = parts[-1]
            if '' not in current:
                current[''] = []
            current[''].append(file_part)

        # 渲染 ASCII 树
        lines = ["文件树：", "=" * 50, ""]
        
        # 添加根目录名称
        root_name = self.root_path.name or "."
        lines.append(root_name + "/")
        
        def render_tree(node: Dict, prefix: str = "", is_last_root: bool = True):
            """递归渲染树节点"""
            # 获取所有条目（目录和文件列表）
            items = []
            files = node.get('', [])
            
            # 添加子目录
            for dirname, subnode in node.items():
                if dirname != '':
                    items.append(('dir', dirname, subnode))
            
            # 添加文件
            for filename in sorted(files):
                items.append(('file', filename, None))
            
            # 渲染每个条目
            count = len(items)
            for i, (item_type, name, subnode) in enumerate(items):
                is_last = (i == count - 1)
                
                if item_type == 'dir':
                    # 渲染目录
                    lines.append(f"{prefix}+-- {name}/")
                    # 准备下一级前缀
                    next_prefix = prefix + (" " if is_last else "| ")
                    render_tree(subnode, next_prefix, is_last)
                else:
                    # 渲染文件
                    connector = "\\-- " if is_last else "+-- "
                    lines.append(f"{prefix}{connector}{name}")
        
        render_tree(tree)
        lines.extend(["", "=" * 50, ""])
        return "\n".join(lines)

    def _get_code_block_lang(self, file_path: Path) -> str:
        """根据文件扩展名获取代码块语言标识"""
        return self.EXTENSION_LANG_MAP.get(file_path.suffix.lower(), "text")

    def collect_to_file(self, output_filename=None):
        """将所有文件收集到codelog文件夹中的单个文件"""
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
            f.write(f"""代码收集日志
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
项目路径: {self.root_path}
扫描模式: 收集所有文件（应用排除规则）
排除的文件类型: {', '.join(sorted(self.excluded_extensions))}
排除的目录: {', '.join(sorted(self.excluded_dirs))}
{"=" * 80}
""")

            # 写入文件树
            f.write(file_tree_str)

            # 按扩展名分组写入
            files_by_ext = {}
            for file_path in self.collected_files:
                ext = file_path.suffix.lower()
                if ext not in files_by_ext:
                    files_by_ext[ext] = []
                files_by_ext[ext].append(file_path)

            # 按扩展名排序后写入
            for ext in sorted(files_by_ext.keys()):
                ext_name = ext[1:].upper() if ext else "无扩展名"
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"# {ext_name} 文件\n")
                f.write("=" * 80 + "\n\n")
                self._write_files(f, sorted(files_by_ext[ext]))

        print(f"✅ 完成！共记录 {len(self.collected_files)} 个文件")
        print(f"📂 输出文件: {output_path}")
        return output_path

    def _write_files(self, f, file_list: List[Path]):
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

                # 显示进度
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
        description="收集指定路径下的所有文件到codelog文件夹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path", help="要扫描的项目路径"
    )

    args = parser.parse_args()

    try:
        # 自定义排除的文件类型和目录（在此修改）
        custom_excluded_extensions = {
            # 二进制和编译文件
            ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".o", ".obj", ".class",
            # 图片和媒体
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg", ".webp",
            ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
            # 文档和压缩包
            ".pdf", ".doc", ".docx", ".ppt", ".pptx",
            ".zip", ".tar", ".gz", ".rar", ".7z", ".bz2", ".xz",
            # 数据库和锁文件
            ".sqlite", ".db", ".lock",
            ".md"
        }
        
        custom_excluded_dirs = {
            # 编译输出
            "target", "build", "dist", "out", "output",
            # 版本控制
            ".git", ".svn", ".hg",
            # 依赖和缓存
            "node_modules", "__pycache__", ".venv", "venv", "env",
            "vendor", ".cache", ".gradle", ".cargo",
            # IDE
            ".idea", ".vscode", ".vs",
            # 其他
            "codelog",".ai"
        }
        
        collector = CodeCollector(
            args.path,
            excluded_extensions=custom_excluded_extensions,
            excluded_dirs=custom_excluded_dirs,
        )
        collector.collect_to_file()
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()