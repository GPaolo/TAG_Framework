import subprocess
import sys
from pathlib import Path


def ensure_python_path():
    """Add src directory to Python path."""
    src_path = str(Path.cwd() / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def create_api_docs():
    """Create API documentation with auto-generated docstrings."""
    api_dir = Path("docs/api")
    api_dir.mkdir(parents=True, exist_ok=True)

    module_files = {
        "hierarchy.md": "tame.hierarchy",
        "agents.md": "tame.agents",
        "utils.md": "tame.utils",
    }

    for doc_file, module_name in module_files.items():
        content = [f"# {doc_file.replace('.md','').title()} API Reference\n"]
        content.append(f"```python\nfrom {module_name} import *\n```\n")
        content.append(f"::: {module_name}")

        (api_dir / doc_file).write_text("\n".join(content))


def build_docs():
    """Build complete documentation."""
    ensure_python_path()
    create_api_docs()
    subprocess.run(["mkdocs", "build"])


if __name__ == "__main__":
    build_docs()
