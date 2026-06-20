from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_CONFIG


def test_config_points_to_fengshen():
    assert DEFAULT_CONFIG.fengshen_text_path == "./封神演义.txt"
    assert DEFAULT_CONFIG.faiss_index_path == "./fengshen_faiss_index"


def test_readme_reflects_fengshen_scope():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "封神演义" in readme
    assert "import_fengshen_to_neo4j.py" in readme

    legacy_terms = [
        "东周",
        "资治通鉴",
        "dongzhou_faiss_index",
        "中国近现代史纲要",
        "modern_history_text_dir",
    ]
    for term in legacy_terms:
        assert term not in readme, f"README should not contain legacy term: {term}"


def test_project_metadata_matches_fengshen_scope():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert "fengshen" in project["name"].lower()
    assert "封神演义" in project["description"]
