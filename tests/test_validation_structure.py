import py_compile
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parent.parent
VALIDATION_ROOT = REPO_ROOT / "validation"

EXPECTED_CATEGORIES = [
    "acquisition_strategies",
    "learner_evaluation",
    "large_scale",
    "uncertainty",
    "performance",
    "pruning",
]


def test_all_category_dirs_exist():
    for category in EXPECTED_CATEGORIES:
        category_dir = VALIDATION_ROOT / category
        assert category_dir.is_dir(), f"Missing category directory: {category}"


def test_no_scripts_in_old_location():
    old_scripts_dir = VALIDATION_ROOT / "scripts"
    assert not old_scripts_dir.exists(), "Old validation/scripts/ directory should not exist"


@pytest.mark.parametrize(
    "script",
    sorted(
        list(VALIDATION_ROOT.glob("*/scripts/validate_*.py"))
        + list(VALIDATION_ROOT.glob("*/scripts/benchmark_*.py"))
        + list(VALIDATION_ROOT.glob("performance/speedup/scripts/benchmark_*.py"))
    ),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_all_scripts_importable(script):
    py_compile.compile(str(script), doraise=True)


def test_no_pycache_dirs_in_git():
    import subprocess
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "validation/**/__pycache__"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode != 0, "Found __pycache__ dirs tracked in git"


def test_no_checkpoint_files():
    import subprocess
    result = subprocess.run(
        ["git", "ls-files", "--", "validation/**/*.ckpt"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert tracked == [], f"Checkpoint files tracked in git: {tracked}"


def test_gitignore_patterns():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    required_patterns = [
        "validation/reports/",
        "**/__pycache__/",
        "**/*.ckpt",
    ]
    for pattern in required_patterns:
        assert pattern in gitignore, f"Missing .gitignore pattern: {pattern}"


def test_orchestrator_discovers_minimum_scripts():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_all_validations",
        str(VALIDATION_ROOT / "run_all_validations.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    discovered = module.discover_scripts()
    total = sum(len(scripts) for scripts in discovered.values())
    assert total >= 13, f"Expected >= 13 scripts, found {total}"


def test_dataset_config_no_stale_paths():
    config_path = VALIDATION_ROOT / "lib" / "dataset_config.py"
    content = config_path.read_text(encoding="utf-8")
    assert "ampc_30k_subsample.csv" not in content, "Stale ampc_30k_subsample.csv reference in dataset_config.py"


def test_yaml_config_paths_valid():
    yaml_path = VALIDATION_ROOT / "config" / "validation_strategies.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_base = config["global"]["output_base_dir"]
    assert "validation/reports" in output_base, f"output_base_dir should contain 'validation/reports', got: {output_base}"

    for strategy_name, strategy in config.get("strategies", {}).items():
        assert "category" in strategy, f"Strategy '{strategy_name}' missing 'category' field"


def test_lib_consolidation_api_surface():
    from validation.lib import matrix_visualizations as mv

    required_functions = [
        "create_top_k_heatmap",
        "create_all_heatmaps",
        "create_greedy_cycle_plot",
        "create_summary_report",
        "generate_comprehensive_visualizations",
        "create_learner_cycle_plot",
        "calculate_cumulative_timing",
        "create_time_heatmap",
        "create_performance_heatmap",
        "create_efficiency_heatmap",
        "create_all_cost_performance_plots",
        "featurizer_create_all_heatmaps",
        "featurizer_create_summary_report",
        "featurizer_generate_comprehensive_visualizations",
    ]
    for func_name in required_functions:
        assert hasattr(mv, func_name), f"matrix_visualizations missing function: {func_name}"


def test_absolute_imports_no_sys_path_hack():
    scripts = (
        list(VALIDATION_ROOT.glob("*/scripts/validate_*.py"))
        + list(VALIDATION_ROOT.glob("*/scripts/benchmark_*.py"))
        + list(VALIDATION_ROOT.glob("performance/speedup/scripts/benchmark_*.py"))
    )
    violations = []
    for script in scripts:
        content = script.read_text(encoding="utf-8")
        if "sys.path.insert" in content:
            violations.append(str(script.relative_to(REPO_ROOT)))
    assert len(violations) == 0, f"Scripts still using sys.path.insert: {violations}"
