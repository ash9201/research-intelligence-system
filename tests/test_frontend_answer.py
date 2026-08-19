"""Regression checks for the Streamlit Ask rendering contract."""
import ast
from pathlib import Path


def test_answer_metrics_unpack_three_columns():
    """The three metrics must bind three columns, preventing the observed ValueError."""
    source = Path("src/frontend/main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Attribute) or node.value.func.attr != "columns":
            continue
        if len(node.value.args) == 1 and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value == 3:
            target = node.targets[0]
            if isinstance(target, ast.Tuple):
                matches.append(len(target.elts))
    assert 3 in matches