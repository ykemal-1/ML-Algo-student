"""Shared pytest fixture: voert het student-notebook uit en geeft de namespace terug."""
import nbformat
import pytest
from nbclient import NotebookClient
from pathlib import Path

NOTEBOOKS = {
    "week01": Path(__file__).resolve().parents[1] / "week01" / "knn_checkpoint_opgave.ipynb",
    "week02": Path(__file__).resolve().parents[1] / "week02" / "svm_checkpoint_opgave.ipynb",
    "week03": Path(__file__).resolve().parents[1] / "week03" / "decision_trees_oefening_checkpoint_opgave.ipynb",
    "week04": Path(__file__).resolve().parents[1] / "week04" / "arima_checkpoint_opgave.ipynb",
    "week05": Path(__file__).resolve().parents[1] / "week05" / "herhaling_classification_checkpoint_opgave.ipynb",
    "week06": Path(__file__).resolve().parents[1] / "week06" / "kmeans_dbscan_checkpoint_opgave.ipynb",
}


def _notebook_namespace(week: str) -> dict:
    path = NOTEBOOKS[week]
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3", allow_errors=True,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    ns = {}
    import os
    old_cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            try:
                exec(compile(cell.source, f"<{week}:cell>", "exec"), ns)
            except Exception:
                pass  # student-code fouten: variabelen ontbreken dan gewoon -> test faalt
    finally:
        os.chdir(old_cwd)
    return ns

@pytest.fixture(scope="session")
def week03():
    return _notebook_namespace("week03")


@pytest.fixture(scope="session")
def week04():
    return _notebook_namespace("week04")

@pytest.fixture(scope="session")
def week01():
    return _notebook_namespace("week01")


@pytest.fixture(scope="session")
def week02():
    return _notebook_namespace("week02")


@pytest.fixture(scope="session")
def week05():
    return _notebook_namespace("week05")


@pytest.fixture(scope="session")
def week06():
    return _notebook_namespace("week06")
