"""Checkpoint-tests week 3: Decision Trees & Ensembles (opgave-notebook van de student)."""

import numpy as np


# ---------- Oefening 1: Decision Trees op Wine ----------
def test_oef1_theorie_tree_variatie(week03):
    antw = week03.get("antwoord_tree_variatie")
    assert antw is not None, "variabele 'antwoord_tree_variatie' ontbreekt"
    assert str(antw).strip().upper() == "B", (
        "Fout: kijk naar het effect van een andere random state op de boomstructuur"
    )


# ---------- Oefening 2: Voting classifier op Breast Cancer ----------
def test_oef2_accuracy_voting(week03):
    acc = week03.get("accuracy_voting")
    assert acc is not None, "variabele 'accuracy_voting' ontbreekt"
    acc = float(acc)
    assert 0.9 <= acc <= 1.0, f"accuracy_voting={acc} is te laag (verwacht >= 0.9)"


def test_oef2_y_pred_voting(week03):
    y_pred = week03.get("y_pred_voting")
    assert y_pred is not None, "variabele 'y_pred_voting' ontbreekt"
    assert len(np.asarray(y_pred)) == 114, (
        "y_pred_voting heeft niet de juiste lengte (test_size=0.2 van 569)"
    )


def test_oef2_theorie_ensemble(week03):
    antw = week03.get("antwoord_ensemble")
    assert antw is not None, "variabele 'antwoord_ensemble' ontbreekt"
    assert str(antw).strip().upper() == "C", (
        "Fout: denk na over hoe de voorspellingen van de individuele modellen gecombineerd worden"
    )


# ---------- Oefening 3: Bagging op Wine ----------
def test_oef3_accuracy_bagging(week03):
    acc = week03.get("accuracy_bagging")
    assert acc is not None, "variabele 'accuracy_bagging' ontbreekt"
    acc = float(acc)
    assert 0.9 <= acc <= 1.0, f"accuracy_bagging={acc} is te laag (verwacht >= 0.9)"


def test_oef3_accuracy_single_tree(week03):
    acc = week03.get("accuracy_single_tree")
    assert acc is not None, "variabele 'accuracy_single_tree' ontbreekt"
    acc = float(acc)
    assert 0.85 <= acc <= 1.0, f"accuracy_single_tree={acc} is te laag (verwacht >= 0.85)"


# ---------- Oefening 4: AdaBoost vs XGBoost op Titanic ----------
def test_oef4_y_pred_ada(week03):
    y_pred = week03.get("y_pred_ada")
    assert y_pred is not None, "variabele 'y_pred_ada' ontbreekt"
    assert len(np.asarray(y_pred)) == 179, (
        "y_pred_ada heeft niet de juiste lengte (test_size=0.2 van 891)"
    )


def test_oef4_accuracy_ada(week03):
    acc = week03.get("accuracy_ada")
    assert acc is not None, "variabele 'accuracy_ada' ontbreekt"
    acc = float(acc)
    assert 0.75 <= acc <= 1.0, f"accuracy_ada={acc} is te laag (verwacht >= 0.75)"


def test_oef4_f1_ada(week03):
    f1 = week03.get("f1_ada")
    assert f1 is not None, "variabele 'f1_ada' ontbreekt"
    f1 = float(f1)
    assert 0.65 <= f1 <= 1.0, f"f1_ada={f1} is te laag (verwacht >= 0.65)"


def test_oef4_y_pred_xgb(week03):
    y_pred = week03.get("y_pred_xgb")
    assert y_pred is not None, "variabele 'y_pred_xgb' ontbreekt"
    assert len(np.asarray(y_pred)) == 179, (
        "y_pred_xgb heeft niet de juiste lengte (test_size=0.2 van 891)"
    )


def test_oef4_accuracy_xgb(week03):
    acc = week03.get("accuracy_xgb")
    assert acc is not None, "variabele 'accuracy_xgb' ontbreekt"
    acc = float(acc)
    assert 0.75 <= acc <= 1.0, f"accuracy_xgb={acc} is te laag (verwacht >= 0.75)"


def test_oef4_f1_xgb(week03):
    f1 = week03.get("f1_xgb")
    assert f1 is not None, "variabele 'f1_xgb' ontbreekt"
    f1 = float(f1)
    assert 0.65 <= f1 <= 1.0, f"f1_xgb={f1} is te laag (verwacht >= 0.65)"
