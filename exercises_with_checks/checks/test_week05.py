"""Checks voor het week05 checkpoint (Herhalingslabo Classificatie).

Elke test valideert exact één verplichte variabele uit het notebook:
bestaat -> type -> drempelwaarde. Geen implementatiehints in de meldingen.
"""
import numpy as np
import pytest


def _get(week, name):
    val = week.get(name)
    assert val is not None, (
        f"Variabele '{name}' ontbreekt of heeft geen waarde "
        f"(verwacht: gevuld volgens de opdracht)."
    )
    return val


# ---------------- Diabetes: SVC ----------------

def test_diabetes_svc_y_pred(week05):
    y_pred = _get(week05, "y_pred_diabetes_svc")
    assert len(np.asarray(y_pred)) == 154, (
        f"y_pred_diabetes_svc heeft {len(np.asarray(y_pred))} voorspellingen, verwacht 154"
    )


def test_diabetes_svc_accuracy(week05):
    acc = _get(week05, "accuracy_diabetes_svc")
    assert float(acc) >= 0.70, (
        f"accuracy_diabetes_svc={float(acc):.4f} is te laag (verwacht >= 0.70)"
    )


def test_diabetes_svc_accuracy_boven_gok(week05):
    acc = float(_get(week05, "accuracy_diabetes_svc"))
    assert acc > 0.5, (
        f"accuracy_diabetes_svc={acc:.4f} is niet beter dan gokken (verwacht > 0.5)"
    )


# ---------------- Diabetes: KNN ----------------

def test_diabetes_knn_y_pred(week05):
    y_pred = _get(week05, "y_pred_diabetes_knn")
    assert len(np.asarray(y_pred)) == 154, (
        f"y_pred_diabetes_knn heeft {len(np.asarray(y_pred))} voorspellingen, verwacht 154"
    )


def test_diabetes_knn_accuracy(week05):
    acc = _get(week05, "accuracy_diabetes_knn")
    assert float(acc) >= 0.60, (
        f"accuracy_diabetes_knn={float(acc):.4f} is te laag (verwacht >= 0.60)"
    )


# ---------------- Churn ----------------

def test_churn_y_pred(week05):
    y_pred = _get(week05, "y_pred_churn")
    assert len(np.asarray(y_pred)) == 1409, (
        f"y_pred_churn heeft {len(np.asarray(y_pred))} voorspellingen, verwacht 1409"
    )


def test_churn_accuracy(week05):
    acc = _get(week05, "accuracy_churn")
    assert float(acc) >= 0.75, (
        f"accuracy_churn={float(acc):.4f} is te laag (verwacht >= 0.75)"
    )


# ---------------- Creditcard ----------------

def test_creditcard_ada_y_pred(week05):
    y_pred = _get(week05, "y_pred_creditcard_ada")
    assert len(np.asarray(y_pred)) == 56962, (
        f"y_pred_creditcard_ada heeft {len(np.asarray(y_pred))} voorspellingen, verwacht 56962"
    )


def test_creditcard_ada_accuracy(week05):
    acc = _get(week05, "accuracy_creditcard_ada")
    assert float(acc) >= 0.998, (
        f"accuracy_creditcard_ada={float(acc):.4f} is te laag (verwacht >= 0.998)"
    )


def test_creditcard_ada_detecteert_fraude(week05):
    y_pred = np.asarray(_get(week05, "y_pred_creditcard_ada"))
    assert int(np.sum(np.asarray(y_pred) == 1)) > 0, (
        "y_pred_creditcard_ada bevat geen enkele fraudedetectie (alle 0)"
    )


def test_creditcard_xgb_y_pred(week05):
    y_pred = _get(week05, "y_pred_creditcard_xgb")
    assert len(np.asarray(y_pred)) == 56962, (
        f"y_pred_creditcard_xgb heeft {len(np.asarray(y_pred))} voorspellingen, verwacht 56962"
    )


def test_creditcard_xgb_accuracy(week05):
    acc = _get(week05, "accuracy_creditcard_xgb")
    assert float(acc) >= 0.999, (
        f"accuracy_creditcard_xgb={float(acc):.4f} is te laag (verwacht >= 0.999)"
    )


def test_creditcard_xgb_detecteert_fraude(week05):
    y_pred = np.asarray(_get(week05, "y_pred_creditcard_xgb"))
    assert int(np.sum(y_pred == 1)) > 0, (
        "y_pred_creditcard_xgb bevat geen enkele fraudedetectie (alle 0)"
    )


# ---------------- Meerkeuzevragen ----------------

def test_antwoord_algoritme(week05):
    antw = _get(week05, "antwoord_algoritme")
    assert str(antw).strip().upper() == "B", (
        f"antwoord_algoritme={str(antw)!r} is niet correct"
    )


def test_antwoord_ongebalanceerd(week05):
    antw = _get(week05, "antwoord_ongebalanceerd")
    assert str(antw).strip().upper() == "B", (
        f"antwoord_ongebalanceerd={str(antw)!r} is niet correct"
    )
