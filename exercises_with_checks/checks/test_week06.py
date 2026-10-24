"""Checks voor het week06 checkpoint (KMeans & DBSCAN).

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


# ---------------- Oefening 1: Wijn (KMeans) ----------------

def test_best_k_wine(week06):
    k = _get(week06, "best_k_wine")
    assert isinstance(k, (int, np.integer)) and 2 <= int(k) <= 5, (
        f"best_k_wine={k!r} is geen geldig aantal clusters (verwacht 2 t/m 5)"
    )


def test_silhouette_wine(week06):
    sil = _get(week06, "silhouette_wine")
    assert float(sil) >= 0.25, (
        f"silhouette_wine={float(sil):.4f} is te laag (verwacht >= 0.25)"
    )


# ---------------- Oefening 2: Image Segmentation ----------------

def test_image_ndim(week06):
    image = _get(week06, "image")
    assert np.asarray(image).ndim == 3, (
        "image moet een 3D numpy-array zijn (pixels, breedte, kleurkanalen)"
    )


def test_n_colors(week06):
    n_colors = _get(week06, "n_colors")
    assert tuple(n_colors) == (10, 8, 6, 4, 2), (
        f"n_colors={tuple(n_colors)!r} klopt niet (verwacht (10, 8, 6, 4, 2))"
    )


def test_segmented_imgs_len(week06):
    segmented_imgs = _get(week06, "segmented_imgs")
    assert len(segmented_imgs) == 5, (
        f"segmented_imgs bevat {len(segmented_imgs)} afbeeldingen, verwacht 5"
    )


def test_segmented_imgs_shape(week06):
    segmented_imgs = _get(week06, "segmented_imgs")
    image = _get(week06, "image")
    assert len(segmented_imgs) == 5
    assert all(np.asarray(s).shape == np.asarray(image).shape
               for s in segmented_imgs), (
        "elke gesegmenteerde afbeelding moet dezelfde vorm hebben als image"
    )


def test_segmented_colors_reduced(week06):
    segmented_imgs = _get(week06, "segmented_imgs")
    assert len(segmented_imgs) == 5
    first = np.asarray(segmented_imgs[0]).reshape(-1, 3)
    last = np.asarray(segmented_imgs[-1]).reshape(-1, 3)
    n_first = len(np.unique(first, axis=0))
    n_last = len(np.unique(last, axis=0))
    assert n_first <= 10, (
        f"eerste gesegmenteerde afbeelding heeft {n_first} kleuren, verwacht <= 10"
    )
    assert n_last <= 2, (
        f"laatste gesegmenteerde afbeelding heeft {n_last} kleuren, verwacht <= 2"
    )


# ---------------- Oefening 3: Iris ----------------

def test_best_k_iris(week06):
    k = _get(week06, "best_k_iris")
    assert isinstance(k, (int, np.integer)) and 2 <= int(k) <= 5, (
        f"best_k_iris={k!r} is geen geldig aantal clusters (verwacht 2 t/m 5)"
    )


def test_silhouette_iris(week06):
    sil = _get(week06, "silhouette_iris")
    assert float(sil) >= 0.35, (
        f"silhouette_iris={float(sil):.4f} is te laag (verwacht >= 0.35)"
    )


# ---------------- Oefening 4: Mall ----------------

def test_best_k_mall(week06):
    k = _get(week06, "best_k_mall")
    assert isinstance(k, (int, np.integer)) and 4 <= int(k) <= 7, (
        f"best_k_mall={k!r} is geen geldig aantal clusters (verwacht 4 t/m 7)"
    )


def test_silhouette_mall(week06):
    sil = _get(week06, "silhouette_mall")
    assert float(sil) >= 0.35, (
        f"silhouette_mall={float(sil):.4f} is te laag (verwacht >= 0.35)"
    )


# ---------------- Meerkeuzevragen ----------------

def test_antwoord_iris(week06):
    antw = _get(week06, "antwoord_iris")
    assert str(antw).strip().upper() == "A", (
        f"antwoord_iris={str(antw)!r} is niet correct"
    )


def test_antwoord_dbscan(week06):
    antw = _get(week06, "antwoord_dbscan")
    assert str(antw).strip().upper() == "A", (
        f"antwoord_dbscan={str(antw)!r} is niet correct"
    )