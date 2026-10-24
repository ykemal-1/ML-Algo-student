# Week 5 — Clustering met DBSCAN

> Notebooks: `clustering/dbscan/dbscan.ipynb`, `clustering/kmeans/kmeans_dbscan_oplossingen.ipynb`

## Wat als KMeans onvoldoende is

- Data is niet altijd netjes gecentreerd — bv. de typische **moon-dataset** (data rond elkaar gedraaid). KMeans gaat daar nooit echt goed werken.
- KMeans houdt **geen rekening met 'lokale structuur'**.

## DBSCAN

- Geen eisen op vlak van de vorm van de data.
- DBSCAN "scant" de omgeving van een punt en rijgt **ketens van punten** aan elkaar.
- Werkt goed als er zones zijn met gelijkaardige punten dicht op elkaar (**dense zones, hotspots**).

### Werking

- Het algoritme kijkt bij elk punt of er **voldoende buren** zijn.
  - Ja → dat punt wordt een **core point**.
  - Overlappende omgevingen vormen samen 1 grote cluster.
- Parameters: `eps` (straal van de omgeving) en `min_samples` (minimaal aantal buren).

```python
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)

# items opdelen in core / anomalies / non-core
core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_mask[dbscan.core_sample_indices_] = True
anomalies_mask = dbscan.labels_ == -1      # items zonder label
non_core_mask = ~(core_mask | anomalies_mask)
```

## Overige clusteringalgoritmes

- Naast KMeans en DBSCAN:
  - **HDBSCAN** — geavanceerde versie van DBSCAN
  - **BIRCH** — vooral nuttig voor grote datasets
  - Zie de Python notebook voor details.

## Clustering op images

- We gebruiken **opencv2** om met afbeeldingen te werken.
- Via clustering kunnen we afbeeldingen verdelen in clusters: door te clusteren op de **soorten kleuren** die we tegenkomen.

## Beste praktijken (Week 5)

- Kies KMeans voor bolvormige, gecentreerde clusters; kies DBSCAN wanneer clusters een willekeurige vorm hebben of lokale structuur belangrijk is.
- Stem `eps` en `min_samples` af op de dichtheid van je data; punten met label −1 zijn geen cluster maar anomalieën/outliers.
- Voor grote datasets: overweeg BIRCH; voor variërende dichtheden: HDBSCAN.

## Kernpunten

- DBSCAN is vorm-onafhankelijk en werkt op lokale dichtheid via core points.
- Punten zonder cluster (label −1) vallen er meteen uit als outliers.
- Alternatieven: HDBSCAN (geavanceerder), BIRCH (grote datasets).