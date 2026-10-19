# Week 4 — Clustering: KMeans

> Notebook: `clustering/kmeans/kmeans.ipynb`

## Waarom clustering

- Segmentering
- Versimpeling van de verscheidenheid van data
- Dimensie-reductie
- Segmenteren van beelden (labo)

## KMeans

- Het aantal clusters **k moet je zelf meegeven** — het algoritme beslist dit niet zelf.
- **Werking** (herhaal tot convergentie):
  1. Kies centroids
  2. Bepaal voor elk punt de cluster
  3. Update centroids als nieuw gemiddelde van de cluster
  4. Herhaal tot alles stabiel is (convergentie)
- De decision boundaries van KMeans zijn **Voronoi cellen**.

### Initialisatieprobleem

- KMeans start met centroids op willekeurige plaatsen → niet deterministisch, er kunnen vreemde situaties ontstaan.
- In de praktijk: KMeans start 10× random en houdt de beste oplossing bij. Ken je de locatie van de centroids al, dan kan je die meegeven.

### Inertia (goedheid van de oplossing)

- Bereken voor de stabiele oplossing de afstand van elk punt tot zijn centroid; kwadrateer en tel op → **Mean Squared Distance = inertia**.
- Hoe lager de inertia, hoe beter. De `score()` van het model is typisch een getal dat zo hoog mogelijk moet zijn, dus **score = −inertia**.

### Toepassing: GSM-masten

- Telecomoperatoren plaatsen masten zodanig dat elke mast ongeveer even zwaar belast wordt (en (bijna) iedereen dekking heeft) — ideale locaties kan je met KMeans bepalen. Zelfde idee: wandelafstand tot dichtstbijzijnd metrostation in London.

## OPGELET: KMeans is NIET KNearestNeighbors

| KMeans | KNearestNeighbors |
|---|---|
| Clustering | Classificatie of regressie |
| k = aantal clusters | k = aantal buren; het aantal labels kan iets helemaal anders zijn |
| Inertia/scoring: MSE over héle dataset | MSE voor afstand nieuw item tot andere items |

## Mini-batch (online learning)

- Twee types van learning: **batch learning** vs **online learning**.
- Mini-batch: data wordt in stukjes getraind; handig voor heel grote datasets; iets minder nauwkeurig dan alles in één keer, maar valt in de praktijk goed mee.

```python
from sklearn.cluster import MiniBatchKMeans
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)
```

- Voor datasets die niet in het geheugen passen: **memmap**!

```python
X_memmap = np.memmap('my_map.mmap', dtype='float32', mode='write', shape=X_train.shape)
X_memmap[:] = X_train
X_memmap.flush()
minibatch_kmeans.fit(X_memmap)
```

- Van veel algoritmes bestaat een mini-batch versie die qua training bijna altijd performanter is dan de originele versie.

## Hoe bepaal je het ideale k?

- In het echt kan je niet altijd "kijken" — je kan effectief te veel of te weinig clusters hebben.
- Voor elke nieuwe cluster daalt de totale inertia → zoek naar een **knik** in die curve.

### Elbow method

- Duidelijke knik bij k=4: erna wordt het veel minder snel beter ("elbow method").

### Silhouette score

- Gemiddelde van de silhouette coëfficiënt over alle datapunten (exacte definitie: zie cursusnotebook).
- Typische waarden per punt:
  - **1**: datapunt centraal in eigen cluster
  - **0**: datapunt dicht bij een grens
  - **−1**: datapunt zit mogelijk niet in de juiste/logische cluster
- Hoe hoger de totale silhouette score, hoe beter.

```python
from sklearn.metrics import silhouette_score
silhouette_score(X, kmeans.labels_)
```

- Plot de silhouette score in functie van k → hier: 4, en ook 5, zijn goede keuzes.
- **Silhouette diagram**: per punt de coëfficiënt plotten — toont ook of sommige clusters meer punten bevatten; in rood de totale score.

## Beste praktijken (Week 4)

- **Kies k met elbow + silhouette score/diagram**, niet enkel visueel.
- Vertrouw niet op één run: KMeans is random geïnitieerd; sklearn doet standaard 10 pogingen (`n_init`) en houdt de beste bij.
- Heel grote datasets? Gebruik `MiniBatchKMeans` en/of `np.memmap`.
- Verwar KMeans niet met KNN: het ene is clustering, het andere supervised.

## Kernpunten

- KMeans: k moet vooraf gekozen worden; centroids + convergentie; Voronoi-grenzen.
- Kwaliteitsmaat: inertia (lager = beter); k bepalen via elbow en silhouette score.
- Mini-batch variant voor grote/online data.