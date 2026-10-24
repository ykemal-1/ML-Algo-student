---
marp: true
theme: ap-theme
paginate: true
---

<!-- _class: title-slide -->

# Week 6
## Clustering met DBSCAN

---

<!-- _class: red-bg -->
## Herhalen

---

## Vorige week: KMeans

- Clustering is **unsupervised learning**: het algoritme ontdekt zelf **groepen** in de data
- KMeans verdeelt de data in **k clusters** rond centroids
- KMeans werkt goed op **bolvormige, gecentreerde** clusters

> 📌 Vandaag bekijken we wat er gebeurt als de clusters **geen nette vorm** hebben en de data een **lokale structuur** heeft

---

## Wat als KMeans onvoldoende is

- Data is niet altijd netjes gecentreerd
- Bv. de typische **moons-dataset**, waar de data rond elkaar is gedraaid
- Hierdoor gaat KMeans **nooit echt goed** werken

![kmeans_moons h:300](images/les06_kmeans_moons_plot.png)

---

## KMeans houdt geen rekening met lokale structuur

- KMeans verdeelt de ruimte steeds in **bolvormige** groepen
- Het ziet **geen verband** tussen punten die dicht bij elkaar liggen maar tot verschillende 'armen' behoren
- Nodig: een algoritme dat werkt op basis van **lokale dichtheid**

---

<!-- _class: red-bg -->
# DBSCAN

---

## DBSCAN

- **Geen eisen** op vlak van de vorm van de data
- DBSCAN **scant de omgeving** van een punt en **rijgt ketens van punten** aan elkaar
- Werkt goed als er zones zijn met gelijkaardige punten dicht op elkaar (**dense zones, hotspots**)

![moons h:280](images/les06_moons_plot.png)

---

<!-- _class: red-bg -->
## Werking

---

## Werking: core points

- Het algoritme kijkt bij elk punt of er **voldoende buren** zijn
- Is dat zo, dan wordt dat punt een **core point**
- **Overlappende omgevingen** vormen samen 1 grote cluster
- Parameters:
  - `eps`: de **straal** van de omgeving
  - `min_samples`: het **minimaal aantal buren**

```mermaid
flowchart TD
    A["Punt in omgeving<br/>met straal ε"] --> Q{"Voldoende buren<br/>(≥ min_samples)?"}
    Q -- Ja --> C["Core point"]
    Q -- Nee --> D["Border of anomalie"]
    C --> E["Overlappende omgevingen<br/>vormen één cluster"]
```

---

## Core, border en anomalie

![schets h:330](images/les06_dbscan_schets_plot.png)

- **Core point:** genoeg buren binnen `eps` (dense zone)
- **Border point:** aan de rand van een cluster, geen eigen dichte zone
- **Anomalie:** punt dat bij geen enkele cluster hoort

---

## DBSCAN op de moons-dataset

- DBSCAN gebruikt **enkel de dichtheid** en geen vaste vorm
- Op moons krijgt hij zo wel de **twee armen** als aparte clusters

![dbscan_moons h:300](images/les06_dbscan_moons_plot.png)

---

## De `eps`-parameter

- De waarde van `eps` is belangrijk
- Te klein → **te veel clusters** (of alles wordt anomalie)
- Te groot → **te weinig clusters** (alles smelt samen tot 1 groep)

![eps_plot h:300](images/les06_dbscan_eps_plot.png)

> Let goed op de `eps` en `min_samples`: stem ze af op de dichtheid van je data

---

## DBSCAN in sklearn

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
```

[DEMO]
- DBSCAN heeft **geen `predict`-methode**, maar wel `fit_predict`
- `fit` berekent core points, border points en de cluster-graaf
- `fit_predict` geeft ook een array van labels terug, met **`-1` voor de anomalieën**

---

## Items opdelen: core, anomalie en rest

```python
core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
core_mask[dbscan.core_sample_indices_] = True

anomalies_mask = dbscan.labels_ == -1   # items zonder label

non_core_mask = ~(core_mask | anomalies_mask)
```

- De core points staan in `dbscan.core_sample_indices_`
- De labels zitten in `dbscan.labels_` (met `-1` = anomalie)

---

<!-- _class: red-bg -->
## Overige clusteringalgoritmes

---

## HDBSCAN

- Een **geavanceerde versie** van DBSCAN
- Kan om met **variërende dichtheden**, iets waar DBSCAN moeite mee heeft
- Versta het als clusteren met DBSCAN terwijl je **verschillende `eps`-waarden** probeert
<!-- - Installeren: `conda install -c conda-forge hdbscan` -->

---

## DBSCAN en variërende dichtheid

![gevoeligheid h:300](images/les06_dbscan_gevoeligheid_plot.png)

- De keuze van `eps` bepaalt of kleine of grote clusters ontdekt worden
- Bij **onbekende dichtheden** is de juiste `eps` moeilijk te kiezen

> HDBSCAN lost dat automatisch op (zie de Python notebook)

---

## BIRCH

- **Balanced Iterative Reducing and Clustering using Hierarchies**
- Werkt vooral goed op **grote datasets**, omdat het **in batches** kan werken (net als KMeans)
- Vereenvoudigt eerst de dataset tot een kleinere **summary-dataset**
- Nadeel: kan **enkel met numerieke waarden** werken

![birch h:280](images/les06_birch_plot.png)

---

## Vergelijking clustering methodes

```mermaid
flowchart TD
    V{"Vorm van de clusters?"} --> K["Bolvormig & gecentreerd<br/>→ KMeans"]
    V --> D["Willekeurige vorm<br/>→ DBSCAN"]
    D --> H["Variërende dichtheid<br/>→ HDBSCAN"]
    D --> B["Grote dataset<br/>→ BIRCH"]
```

- KMeans en DBSCAN zijn de technieken voor in de oefeningen

---

<!-- _class: red-bg -->
## Clustering op images

---

## Clustering op images

- We gebruiken **opencv2** om met afbeeldingen te werken
- Via clustering kunnen we afbeeldingen verdelen in clusters
- Hoe?
  - Door te clusteren op de **soorten kleuren** die we tegenkomen

[DEMO]
- Elke pixel is een punt in de kleurruimte (bv. RGB)
- Clustering groepeert de gelijkaardige kleuren samen

---

<!-- _class: red-bg -->
## Samenvatten

---

## Best practices (Week 6)

- Kies **KMeans** voor bolvormige, gecentreerde clusters
- Kies **DBSCAN** bij willekeurige vormen of wanneer lokale structuur belangrijk is
- Stem **`eps` en `min_samples`** af op de dichtheid van je data
- Punten met label **`-1`** zijn geen cluster, maar **anomalieën / outliers**
- Grote datasets → **BIRCH**; variërende dichtheden → **HDBSCAN**

---

## Kernpunten

- DBSCAN is **vorm-onafhankelijk** en werkt op **lokale dichtheid** via core points
- Parameters: `eps` (straal) en `min_samples` (aantal buren)
- Punten zonder cluster (label `-1`) vallen er meteen uit als **outliers**
- Alternatieven: **HDBSCAN** (geavanceerder), **BIRCH** (grote datasets)

---

## Vragen?

- [https://app.wooclap.com/events/ZXTOQF](https://app.wooclap.com/events/ZXTOQF)
