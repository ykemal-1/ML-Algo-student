# Richtcijfers voor kleine, middelgrote en grote datasets (2026, met input van AI)

Er bestaat geen universele grens tussen een kleine, middelgrote en grote dataset. De grootte hangt niet alleen af van het aantal records (rijen), maar ook van het aantal features (kolommen).

## Grootte op basis van aantal records

| Grootte dataset | Aantal records |
|-----------------|----------------|
| Klein | < 10.000 |
| Middelgroot | 10.000 - 100.000 |
| Groot | 100.000 - 1.000.000 |
| Zeer groot | > 1.000.000 |

## Grootte op basis van aantal features

| Aantal features | Interpretatie |
|-----------------|---------------|
| < 20 | Weinig features |
| 20 - 100 | Gemiddeld |
| 100 - 1.000 | Veel features |
| > 1.000 | Heel veel features |

---

# Algoritmes en datasetgrootte

## K-Nearest Neighbours (KNN)

**Geschikt voor:**
- Kleine tot middelgrote datasets (< 100.000 records)
- Weinig tot gemiddeld aantal features (< 50 à 100)

**Waarom?**
- Eenvoudig algoritme
- Snel te trainen
- Predicties worden traag wanneer de dataset groot wordt
- Werkt minder goed wanneer er veel features zijn

---

## Support Vector Machines (SVM)

**Geschikt voor:**
- Kleine tot middelgrote datasets (< 50.000 à 100.000 records)
- Kan goed omgaan met veel features

**Waarom?**
- Zeer sterk voor classificatie
- Training wordt snel duur bij grote datasets
- Kernelkeuze en hyperparameter tuning zijn belangrijk

---

## Decision Trees

**Geschikt voor:**
- Kleine en middelgrote datasets
- Zowel numerieke als categorische data

**Waarom?**
- Goed interpreteerbaar ("white box")
- Kan omgaan met ontbrekende waarden
- Risico op overfitting is relatief groot

**Tip:** Bij twijfel is een Random Forest vaak een betere keuze.

---

## Random Forest

**Geschikt voor:**
- Middelgrote tot grote datasets
- Data met veel features

**Waarom?**
- Minder risico op overfitting dan één enkele decision tree
- Vaak sterk zonder uitgebreide tuning
- Meer trainingstijd dan een gewone decision tree

---

## AdaBoost

**Geschikt voor:**
- Kleine tot middelgrote datasets
- Problemen met minder goed gebalanceerde klassen

**Waarom?**
- Combineert meerdere zwakke modellen tot een sterker model
- Vermijdt vaak onderfitting

**Opmerking:** In de praktijk wordt AdaBoost anno 2026 minder gebruikt dan XGBoost.

---

## XGBoost

**Geschikt voor:**
- Middelgrote tot zeer grote datasets
- Data met veel of zeer veel features

**Waarom?**
- Vaak één van de best presterende algoritmes voor tabulaire data
- Ondersteunt gedeeltelijke parallellisatie
- Sterke standaardkeuze wanneer je niet zeker weet welk model te gebruiken

---

# Eenvoudige beslisboom

1. Wil je maximale interpreteerbaarheid?
   - Decision Tree
   - Random Forest

2. Heb je weinig data?
   - KNN
   - SVM

3. Heb je veel data of veel features?
   - Random Forest
   - XGBoost

4. Twijfel je?
   - Probeer eerst XGBoost.

---

# Belangrijk

In de praktijk zijn er vaak meerdere goede keuzes. Het is belangrijk om te begrijpen waarom een model geschikt of minder geschikt is voor een bepaald probleem. Uiteindelijk blijft het vaak nodig om meerdere veelbelovende modellen te trainen en hun prestaties te vergelijken.