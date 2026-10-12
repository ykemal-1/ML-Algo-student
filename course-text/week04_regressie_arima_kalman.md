# Week 7 — Regressie en Forecasting (ARIMA, Kalman Filters)

> Notebooks: `regression/basis_regressie.ipynb`, `regression/ARIMA/arima_nieuw_2024.ipynb`, `regression/Kalman/kalman_oefening_oplossing.ipynb`

## Regressie: herhaling en varianten

- Vorige weken (Naive Bayes, KNN, SVM, Random Forests) zijn allemaal classifiers van gelabelde, categorieke data. **Wat als de output numeriek is in plaats van categorisch?**

### Dezelfde algoritmes als regressor

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

knn = KNeighborsRegressor(n_neighbors=5)
svr = SVR(kernel='linear')
rf  = RandomForestRegressor(n_estimators=100, random_state=42)
# elk: .fit(X_train, y_train); predictions = model.predict(X_test)
```

### Regressie evalueren

```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
```

- De **MSE** is de meest gekende; typisch is een zo laag mogelijk resultaat het best. $R^2$ is de absolute standaard.

## Tijdsreeksen

- Data uit tijdsreeksen komen heel vaak voor in (industriële) processen: meetwaarden van machines, logs, winstopvolging doorheen de tijd, …
- Speciale technieken geven de tijdsdimensie een speciale status; vaak willen we patronen vinden en/of **forecasting** (datapunten in de toekomst voorspellen).

### Lineaire regressie

- Een heel simpel model voor time series: geeft enkel een lineair verband — kan dus in veel datasets geen goede fit geven.

## ARIMA

**A**uto**R**egressive **I**ntegrated **M**oving **A**verage — een algemener regressiemodel met 3 parameters: **p, d, q**.

### p — AutoRegression

- Met hoeveel vorige datapunten moet rekening gehouden worden?
- **Hoge p**: veel vorige datapunten meenemen, weinig responsief voor plotse wijzigingen.
- **Lage p**: minder lange geschiedenis, reageert sneller op plotse wijzigingen.

### q — Moving Average

- Met hoeveel vorige **errorwaardes** moet rekening gehouden worden?
- **Hoge q**: veel vorige foutwaardes meenemen. **Lage q**: weinig.
- Opgelet: de term "moving average" heeft óók een veel simpelere betekenis (bv. 7-daags gemiddelde, zoals bij covid19-data van Sciensano). Dat is NIET wat hier gebeurt.

### d — Integration (differencing)

- De vorige methodes werken enkel wanneer de tijdsreeks constant (stationair) blijft door de tijd — in de praktijk zelden.
- We kijken naar de **verschillen tussen elke tijdsstap** en maken daar een nieuwe tijdsreeks mee; dit herhalen we (parameter d) tot de nieuwe reeks stabiel is.
- **Hoge d**: voor reeksen met kwadratisch verband (na d=2 constant) of heel abrupt gedrag. **Lage d**: lineaire reeksen zijn na d=1 al stationair.

### Parameters bepalen

**1. Is differencing nodig? — Adfuller-test** (null-hypothese: de time series is níet-stationair):

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(df.value.dropna())
print('ADF statistiek: %f' % result[0])
print('p-waarde: %f' % result[1])   # als p > 0.05 → differencing toepassen
```

**2. Differencing + ACF bekijken** (de ACF-plot wil je dicht bij 0):

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.plot(df.value.diff())            # 1st: df.value.diff(), 2nd: .diff().diff()
plot_acf(df.value.diff().dropna())
```

**3. p bepalen — Partial Autocorrelation (PACF)**, op de gedifferenceerde reeks:

```python
plot_pacf(df.value.diff().dropna())
```

**4. q bepalen — Autocorrelation (ACF)**, op de gedifferenceerde reeks:

```python
plot_acf(df.value.diff().dropna())
```

### ARIMA toepassen

```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df.value, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())
```

- In de summary: **AIC** = foutscore, hoe lager hoe beter; de uiteindelijke coëfficiënten van het model en de p-waarde van die coëfficiënt.

### Voorspellen en seizoenspatronen

```python
plt.plot(np.arange(len(df)), df['value'], label='Gegeven data')
plt.plot(np.arange(len(df), len(df)+forecast_steps), forecast, label='Forecast', color='red')
```

- Zijn er seizoenspatronen, dan is de data niet stationair en is de basisvoorwaarde van ARIMA niet vervuld → **SARIMA** (`statsmodels.tsa.statespace.sarimax.SARIMAX`). Hier gaan we niet verder op in, maar handig voor projecten later.

## Kalman Filters

- Tweede techniek voor tijdsreeksen; voorbeeldprobleem: een app die de GPS-locatie van een vliegtuig aangeeft en voorspelt waar het vliegtuig naartoe gaat.
- **Probleem met sensors**: GPS bevat een foutmarge; het ruwe traject ziet er "bumpy" uit.
- **Werking**:
  - Begint met een initiële waarde.
  - **Predict-fase**: geeft een nieuwe staat + een (on)zekerheid over die staat.
  - **Update-fase** (een echte meting): kijkt of de staat "dicht genoeg" bij de meting ligt.
- In real life gebruikt in boordnavigatie van vliegtuigen: de filter vermijdt dat een boordcomputer op hol slaat als 1 sensor een afwijkende waarde meet.

## Beste praktijken (Week 7)

- Evalueer regressie met MAE/MSE/R²; probeer dezelfde algoritmes (KNN, SVM, RF) eerst als *Regressor*-variant voor je iets complexers pakt.
- ARIMA-protocol: eerst **ADF-test** → differencing (d), dan **PACF** → p, dan **ACF** → q; vergelijk modellen via de **AIC** (lager = beter).
- Verwar "moving average" in ARIMA niet met een simpel rollend gemiddelde.
- Seizoenspatronen in de data? Gebruik SARIMA in plaats van ARIMA.
- Ruisende sensordata? Kalman filter gladstrijken en voorspellen tegelijk.

## Kernpunten

- Klassieke classifiers hebben ook een regressor-variant; evaluatie via MAE/MSE/R².
- ARIMA = AR(p) + I(d) + MA(q); parameters via ADF-test, PACF en ACF.
- SARIMA voor seizoensdata; Kalman filter voor ruisende metingen (predict + update).