"""Checkpoint-tests week 4: ARIMA op 3 time series-datasets."""
import pandas as pd
import pytest


# ---------- Oefening 1: TF_BIRTHS ----------
def test_oef1_series_births(week04):
    df = week04.get("series_births")
    assert df is not None, "variabele 'series_births' ontbreekt"
    assert isinstance(df, pd.DataFrame), f"series_births is geen DataFrame maar {type(df)}"
    assert "value" in df.columns, "kolom 'value' ontbreekt in series_births (hernoem MS_NUM_BIRTHS)"
    assert len(df) >= 10000, f"series_births heeft {len(df)} rijen, verwacht >= 10000"
    assert hasattr(df.index, "year"), "series_births heeft geen DatetimeIndex"


def test_oef1_births_2022_daily(week04):
    df = week04.get("births_2022_daily")
    assert df is not None, "variabele 'births_2022_daily' ontbreekt"
    assert isinstance(df, pd.DataFrame), "births_2022_daily is geen DataFrame"
    assert 300 <= len(df) <= 366, f"births_2022_daily heeft {len(df)} rijen, verwacht ~365 (dagcijfers 2022)"
    idx = df.index.min(), df.index.max()
    assert idx[0].year == 2022 and idx[1].year == 2022, \
        f"births_2022_daily bevat data buiten 2022 ({idx[0]} tot {idx[1]})"


def test_oef1_monthly_births(week04):
    df = week04.get("monthly_births")
    assert df is not None, "variabele 'monthly_births' ontbreekt"
    assert isinstance(df, pd.DataFrame), "monthly_births is geen DataFrame"
    assert 360 <= len(df) <= 380, \
        f"monthly_births heeft {len(df)} rijen, verwacht ~374 (maandtotalen van 31 jaar)"


def test_oef1_best_order_births(week04):
    order = week04.get("best_order_births")
    assert order is not None, "variabele 'best_order_births' ontbreekt"
    assert isinstance(order, tuple) and len(order) == 3, \
        f"best_order_births={order} is geen tuple (p, d, q)"
    p, d, q = order
    assert all(v in (0, 1, 2, 4) for v in order), \
        f"best_order_births={order} bevat waarden buiten de grid p=[0,1,2,4], d=[0,1], q=[0,1]"


def test_oef1_rmse_births(week04):
    rmse = week04.get("rmse_births")
    assert rmse is not None, "variabele 'rmse_births' ontbreekt"
    rmse = float(rmse)
    assert rmse > 0, f"rmse_births={rmse} moet positief zijn"
    assert rmse < 2000, f"rmse_births={rmse} is te hoog (verwacht < 2000)"


# ---------- Oefening 2: AQI Duitsland ----------
def test_oef2_aqi_germany(week04):
    df = week04.get("aqi_germany")
    assert df is not None, "variabele 'aqi_germany' ontbreekt"
    assert isinstance(df, pd.DataFrame), "aqi_germany is geen DataFrame"
    assert "value" in df.columns, "kolom 'value' ontbreekt in aqi_germany (hernoem AQI Value)"
    assert len(df) >= 200, f"aqi_germany heeft {len(df)} rijen, verwacht >= 200 (alle Duitse metingen)"


def test_oef2_aqi_filtered(week04):
    df = week04.get("aqi_filtered")
    assert df is not None, "variabele 'aqi_filtered' ontbreekt"
    assert isinstance(df, pd.DataFrame), "aqi_filtered is geen DataFrame"
    assert 50 <= len(df) <= 70, \
        f"aqi_filtered heeft {len(df)} rijen, verwacht ~63 (21 jul tot 22 sep 2022)"


def test_oef2_best_order_aqi(week04):
    order = week04.get("best_order_aqi")
    assert order is not None, "variabele 'best_order_aqi' ontbreekt"
    assert isinstance(order, tuple) and len(order) == 3, \
        f"best_order_aqi={order} is geen tuple (p, d, q)"
    assert all(v in (0, 1, 2, 4) for v in order), \
        f"best_order_aqi={order} bevat waarden buiten de grid p=[0,1,2,4], d=[0,1], q=[0,1]"


def test_oef2_rmse_aqi(week04):
    rmse = week04.get("rmse_aqi")
    assert rmse is not None, "variabele 'rmse_aqi' ontbreekt"
    rmse = float(rmse)
    assert 0 < rmse < 100, f"rmse_aqi={rmse} is te hoog (verwacht < 100)"


# ---------- Oefening 3: Electric Production ----------
def test_oef3_series_electric(week04):
    df = week04.get("series_electric")
    assert df is not None, "variabele 'series_electric' ontbreekt"
    assert isinstance(df, pd.DataFrame), "series_electric is geen DataFrame"
    assert "value" in df.columns, "kolom 'value' ontbreekt in series_electric (hernoem Value)"
    assert 390 <= len(df) <= 400, f"series_electric heeft {len(df)} rijen, verwacht ~397"


def test_oef3_best_order_electric(week04):
    order = week04.get("best_order_electric")
    assert order is not None, "variabele 'best_order_electric' ontbreekt"
    assert isinstance(order, tuple) and len(order) == 3, \
        f"best_order_electric={order} is geen tuple (p, d, q)"
    assert all(v in (0, 1, 2, 4) for v in order), \
        f"best_order_electric={order} bevat waarden buiten de grid p=[0,1,2,4], d=[0,1], q=[0,1]"


def test_oef3_rmse_electric(week04):
    rmse = week04.get("rmse_electric")
    assert rmse is not None, "variabele 'rmse_electric' ontbreekt"
    rmse = float(rmse)
    assert 0 < rmse < 30, f"rmse_electric={rmse} is te hoog (verwacht < 30)"


# ---------- Meerkeuzevragen ----------
def test_theorie_adf(week04):
    antw = week04.get("antwoord_adf")
    assert antw is not None, "variabele 'antwoord_adf' ontbreekt"
    assert str(antw).strip().upper() == "B", \
        f"Fout: antwoord_adf={antw}; p-waarde < 0.05 betekent dat de reeks stationair is"


def test_theorie_seizoen_births(week04):
    antw = week04.get("antwoord_seizoen_births")
    assert antw is not None, "variabele 'antwoord_seizoen_births' ontbreekt"
    assert str(antw).strip().upper() == "B", \
        f"Fout: antwoord_seizoen_births={antw}; er is een piek rond juli-september"
