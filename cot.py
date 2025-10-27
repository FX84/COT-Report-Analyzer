#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cot.py — Commitments of Traders (COT) Analyzer
Описание:
    Скрипт загружает, парсит и анализирует COT-отчёты CFTC,
    рассчитывает индикаторы (net positions, COT Index, Percentile, Z-score),
    выявляет экстремумы и сигналы, визуализирует результаты и экспортирует в CSV/JSON/SQLite.
"""

import argparse
import os
import sys
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import logging
from datetime import datetime
from io import StringIO
from tqdm import tqdm
from dateutil import parser as dateparser

# --- Константы ----------------------------------------------------------

CFTC_BASE_URLS = {
    "legacy": "https://www.cftc.gov/dea/futures/deacot.txt",
    "legacy_futopt": "https://www.cftc.gov/dea/futures/deacot_futopt.txt",
    "disaggregated": "https://www.cftc.gov/dea/futures/deacotdisagg.txt",
    "tff": "https://www.cftc.gov/dea/futures/deatif.txt",
}

DEFAULT_WINDOW = 156
DEFAULT_EXTREMES = 5

# Базовый словарь алиасов рынков
MARKET_ALIASES = {
    "EUR": {"display": "Euro FX", "yfinance": "6E=F", "keyword": "EURO FX"},
    "DX": {"display": "US Dollar Index", "yfinance": "DX-Y.NYB", "keyword": "US DOLLAR INDEX"},
    "GC": {"display": "Gold", "yfinance": "GC=F", "keyword": "GOLD"},
    "CL": {"display": "Crude Oil WTI", "yfinance": "CL=F", "keyword": "CRUDE OIL"},
    "ES": {"display": "E-mini S&P 500", "yfinance": "ES=F", "keyword": "S&P 500"},
}


# --- Утилиты ------------------------------------------------------------

def setup_logger(verbose: bool):
    """Настраивает логгер."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def ensure_dir(path):
    """Создаёт директорию, если она отсутствует."""
    os.makedirs(path, exist_ok=True)


# --- Загрузка и парсинг данных ------------------------------------------

def fetch_cftc_data(report: str, cache_dir: str, force_download=False) -> str:
    """
    Скачивает файл COT соответствующего типа.
    Возвращает путь к локальному файлу.
    """
    ensure_dir(cache_dir)
    url = CFTC_BASE_URLS.get(report)
    if not url:
        raise ValueError(f"Неизвестный тип отчёта: {report}")

    filename = os.path.join(cache_dir, f"{report}.txt")

    if os.path.exists(filename) and not force_download:
        logging.info(f"Использую кешированный файл {filename}")
        return filename

    logging.info(f"Скачиваю {url} ...")
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"Ошибка загрузки {url}: {r.status_code}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(r.text)

    logging.info(f"Файл сохранён: {filename}")
    return filename


def parse_cftc_text(filepath: str, keyword: str) -> pd.DataFrame:
    """
    Извлекает строки из CFTC-файла по ключевому слову (названию рынка).
    Возвращает DataFrame.
    """
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    # Фильтруем строки по ключевому слову
    relevant = [line for line in lines if keyword.upper() in line.upper()]
    if not relevant:
        raise ValueError(f"Не найдено записей по ключевому слову: {keyword}")

    # Конвертируем в таблицу (поля разделены несколькими пробелами)
    data = pd.read_csv(StringIO("\n".join(relevant)), sep=r"\s{2,}", engine="python")
    logging.info(f"Найдено {len(data)} строк для {keyword}")

    # Попытка нормализовать имена колонок
    data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
    return data


# --- Расчёт метрик ------------------------------------------------------

def compute_metrics(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Рассчитывает net, COT Index, Percentile и Z-score для выбранных колонок."""
    # Определяем доступные поля (по отчёту)
    possible_groups = ["noncommercial", "managed_money", "leveraged_funds", "dealer_intermediary", "asset_manager"]
    for group in possible_groups:
        long_col = f"long_{group}"
        short_col = f"short_{group}"
        if long_col in df.columns and short_col in df.columns:
            df[f"net_{group}"] = df[long_col] - df[short_col]
            x = df[f"net_{group}"]
            df[f"cot_index_{group}"] = 100 * (x - x.rolling(window).min()) / (x.rolling(window).max() - x.rolling(window).min())
            df[f"cot_percentile_{group}"] = x.rolling(window).apply(lambda s: pd.Series(s).rank(pct=True).iloc[-1] * 100)
            df[f"zscore_{group}"] = (x - x.rolling(window).mean()) / x.rolling(window).std()
    return df


def detect_extremes(df: pd.DataFrame, extremes: int) -> pd.DataFrame:
    """Помечает экстремумы по COT Percentile."""
    for c in df.columns:
        if "cot_percentile_" in c:
            df[f"extreme_high_{c}"] = df[c] >= (100 - extremes)
            df[f"extreme_low_{c}"] = df[c] <= extremes
    return df


# --- Экспорт ------------------------------------------------------------

def export_data(df: pd.DataFrame, formats: list, outdir: str, db_path: str = None):
    """Сохраняет результаты в CSV, JSON, SQLite."""
    ensure_dir(outdir)
    if "csv" in formats:
        path = os.path.join(outdir, "cot_data.csv")
        df.to_csv(path, index=False)
        logging.info(f"CSV экспортирован: {path}")

    if "json" in formats:
        path = os.path.join(outdir, "cot_data.json")
        df.to_json(path, orient="records", indent=2)
        logging.info(f"JSON экспортирован: {path}")

    if "sqlite" in formats:
        if not db_path:
            db_path = os.path.join(outdir, "cot.db")
        with sqlite3.connect(db_path) as conn:
            df.to_sql("cot", conn, if_exists="replace", index=False)
        logging.info(f"SQLite экспортирован: {db_path}")


# --- Графики ------------------------------------------------------------

def plot_market(df: pd.DataFrame, outdir: str, market: str, group="noncommercial"):
    """Строит базовый график для рынка."""
    ensure_dir(outdir)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_title(f"{market} — Net {group.title()}")
    ax1.plot(df["date"], df[f"net_{group}"], label="Net Position", color="tab:blue")
    ax1.set_ylabel("Net")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df[f"cot_index_{group}"], label="COT Index", color="tab:orange", alpha=0.7)
    ax2.set_ylabel("COT Index")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()

    filename = os.path.join(outdir, f"cot_{market}_{group}.png")
    plt.savefig(filename)
    plt.close()
    logging.info(f"График сохранён: {filename}")


# --- Основная логика ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Commitments of Traders (COT) Analyzer")
    parser.add_argument("--markets", required=True, help="Список рынков через запятую, например EUR,GC")
    parser.add_argument("--report", default="disaggregated", choices=list(CFTC_BASE_URLS.keys()), help="Тип отчёта")
    parser.add_argument("--start", type=str, default=None, help="Дата начала (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Дата окончания (YYYY-MM-DD)")
    parser.add_argument("--outdir", default="./cot_out", help="Папка для экспорта")
    parser.add_argument("--export", nargs="+", choices=["csv", "json", "sqlite"], default=["csv"], help="Форматы экспорта")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Окно для расчёта индексов")
    parser.add_argument("--extremes", type=int, default=DEFAULT_EXTREMES, help="Порог экстремумов (в %)")
    parser.add_argument("--cache", default="./cot_cache", help="Папка для кеша")
    parser.add_argument("--force-download", action="store_true", help="Принудительно скачать свежие файлы")
    parser.add_argument("--plot", action="store_true", help="Строить графики")
    parser.add_argument("--verbose", action="store_true", help="Расширенный лог")
    args = parser.parse_args()

    setup_logger(args.verbose)
    ensure_dir(args.outdir)

    report = args.report
    filepath = fetch_cftc_data(report, args.cache, args.force_download)

    markets = [m.strip().upper() for m in args.markets.split(",")]
    all_data = []

    for market in tqdm(markets, desc="Анализ рынков"):
        if market not in MARKET_ALIASES:
            logging.warning(f"Неизвестный рынок: {market}, пропускаю.")
            continue

        alias = MARKET_ALIASES[market]
        try:
            df = parse_cftc_text(filepath, alias["keyword"])
            df["date"] = pd.to_datetime(df["as_of_date_in_form_yyyymmdd"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            df["market"] = market
            df = compute_metrics(df, args.window)
            df = detect_extremes(df, args.extremes)
            all_data.append(df)
            if args.plot:
                plot_market(df, args.outdir, market)
        except Exception as e:
            logging.error(f"Ошибка при обработке {market}: {e}")

    if not all_data:
        logging.error("Нет данных для экспорта.")
        sys.exit(1)

    final_df = pd.concat(all_data, ignore_index=True)
    export_data(final_df, args.export, args.outdir)

    logging.info("Готово!")


if __name__ == "__main__":
    main()