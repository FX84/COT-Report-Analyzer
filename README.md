# Анализ отчетов Commitments of Traders от CFTC

**COT Analyzer** `cot.py` — это Python-скрипт для загрузки, анализа и визуализации еженедельных отчетов Commitments of Traders, публикуемых CFTC (Commodity Futures Trading Commission).

С помощью этого инструмента трейдер или аналитик может:
- Скачать и обработать COT-отчеты по выбранным инструментам;
- Рассчитать ключевые показатели (Net Positions, COT Index, Percentile, Z-score);
- Определить рыночные **экстремумы** и потенциальные **развороты**;
- Визуализировать динамику позиций участников рынка;
- Сохранить результаты анализа в форматах CSV, JSON или SQLite.

---

## ⚙️ Основные возможности

✅ Загрузка COT-отчетов напрямую с сайта **CFTC**  
✅ Кеширование и повторное использование загруженных файлов  
✅ Поддержка основных типов отчетов:
- `legacy`  
- `legacy_futopt`  
- `disaggregated`  
- `tff` (Traders in Financial Futures)

✅ Расчет метрик:
- **Net Positions**
- **COT Index (Williams COT Index)**
- **Percentile (COT Percentile)**
- **Z-score**

✅ Обнаружение:
- Экстремальных значений (overbought/oversold)
- Пересечений уровня 50 (смена доминирования)
- Потенциальных дивергенций (в будущем обновлении)

✅ Визуализация:
- Графики динамики позиций и индексов
- Автоматическое сохранение графиков в PNG

✅ Экспорт результатов:
- `CSV`, `JSON`, `SQLite`

---

## 🧩 Требования

- Python **3.9+**
- Библиотеки:
```bash
pip install pandas numpy matplotlib requests tqdm python-dateutil
````

---

## 🚀 Пример использования

Сканирование и анализ по евро и золоту с построением графиков и экспортом в CSV+JSON:

```bash
python cot.py --markets "EUR,GC" --report disaggregated --window 156 --plot --export csv json
```

Анализ доллара и индекса S&P 500, с экспортом в базу данных SQLite:

```bash
python cot.py --markets "DX,ES" --report tff --export sqlite --db cot.db
```

Принудительная перезагрузка COT-файлов и повышенный уровень логирования:

```bash
python cot.py --markets "CL" --report legacy --force-download --verbose
```

---

## ⚙️ Параметры командной строки

| Аргумент           | Описание                                           | По умолчанию    |
| ------------------ | -------------------------------------------------- | --------------- |
| `--markets`        | Список рынков через запятую (`EUR,GC,DX`)          | —               |
| `--report`         | Тип отчёта (`legacy`, `disaggregated`, `tff`, ...) | `disaggregated` |
| `--start`, `--end` | Фильтрация по датам (формат `YYYY-MM-DD`)          | все даты        |
| `--outdir`         | Папка для сохранения результатов                   | `./cot_out`     |
| `--export`         | Форматы экспорта (`csv`, `json`, `sqlite`)         | `csv`           |
| `--db`             | Путь к базе SQLite (если выбран `sqlite`)          | `cot.db`        |
| `--window`         | Окно (в неделях) для расчёта индексов              | `156`           |
| `--extremes`       | Порог экстремумов в процентах                      | `5`             |
| `--cache`          | Папка кеша загруженных файлов                      | `./cot_cache`   |
| `--force-download` | Принудительная загрузка новых данных               | `False`         |
| `--plot`           | Генерация PNG-графиков                             | `False`         |
| `--verbose`        | Расширенные логи                                   | `False`         |

---

## 📊 Пример результата

После запуска создаются файлы в указанной папке (`cot_out` по умолчанию):

```
cot_out/
├── cot_data.csv
├── cot_data.json
├── cot_EUR_noncommercial.png
└── cot_GC_noncommercial.png
```

### Пример консольного вывода:

```
14:22:03 [INFO] Скачиваю https://www.cftc.gov/dea/futures/deacotdisagg.txt ...
14:22:06 [INFO] Найдено 750 строк для EURO FX
14:22:07 [INFO] Найдено 740 строк для GOLD
14:22:07 [INFO] Экспортировано: cot_data.csv
14:22:07 [INFO] График сохранён: cot_EUR_noncommercial.png
14:22:07 [INFO] Готово!
```

---

## 📈 Расчётные формулы

**Net Positions:**

```
Net = Long - Short
```

**COT Index:**

```
COT Index = 100 * (Net - MIN(Net, N)) / (MAX(Net, N) - MIN(Net, N))
```

**COT Percentile:**

```
Процентиль позиции за последние N недель (0–100%)
```

**Z-score:**

```
Z = (Net - Среднее(Net, N)) / Стандартное_откл(Net, N)
```

**Экстремумы:**

* Перепроданность: `COT Percentile <= extremes`
* Перекупленность: `COT Percentile >= 100 - extremes`

---

## 🧭 Поддерживаемые рынки (по умолчанию)

| Алиас | Название рынка  | Источник цены (yfinance) |
| ----- | --------------- | ------------------------ |
| `EUR` | Euro FX         | `6E=F`                   |
| `DX`  | US Dollar Index | `DX-Y.NYB`               |
| `GC`  | Gold            | `GC=F`                   |
| `CL`  | Crude Oil WTI   | `CL=F`                   |
| `ES`  | E-mini S&P 500  | `ES=F`                   |

Вы можете добавить собственные рынки через JSON/CSV маппинг (см. флаг `--alias-file` в будущих версиях).

---

## 🗂️ Структура проекта

```
COT-Report-Analyzer/
├── cot.py
├── README.md
├── LICENSE
├── cot_cache/
└── cot_out/
```

---

## 🧱 Архитектура кода

* **main()** — парсинг аргументов и общий workflow
* **fetch_cftc_data()** — загрузка COT-файлов и кеширование
* **parse_cftc_text()** — фильтрация и парсинг по ключевым словам
* **compute_metrics()** — расчёт индексов и перцентилей
* **detect_extremes()** — определение экстремумов
* **plot_market()** — визуализация
* **export_data()** — экспорт в выбранные форматы

---

## 🧪 План развития

* Добавить поддержку **цен из Yahoo Finance**
* Поиск **дивергенций** между ценой и позицией участников
* Веб-интерфейс через **Streamlit**
* Telegram-уведомления о новых сигналах
* Кросс-рынковое сравнение участников (Dealer / Managed Money и т.п.)

---

## 📜 Лицензия

Проект распространяется под лицензией [MIT](LICENSE). Вы можете свободно использовать, изменять и распространять код с указанием автора.
