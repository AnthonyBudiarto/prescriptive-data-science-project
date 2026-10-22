# Prescriptive Data Science — Assignment 1

Optimal portfolio construction on the 50 largest IDX stocks (as of August 2026),
following the workflow: **Public Data → Descriptive → Predictive → Prescriptive → Optimal Decision**.

## Project structure
```
PRESCRIPTIVE DATA SCIENCE/
├── code/
│   ├── 01_fetch_data.py        # Public Data: download prices from Yahoo Finance
│   ├── 02_descriptive.py       # Descriptive analytics + figures
│   ├── 03_predictive.py        # Predictive models (Ridge, RandomForest)
│   ├── 04_prescriptive.py      # Markowitz mean-variance optimization
│   ├── 05_optimal_decision.py  # Out-of-sample backtest + final recommendation
│   └── run_all.py              # Runs the full pipeline end-to-end
├── data/
│   ├── tickers_idx_top50.csv   # stock universe (ticker, company, sector)
│   ├── prices.csv              # cleaned adjusted-close price matrix
│   ├── returns.csv             # daily log returns
│   └── valid_tickers.csv       # tickers that passed cleaning
├── output/                     # all CSV tables + PNG figures
├── report/
│   └── Report.md               # full report (with executive summary)
├── AI_Use_Declaration_Form.md  # to be completed by the student
├── requirements.txt
└── README.md
```

## How to run (no IDE needed — just a terminal)

**Prerequisites:** Python 3.9+ and an internet connection (prices are pulled from Yahoo Finance).

### Step 1 — open a terminal in this project folder
- Windows: open the `PRESCRIPTIVE DATA SCIENCE` folder, type `cmd` in the address bar, press Enter.
- macOS/Linux: `cd` into the project folder.

### Step 2 — install dependencies (one time)
```bash
pip install -r requirements.txt
```

### Step 3 — run the whole pipeline with one command
```bash
python code/run_all.py
```
This runs all five stages in order and regenerates every table and figure into `output/`.

**Or** run each stage individually to inspect them one at a time:
```bash
python code/01_fetch_data.py        # Public Data
python code/02_descriptive.py       # Descriptive
python code/03_predictive.py        # Predictive
python code/04_prescriptive.py      # Prescriptive
python code/05_optimal_decision.py  # Optimal Decision
```

### Where to look at results
- **Report:** `report/Report.md` (open in any Markdown viewer, or VS Code / browser).
- **Figures (charts):** `output/*.png`
- **Tables (numbers):** `output/*.csv`

## How to tell it worked (success indicators)
1. Each command finishes **without a red error / traceback** and prints a stage summary.
2. `run_all.py` ends with: `Pipeline complete. See output/ for all tables and figures.`
3. `output/` contains **8 PNG charts** and **10 CSV tables**.
4. Fetch stage reports `952 trading days x 47 tickers` (3 dropped: AADI, CDIA, EMAS —
   too little price history on Yahoo Finance).
5. The final recommended portfolio weights **sum to 1.0** (printed at the end).

## Notes
- Prices are pulled live from Yahoo Finance, so numbers may shift slightly between runs.
- Key parameters (risk-free rate, prediction horizon, max weight) are constants at the top
  of each script.
- The report's honest headline finding: the naive Max-Sharpe portfolio looked great in-sample
  (Sharpe 2.72) but underperformed equal-weight out-of-sample — a lesson in estimation error.
