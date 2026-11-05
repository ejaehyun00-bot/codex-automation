#!/usr/bin/env python3
"""Reproduce POE5 Chapter 12 Exercises 12.7 and 12.22.

Run: python ch12_unstable_variables.py
Requires pandas, numpy, matplotlib, statsmodels, and openpyxl.
Downloads official POE5 Excel datasets and writes figures/results in ch12_output/.
"""
from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

OUT = Path("ch12_output")
OUT.mkdir(exist_ok=True)
BASE = "https://www.principlesofeconometrics.com/poe5/data/xlsx/{}.xlsx"


def load_data(name):
    path = OUT / f"{name}.xlsx"
    if not path.exists():
        try:
            with urllib.request.urlopen(BASE.format(name), timeout=30) as response:
                path.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(f"Download failed for {name}.xlsx. Save the official POE5 Excel file as {path} and rerun.") from exc
    xls = pd.ExcelFile(path)
    df = pd.read_excel(path, sheet_name=xls.sheet_names[0])
    df.columns = [str(c).strip() for c in df.columns]
    return df


def getcol(df, names):
    lookup = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return pd.to_numeric(df[lookup[name.lower()]], errors="coerce").dropna()
    raise KeyError(f"Expected column among {names}; found {list(df.columns)}")


def adf(series, label, regression="c"):
    stat, p, lag, nobs, crit, _ = adfuller(series.dropna(), regression=regression, autolag="BIC")
    return dict(series=label, lag=lag, n=nobs, tau=stat, p=p, cv5=crit["5%"], reject=stat < crit["5%"])


def plot_series(series_list, filename, title):
    fig, axes = plt.subplots(len(series_list), 1, figsize=(10, 2.7 * len(series_list)), squeeze=False)
    for ax, (label, values, start) in zip(axes[:, 0], series_list):
        ax.plot(np.arange(start, start + len(values)) / 1, values, lw=1.1)
        ax.set_title(label)
        ax.grid(alpha=.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def engle_granger(y, x, spec, label):
    t = np.arange(len(y), dtype=float)
    cols = []
    if spec in ("c", "ct"):
        cols.append(np.ones(len(y)))
    if spec == "ct":
        cols.append(t)
    cols.append(np.asarray(x))
    fit = sm.OLS(np.asarray(y), np.column_stack(cols)).fit()
    # coint supplies MacKinnon residual-based critical values for the chosen deterministic terms.
    tau, p, crit = coint(y, x, trend=spec, autolag="BIC")
    return dict(case=label, tau=tau, p=p, cv5=crit[1], reject=tau < crit[1], beta=fit.params[-1])


def f(x):
    return f"{x:.4f}"


def main():
    report = ["# Ch. 12: Exercises 12.7 and 12.22 — Reproducible results", "",
              "Official POE5 Excel files; ADF lag order selected by BIC. ADF alternative is one-sided.", ""]
    us = load_data("usmacro")
    us_series = [(v, getcol(us, [v]), 1948) for v in ("U", "G", "INF")]
    plot_series(us_series, "usmacro_levels.png", "US macro series, 1948Q1–2016Q1")
    rows = []
    for name, values, _ in us_series:
        r = adf(values, name, "c")
        rows.append(r)
        if not r["reject"]:
            rows.append(adf(values.diff().dropna(), "Δ" + name, "c"))
    report += ["## Exercise 12.7 — US macro", "",
               "The level plots are in `usmacro_levels.png`. Use an intercept and no deterministic trend: "
               "the series fluctuate around nonzero means without a sustained linear deterministic trend. "
               "ADF: Δyₜ = α + γyₜ₋₁ + ΣφᵢΔyₜ₋ᵢ + εₜ; H₀: γ=0 (unit root), H₁: γ<0 (stationary).", "",
               "| Series | ADF lag (BIC) | τ | 5% critical | p-value | Decision |", "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        report.append(f"| {r['series']} | {r['lag']} | {f(r['tau'])} | {f(r['cv5'])} | {f(r['p'])} | {'Reject H₀' if r['reject'] else 'Do not reject H₀'} |")
    report += ["", "POE5 selected answers: U uses one augmentation lag (p=0.0003); G uses zero (p<0.0001); "
               "INF uses two (p=0.0017). All reject a unit root at 5%, so U, G, and INF are I(0) under the "
               "textbook specification. The table above reports computed test statistics and critical values.", ""]

    data = load_data("mexico")
    mexico = getcol(data, ["mexico", "mex", "gdp_mex", "gdpmex", "mexgdp"])
    usa = getcol(data, ["us", "usa", "gdp_us", "gdpusa", "usgdp", "united states"])
    n = min(len(mexico), len(usa))
    mexico, usa = mexico.iloc[:n].reset_index(drop=True), usa.iloc[:n].reset_index(drop=True)
    plot_series([("Mexico real GDP", mexico, 1980), ("US real GDP", usa, 1980)], "mexico_levels.png", "Real GDP, 1980Q1–2006Q3")
    results = [engle_granger(mexico, usa, s, label) for s, label in
               [("n", "No intercept/no trend"), ("c", "Intercept/no trend"), ("ct", "Intercept/linear trend")]]
    report += ["## Exercise 12.22 — Mexico and US real GDP", "",
               f"Matched sample: {n} quarterly observations (nominal target 1980Q1–2006Q3). "
               "Levels are plotted in `mexico_levels.png`. The three first-stage equations include "
               "(1) neither constant nor trend, (2) an intercept, or (3) an intercept and linear trend. "
               "Residual-test H₀: no cointegration; H₁: cointegration. Critical values are MacKinnon "
               "Engle–Granger values, not ordinary ADF values.", "",
               "| Case | β on US GDP | τEG | 5% critical | p-value | Decision |", "|---|---:|---:|---:|---:|---|"]
    for r in results:
        report.append(f"| {r['case']} | {f(r['beta'])} | {f(r['tau'])} | {f(r['cv5'])} | {f(r['p'])} | {'Reject no cointegration' if r['reject'] else 'Do not reject'} |")
    report += ["", "### 해석", "",
               "(a) 세 사양의 결론을 비교한다. 결정항 선택에 따라 결과가 달라지면 결론이 사양에 민감하다. "
               "우선 사양은 GDP 수준의 추세와 경제이론에 근거해 정해야 한다.", "",
               "(b) 비례적 수렴은 절편과 추세가 없는 관계 Mexico GDP = β·US GDP를 뜻한다. 첫 번째 "
               "케이스에서 공적분 귀무가설을 기각하면 자료가 이를 지지한다. β=1인지 여부는 별도 제한 검정이 필요하다.", "",
               "(c) 공적분이 없다면 수준 회귀를 장기 균형식으로 해석하지 않는다. ΔMexico GDP와 ΔUS GDP의 "
               "회귀 또는 차분 변수 VAR로 단기 관계를 분석한다. I(1) 변수들이 공적분되면 ECM/VECM을 쓴다. "
               "로그 GDP 성장률, 시차, 구조변화도 점검한다.", "",
               "### 재현", "",
               "실행: `python ch12_unstable_variables.py`. 결과 Markdown과 PNG 및 내려받은 Excel 파일은 `ch12_output/`에 생성된다. "
               "외부 다운로드가 막히면 공식 POE5 Excel 파일을 해당 폴더에 `usmacro.xlsx`, `mexico.xlsx`로 저장 후 재실행한다.", ""]
    (OUT / "results.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
