#!/usr/bin/env python3
"""Download Korean Treasury yields from BOK ECOS and derive curve indicators.

Requires an ECOS Open API key in ECOS_API_KEY and pandas. Example:
  ECOS_API_KEY=... python scripts/korea_treasury_yields.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

STAT_CODE = "817Y002"  # 시장금리(일별)
TENORS = {
    "1y": "010190000",
    "2y": "010195000",
    "3y": "010200000",
    "5y": "010200001",
    "10y": "010210000",
    "20y": "010220000",
    "30y": "010230000",
}


def ecos_rows(api_key: str, item_code: str, start: str, end: str) -> list[dict]:
    """Fetch one series, paging through ECOS's 100-row response limit."""
    all_rows: list[dict] = []
    start_index = 1
    while True:
        stop_index = start_index + 99
        url = (
            "https://ecos.bok.or.kr/api/StatisticSearch/"
            f"{quote(api_key, safe='')}/json/kr/{start_index}/{stop_index}/"
            f"{STAT_CODE}/D/{start}/{end}/{item_code}"
        )
        try:
            with urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"ECOS request failed for {item_code}: {exc}") from exc
        if "RESULT" in payload:
            result = payload["RESULT"]
            raise RuntimeError(f"ECOS error {result.get('CODE')}: {result.get('MESSAGE')}")
        search = payload.get("StatisticSearch", {})
        rows = search.get("row", [])
        all_rows.extend(rows)
        total = int(search.get("list_total_count", len(all_rows)))
        if not rows or len(all_rows) >= total:
            break
        start_index += len(rows)
        time.sleep(0.05)
    return all_rows


def fetch_series(api_key: str, item_code: str, tenor: str, start: str, end: str) -> pd.Series:
    import pandas as pd

    rows = ecos_rows(api_key, item_code, start, end)
    points = {}
    for row in rows:
        value = row.get("DATA_VALUE")
        if value not in (None, "", "-"):
            points[pd.to_datetime(row["TIME"], format="%Y%m%d")] = float(value)
    return pd.Series(points, name=f"yield_{tenor}_pct", dtype="float64").sort_index()


def derive_indicators(yields: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    out = yields.copy()
    out["spread_10y_3y_bp"] = (out["yield_10y_pct"] - out["yield_3y_pct"]) * 100
    out["spread_10y_1y_bp"] = (out["yield_10y_pct"] - out["yield_1y_pct"]) * 100
    # Slope is the 10y-2y level difference; curvature is the butterfly
    # 2*5y - 3y - 10y, both in basis points. These are transparent proxies.
    out["slope_10y_2y_bp"] = (out["yield_10y_pct"] - out["yield_2y_pct"]) * 100
    out["curvature_5y_bp"] = (
        2 * out["yield_5y_pct"] - out["yield_3y_pct"] - out["yield_10y_pct"]
    ) * 100
    for tenor in ("1y", "2y", "3y", "5y", "10y", "20y", "30y"):
        col = f"yield_{tenor}_pct"
        out[f"delta_{tenor}_1d_bp"] = out[col].diff() * 100
        out[f"delta_{tenor}_5d_bp"] = (out[col] - out[col].shift(5)) * 100
    return out


def write_summary(frame: pd.DataFrame, path: Path) -> None:
    import pandas as pd

    yield_cols = [f"yield_{t}_pct" for t in ("1y", "2y", "3y", "5y", "10y", "20y", "30y")]
    numeric = frame[yield_cols]
    stats = numeric.describe().T
    stats = stats.rename(columns={"mean": "평균(%)", "std": "표준편차(%)", "min": "최저(%)", "max": "최고(%)"})
    stats = stats[["평균(%)", "표준편차(%)", "최저(%)", "최고(%)"]].round(4)
    complete = frame[yield_cols].dropna()
    lines = [
        "# 한국 국고채 기간구조 분석 요약", "",
        f"- 데이터 기간: {frame.index.min():%Y-%m-%d} ~ {frame.index.max():%Y-%m-%d}",
        f"- 전체 관측치: {len(frame):,}일 (7개 만기 모두 존재하는 일자: {len(complete):,}일)",
        "- 출처: 한국은행 ECOS, 통계표 817Y002 시장금리(일별); 수익률 단위는 %", "",
        "## 만기별 기술통계", "", stats.to_markdown(), "",
        "## 기간구조 특성", "",
    ]
    for col, label in (("spread_10y_3y_bp", "10년-3년 스프레드"), ("spread_10y_1y_bp", "10년-1년 스프레드"),
                       ("slope_10y_2y_bp", "10년-2년 기울기 대리변수"),
                       ("curvature_5y_bp", "5년 butterfly 곡률 대리변수")):
        s = frame[col].dropna()
        lines.append(f"- {label}: 평균 {s.mean():.2f}bp, 표준편차 {s.std():.2f}bp, 최저 {s.min():.2f}bp, 최고 {s.max():.2f}bp")
    inverted = (frame["spread_10y_3y_bp"] < 0).sum()
    lines.extend([f"- 10년-3년 역전 관측: {inverted:,}일 ({inverted / max(len(frame), 1):.1%})", "",
                  "## 일별 변동 특성", ""])
    for tenor in ("1y", "2y", "3y", "5y", "10y", "20y", "30y"):
        s = frame[f"delta_{tenor}_1d_bp"].dropna()
        lines.append(f"- {tenor}: 1일 변동 표준편차 {s.std():.2f}bp, 5일 변동 표준편차 {frame[f'delta_{tenor}_5d_bp'].std():.2f}bp")
    lines.extend(["", "## 해석 메모", "", "기울기와 곡률은 지정 만기의 수익률로 계산한 단순 대리변수이며 Nelson–Siegel 등 모형 기반 추정치가 아닙니다. 5D 변동은 직전 5개 관측 거래일 대비 변화입니다. 발행 공표의 인과효과나 최적 조달 시점은 이 기술통계만으로 식별되지 않으며, 후속 이벤트 스터디에는 공표일·발행계획 및 통제변수가 필요합니다.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="19900101", help="시작일 YYYYMMDD (기본: 19900101)")
    parser.add_argument("--end", default=date.today().strftime("%Y%m%d"), help="종료일 YYYYMMDD (기본: 오늘)")
    parser.add_argument("--out-dir", default="data/research", help="산출물 디렉터리")
    args = parser.parse_args()
    if len(args.start) != 8 or len(args.end) != 8 or args.start > args.end:
        parser.error("--start/--end는 YYYYMMDD이고 시작일은 종료일보다 빠르거나 같아야 합니다")
    api_key = os.environ.get("ECOS_API_KEY")
    if not api_key:
        print("ECOS_API_KEY 환경변수를 설정하세요. ECOS Open API 키가 필요합니다.", file=sys.stderr)
        return 2
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        print("pandas가 없습니다. 먼저 'python -m pip install -r requirements.txt'를 실행하세요.", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    series = {}
    for tenor, code in TENORS.items():
        print(f"Fetching {tenor} ...", file=sys.stderr)
        series[tenor] = fetch_series(api_key, code, tenor, args.start, args.end)
    yields = pd.concat(series.values(), axis=1).sort_index()
    yields.index.name = "date"
    result = derive_indicators(yields)
    result.to_csv(out_dir / "korea_treasury_yields_daily.csv", float_format="%.6f", encoding="utf-8-sig")
    write_summary(result, out_dir / "korea_treasury_yields_summary.md")
    print(f"Saved {len(result)} rows to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
