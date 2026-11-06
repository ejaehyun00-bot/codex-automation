# codex-automation

맥미니(M4) Codex CLI + GitHub Actions 자동화 파이프라인 전용 비공개 저장소입니다.
Pickle AI가 이슈에 `codex` 라벨을 달아 발행하면 로컬 맥미니 러너가 자율 코딩을 수행하고 Draft PR을 생성합니다.

## 한국 국고채 기간구조 분석

`scripts/korea_treasury_yields.py`는 한국은행 ECOS의 `817Y002`(시장금리, 일별)에서 국고채 1·2·3·5·10·20·30년 수익률을 내려받고, 스프레드·기울기/곡률 대리변수와 1/5 거래일 금리변동을 계산합니다. CSV와 데이터 기간·관측치 수·기술통계를 담은 Markdown 요약을 `data/research/`에 기록합니다.

실행 전 ECOS Open API 키를 발급하고 의존성을 설치합니다.

```sh
python -m pip install -r requirements.txt
export ECOS_API_KEY="발급받은 키"
python scripts/korea_treasury_yields.py --start 20000101 --end 20261231
```

`--start`, `--end`는 `YYYYMMDD` 형식이며 기본 시작일은 1990-01-01, 종료일은 실행일입니다. ECOS 응답에 값이 없는 만기/일자는 결측치로 보존합니다. 수익률은 %, 스프레드·변동 지표는 bp입니다. 기울기(10년-2년)와 곡률(`2×5년−3년−10년`)은 단순 대리변수이고, 5D 변화는 이전 5개 관측 거래일 대비입니다. 스크립트의 기술통계만으로 국채 발행 공표의 인과효과나 최적 조달 시점을 결론 내릴 수는 없습니다.
