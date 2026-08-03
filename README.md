# FeFET Design-Window Simulator

HfO₂ 기반 FeFET의 목표-적응 허용 설계범위 시뮬레이터 (compact-model, Streamlit).

목표 Memory Window·손실 허용치·저장 방식(binary/MLC)·Δψ_w 를 입력하면,
그 목표를 만족하는 D_it–t_IL 허용 설계범위와 D_it 상한 처방을 계산합니다.

## 로컬 실행
```
pip install -r requirements.txt
streamlit run app.py
```

## 파일
- `app.py` — Streamlit 앱 (메인 파일)
- `plot_tool.py` — 그림 생성
- `device_model.py`, `sweeps.py`, `designmap.py`, `ferro.py`, `units.py`, `plotting.py` — compact-model 엔진

2026 한국전기전자학회 하계학술대회 · 고승민 · 나영은 (한양대 ERICA 전자공학부)
