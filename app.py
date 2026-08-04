# -*- coding: utf-8 -*-
"""
HfO2 FeFET 허용 설계범위 시뮬레이터 (Streamlit)
────────────────────────────────────────────────────────
실행:  cd 코드 ;  streamlit run app/app.py
배포:  GitHub push → Streamlit Community Cloud (main file = app/app.py)
        ※ 폴더/파일명이 한글이면 배포환경에서 문제될 수 있음 → 배포 시 ASCII 권장.

물리·모델은 기존 엔진(src/) 그대로 호출한다. 이 앱은 UI·그림만 담당.
   base 파라미터 → sweep_2d → window_mask / dit_upper_bounds
목표 MW = (N-1) * ΔV_level  (binary: N=2).  Δψ_w 밴드는 처방에 ∝ 1/Δψ_w 로 전파.
표시 문구는 개조식(-ㅁ/-음) 종결.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import numpy as np
import streamlit as st

from device_model import solve_device
from sweeps import sweep_2d, sweep_1d
from designmap import window_mask, dit_upper_bounds
import plot_tool

st.set_page_config(page_title="FeFET 목표-적응 설계범위 시뮬레이터", layout="wide")
st.markdown("""
<style>
/* 요약 카드(통과율·baseline MW·목표 달성)를 폰에서도 가로 한 줄로 */
div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]){flex-wrap:nowrap !important;}
div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div{min-width:0 !important;}
</style>
""", unsafe_allow_html=True)

# ── 슬라이더 값 숫자 색: 손실 허용치=빨강, 레벨당 마진=검정 (막대는 Streamlit 한계로 파랑 유지) ──
#   (앵커 다음 슬라이더의 값 숫자만 CSS로 색칠 — 이 부분은 확실히 동작)
st.markdown(
    """
    <style>
    div[data-testid="stElementContainer"]:has(.anch-dv),
    div[data-testid="stElementContainer"]:has(.anch-loss){display:none !important;}
    div[data-testid="stElementContainer"]:has(.anch-dv) + div[data-testid="stElementContainer"] [data-testid="stSliderThumbValue"]{color:#111 !important;}
    div[data-testid="stElementContainer"]:has(.anch-loss) + div[data-testid="stElementContainer"] [data-testid="stSliderThumbValue"]{color:#d1341f !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── 헤더 ───────────────────────────
st.header(" HfO₂ FeFET 허용 설계범위 시뮬레이터")
st.caption(
    "Compact-model 시뮬레이션 도구임 (소자 제작/측정이 아님).|n"
    "목표 Memory Window(MW)를 정하면, 그 목표를 만족하는 허용 설계범위(D_it–t_IL)와 D_it 상한 처방을 계산함."
)

# ─────────────────────────── 사이드바 입력 ───────────────────────────
sb = st.sidebar
# ── 처음 쓰는 사람을 위한 값 설명 (펼쳐보기) ──
with sb.expander("ℹ️ 각 값이 무슨 뜻인가요?  (처음이면 눌러서 펼쳐보기)"):
    st.markdown(
        "이 도구는 \"이 목표를 만족하려면 계면(D_it·t_IL)을 얼마나 좋게 만들어야 하나\" 를 계산함.\n\n"
        "- ⚙️ 설계 노브 — 우리가 *만드는* 소자의 구조(강유전체 두께 등). 설계자가 직접 정하는 값.\n"
        "- 🎯 스펙/목표 — 우리가 *요구하는* 성능. 목표 MW(memory window)를 정하면 도구가 허용 범위를 답해줌.\n"
        "- 📉 불확실성 (Δψ_w) — 실험으로 아직 정확한 값을 못 정한 물리값. 하나로 못 박지 않고, "
        "가능한 범위(1.0–2.0 V)를 그래프에 띠(밴드) 로 함께 그려 불확실성의 폭을 보여줌.\n"
        "- 🔲 격자 — 지도 계산 해상도. 매끄러움·속도만 바뀌고 결과 수치엔 거의 영향 없음.\n\n"
        "각 항목 이름 옆 ❓ 에 마우스를 올리면 더 자세한 설명이 나옴."
    )

sb.markdown("### ⚙️ 설계 노브 (스택)")
sb.caption("우리가 *만드는* 소자의 물리 구조 — 직접 정하는 값")
t_fe = sb.slider(
    "t_FE — 강유전체 두께 [nm]", 5.0, 20.0, 10.0, 0.5,
    help="강유전체(HfO₂/HZO) 층의 두께임. 두꺼울수록 메모리 윈도우(MW=두 저장 상태의 "
         "문턱전압 차)가 거의 선형으로 커짐. 목표 MW가 큰 MLC에선 이 값을 키워야 함. "
         "(기준 10 nm)")
pr = sb.slider(
    "P_r — 잔류분극 [μC/cm²]", 5.0, 25.0, 15.0, 1.0,
    help="잔류분극 — 전압을 끊어도 강유전체에 남아 있는 분극량임. 클수록 두 상태의 "
         "MW가 커지지만 점점 포화함. 포화분극 P_s는 자동으로 P_r의 1.3배로 잡히고, "
         "P_r=15에선 논문 기준값 P_s=20으로 고정됨. (기준 15)")
with sb.expander("고급 (기본 고정 권장)"):
    ec = st.number_input(
        "E_c [MV/cm]", 0.5, 3.0, 1.0, 0.1,
        help="항전계 — 분극을 뒤집는 데 필요한 전계. 정규화 기준선 MW_ref=2·E_c·t_FE 에 "
             "들어감. 보통 1.0으로 고정 권장.")
    na = st.number_input(
        "N_a [cm⁻³]", 1e16, 1e18, 1e17, format="%.0e",
        help="p-Si 기판의 도핑(억셉터) 농도. 표면 전위·공핍층 폭에 영향을 줌. 보통 1e17 고정.")

sb.markdown("### 🎯 스펙 / 목표")
sb.caption("우리가 *요구하는* 성능 — 목표를 정하면 허용 범위를 답해줌")
mode = sb.segmented_control(
    "저장 방식", ["binary", "MLC"], default="binary",
    help="binary = 한 셀에 2단계(1비트) 저장. MLC = 여러 단계(다중비트) 저장 → 더 큰 MW가 필요함.")
mode = mode or "binary"
if mode == "MLC":
    n_lv = sb.segmented_control(
        "레벨 수 N", [2, 3, 4], default=3, format_func=lambda n: f"{n}단계",
        help="한 셀에 저장하는 단계 수임. N단계를 구분하려면 창(MW)이 (N−1)×레벨마진 "
             "이상 필요 → 목표 MW = (N−1)·ΔV_level. (예: 3단계·마진 1.0V → 목표 2.0 V)")
    n_lv = n_lv or 3
else:
    n_lv = 2
sb.markdown('<span class="anch-loss"></span>', unsafe_allow_html=True)
loss_max = float(sb.slider(
    "🔴 손실 허용치 MW_loss_max [%]", 10, 50, 30, 5,
    help="이상적 기준선(MW_ref) 대비 MW가 얼마나 줄어드는 것까지 허용할지(상대 기준). "
         "예: 30%면 'MW가 기준선의 70% 이상이면 합격'. 논문 기본값 30%."))
sb.markdown('<span class="anch-dv"></span>', unsafe_allow_html=True)
dv = sb.slider(
    "⚫ 레벨당 마진 ΔV_level [V]", 0.5, 1.5, 1.0, 0.1,
    help="인접한 두 저장 레벨 사이에 확보해야 할 최소 문턱전압 간격(읽기 여유)임. "
         "클수록 안전하지만 요구되는 MW가 커짐.")
target = round((n_lv - 1) * dv, 3)
sb.metric("→ 목표 MW (절대 기준)", f"{target:.2f} V",
          help="실제로 만족해야 하는 절대 기준값임. = (N−1)×ΔV_level. "
               "binary(N=2)이면 ΔV_level과 같음.")

sb.markdown("### 📉 불확실성")
sb.caption("실험으로 아직 값을 못 정한 항목 — 하나로 안 정하고, 가능한 범위(1.0~2.0V)를 그래프에 띠로 함께 표시함")
dpsi = sb.slider(
    "Δψ_w (잠정값) [V]", 1.0, 2.0, 1.5, 0.1,
    help="write(기록) 시 반도체 표면 전위가 프로그램/이레이즈 상태 사이에서 흔들리는 폭임. "
         "계면 트랩이 MW를 깎는 양을 정하는 물리값인데, 실험 앵커를 아직 못 잡아 잠정값(1.5 V)임. "
         "허용 D_it 상한은 이 값에 정확히 반비례함 (1.0V→×1.5, 2.0V→×0.75).")
show_band = sb.checkbox(
    "Δψ_w 1.0–2.0 V 밴드 표시", value=True,
    help="처방 곡선에 Δψ_w 1.0–2.0 V 불확실성 띠를 함께 그림.")

sb.markdown("### 🔲 격자 (계산 해상도)")
sb.caption("많을수록 매끄럽지만 느림 — 결과 수치엔 거의 영향 없음")
n_dit = sb.segmented_control(
    "D_it 점수 (가로축)", [20, 40, 60], default=40,
    help="지도 가로축(D_it, 계면 트랩 밀도)을 몇 개 점으로 계산할지. 많을수록 곡선이 "
         "매끄럽지만 느려짐. 논문 정본 = 40.")
n_dit = int(n_dit or 40)
n_til = sb.segmented_control(
    "t_IL 점수 (세로축)", [12, 24, 40], default=24,
    help="지도 세로축(t_IL, 계면층 두께)을 몇 개 점으로 계산할지. 논문 정본 = 24.")
n_til = int(n_til or 24)

sb.divider()
sb.caption("⚠️ 실측 아님 · compact model 계산 결과. 입력 물리는 선행연구 [1]–[4].")


# ─────────────────────────── 계산 (캐시) ───────────────────────────
def _ps(pr):
    """P_s 규약(승민 회신 2026-08-03): 기본 P_r=15 는 논문 baseline 값 20 고정
    (→ 첫 화면이 곧 논문 기준점, 회귀검증 성립), 그 외에는 1.3·P_r (P_s>P_r 유지)."""
    return 20.0 if abs(pr - 15.0) < 1e-9 else 1.3 * pr


def _base(t_fe, pr, dpsi, ec, na):
    return dict(t_FE=t_fe, t_IL=1.0, Pr=pr, Ps=_ps(pr), Ec=ec, Na=na,
                Qf=0.0, Dit=1e11, dpsi_w=dpsi)


@st.cache_data(show_spinner=False)
def compute_map(t_fe, pr, dpsi, ec, na, n_dit, n_til):
    base = _base(t_fe, pr, dpsi, ec, na)
    dit = np.logspace(11, 13, n_dit)
    til = np.linspace(0.5, 2.0, n_til)
    return sweep_2d("Dit", dit, "t_IL", til, base)


@st.cache_data(show_spinner=False)
def compute_baseline(t_fe, pr, dpsi, ec, na):
    return solve_device(_base(t_fe, pr, dpsi, ec, na))


@st.cache_data(show_spinner=False)
def min_tfe_for_target(pr, dpsi, ec, na, target):
    for tfe in np.linspace(5, 20, 61):
        r = solve_device(dict(t_FE=float(tfe), t_IL=1.0, Pr=pr, Ps=_ps(pr),
                              Ec=ec, Na=na, Qf=0.0, Dit=1e11, dpsi_w=dpsi))
        if r["MW"] >= target:
            return float(tfe)
    return None


@st.cache_data(show_spinner=False)
def table_bounds(t_fe, pr, dpsi, ec, na, loss_max, target):
    base = _base(t_fe, pr, dpsi, ec, na)
    m = sweep_2d("Dit", np.logspace(11, 13, 200), "t_IL",
                 np.array([0.5, 1.0, 1.5, 2.0]), base)
    return dit_upper_bounds(m, loss_max, target)


m = compute_map(t_fe, pr, dpsi, ec, na, n_dit, n_til)
mask = window_mask(m, loss_max, target)
frac = 100.0 * mask.mean()
rows = dit_upper_bounds(m, loss_max, target)
r0 = compute_baseline(t_fe, pr, dpsi, ec, na)
mw_max = float(m["MW"].max())

# ─────────────────────────── 요약 카드 ───────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("통과율", f"{frac:.1f} %",
          help="스윕 격자 중 두 기준(상대+절대) 동시 만족 비율")
c2.metric("기준 MW", f"{r0['MW']:.2f} V", f"loss {r0['MW_loss']:.1f} %")
if target <= mw_max:
    c3.metric("목표 달성", "가능 ✅", f"최대 MW {mw_max:.2f} V")
else:
    c3.metric("목표 달성", "불가 ⚠️", f"최대 MW {mw_max:.2f} V",
              delta_color="inverse")

if frac == 0.0 or target > mw_max:
    mt = min_tfe_for_target(pr, dpsi, ec, na, target)
    if mt is not None:
        st.warning(
            f"현재 스택(t_FE={t_fe:.1f} nm, P_r={pr:.0f} μC/cm²)으로는 목표 MW={target:.2f} V "
            f"달성이 어려움 (통과율 {frac:.1f}%). → t_FE ≥ 약 {mt:.1f} nm 로 올리거나 "
            f"P_r을 높여야 함."
        )
    else:
        st.error(
            f"t_FE=20 nm 로도 목표 MW={target:.2f} V 미달. "
            f"→ P_r을 높이거나 목표(레벨수/ΔV_level)를 낮춰야 함."
        )

# ─────────────────────────── 탭 ───────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 설계범위 지도 & 처방", "📈 1D 민감도", "📖 방법 / 가정"])

with tab1:
    trows = table_bounds(t_fe, pr, dpsi, ec, na, loss_max, target)   # 처방(지도 빨간숫자 + 아래 표 공용)
    presc = [{"t_IL": r["t_IL"], "dit_max": r["dit_max"]} for r in trows]
    # ── 헤드라인: Design window 지도 ──
    #   use_container_width=True: 컨테이너 폭에 맞춰 전체를 스케일(잘림·2배 확대 방지, 크기 일정)
    st.subheader("Design window 지도")
    st.pyplot(plot_tool.plot_designmap(m, target, loss_max, frac, presc, r0["MW_ref"]),
              use_container_width=True)
    # ── 범례(MW_loss·MW·allowable D_it)를 그림 아래에 표기 (동적) ──
    mw_rel = r0["MW_ref"] * (1.0 - loss_max / 100.0)
    loss_abs = (r0["MW_ref"] - target) / r0["MW_ref"] * 100.0
    st.markdown(
        f"<div style='font-size:0.9em;line-height:1.8'>"
        f"<span style='color:#d1341f;font-weight:700'>━━ 빨강 실선</span> = MW_loss {loss_max:.0f}% (상대) ≡ MW {mw_rel:.1f} V"
        f" <br>"
        f"<span style='color:#111;font-weight:700'>╌╌ 검정 파선</span> = MW {target:.1f} V (절대) ≡ {loss_abs:.0f}%"
        f" <br> "
        f"<span style='font-weight:700'>○ 처방점</span> = 허용 D_it 상한 [×10¹² cm⁻²eV⁻¹]"
        f" <br> 색 = MW_loss (5%마다, ≥50% 노랑)"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    colA, colB = st.columns([1.15, 1])
    with colA:
        st.subheader("허용 D_it 상한 처방 곡선")
        dit_nom = [r["dit_max"] for r in rows]
        dit_lo = dit_hi = None
        if show_band:
            m_lo = compute_map(t_fe, pr, 1.0, ec, na, n_dit, n_til)  # Δψ_w=1.0 → 상한↑
            m_hi = compute_map(t_fe, pr, 2.0, ec, na, n_dit, n_til)  # Δψ_w=2.0 → 상한↓
            dit_lo = [r["dit_max"] for r in dit_upper_bounds(m_lo, loss_max, target)]
            dit_hi = [r["dit_max"] for r in dit_upper_bounds(m_hi, loss_max, target)]
        st.pyplot(plot_tool.plot_prescription(m["yvals"], dit_nom, dit_lo, dit_hi,
                                              loss_max, target, show_band))
        st.caption("각 t_IL에서 두 기준을 만족하는 D_it 최대 허용값. 밴드 = Δψ_w 1.0–2.0 V 불확실성.")
    with colB:
        st.subheader("처방 표 (대표 t_IL)")

        def _fmt(x):
            if x is None:
                return "— (전 구간 미달)"
            if x >= 9.9e12:
                return "≥ 1×10¹³ (제약 없음)"
            return f"{x:.2e} cm⁻²eV⁻¹"

        tbl = {
            "t_IL [nm]": [f"{r['t_IL']:.1f}" for r in trows],
            "허용 D_it 상한 (binding)": [_fmt(r["dit_max"]) for r in trows],
            "상대기준 경계": [_fmt(r["dit_rel"]) for r in trows],
            "절대기준 경계": [_fmt(r["dit_abs"]) for r in trows],
        }
        st.table(tbl)
        st.caption(f"현재 스펙: 목표 MW ≥ {target:.2f} V, 손실 ≤ {loss_max:.0f} %, "
                   f"Δψ_w = {dpsi:.1f} V, t_FE = {t_fe:.1f} nm, P_r = {pr:.0f} μC/cm².")

with tab2:
    st.subheader("1D 민감도 (현재 스펙 기준)")
    st.markdown(
        "한 번에 한 변수만 바꿨을 때 memory window(MW)가 어떻게 변하는지 봄 "
        "(나머지 값은 현재 사이드바 설정으로 고정).\n\n"
        "- 왼쪽 · MW vs t_FE — 강유전체가 두꺼울수록 MW가 거의 선형으로 증가함. "
        "빨간 점선 = 목표 MW, 회색 점선 = 현재 t_FE. 파란 곡선이 빨간선 위로 올라가는 t_FE부터 목표 도달.\n"
        "- 오른쪽 · MW vs t_IL — 계면층 두께 단독으로는 MW가 거의 안 변함 "
        "(세로축 눈금 폭이 매우 좁은 것에 주목).\n\n"
        "왜 중요한가 — t_FE·P_r는 MW를 직접 키우는 설계 노브, t_IL은 단독 효과가 거의 없는 축임. "
        "단 t_IL은 트랩 손실(∝ D_it·t_IL)을 통해 D_it와 얽힐 때만 MW를 깎음 → 그래서 허용범위를 "
        "D_it×t_IL 2D 지도로 보는 것이 근거가 됨. 목표 미달이면 왼쪽 그래프로 "
        "t_FE를 얼마나 올려야 하는지를 바로 읽을 수 있음."
    )
    base = _base(t_fe, pr, dpsi, ec, na)
    s_tfe = sweep_1d("t_FE", list(np.linspace(5, 20, 31)), base)
    s_til = sweep_1d("t_IL", list(np.linspace(0.5, 2.0, 25)), base)
    st.pyplot(plot_tool.plot_sensitivity(
        s_tfe["values"], s_tfe["MW"], s_til["values"], s_til["MW"], target, t_fe))
    st.caption(
        f"현재 설정(P_r {pr:.0f} μC/cm², Δψ_w {dpsi:.1f} V) 기준으로 "
        f"t_FE 5→20 nm면 MW가 약 {s_tfe['MW'][0]:.2f}→{s_tfe['MW'][-1]:.2f} V로 변함. "
        f"t_IL 0.5→2.0 nm 변화는 {100*(s_til['MW'][-1]-s_til['MW'][0])/s_til['MW'][0]:+.1f}% 로 미미함."
    )

with tab3:
    st.markdown(
        """
### 방법 (요약)
게이트 적층 Gate / Ferroelectric(HfO₂·HZO, t_FE) / SiO₂ IL(t_IL) / p-Si 를 1차원
직렬 커패시터로 모델링하고, Miller tanh 분극으로 program/erase 두 상태의 문턱전압을
self-consistent하게 계산하여 MW = V_th(erase) − V_th(program), 정규화 손실률
MW_loss = (MW_ref − MW)/MW_ref × 100, (MW_ref = 2·E_c·t_FE, 정규화 기준선)을 구함.

목표 MW(절대)와 손실 허용치(상대)를 동시에 만족하는 D_it–t_IL 영역을 허용 설계범위로,
각 t_IL의 D_it 최대 허용값을 처방으로 추출함. 목표 MW = (N−1)·ΔV_level 로
binary/MLC를 포괄함.

### 가정 · 한계 (정직 고지)
- 실측 아님 — 전부 compact model 계산. 물리·MW 모델은 입력 물리(선행연구) 이며,
  본 도구의 기여는 *임의 목표에 대한 허용범위 일반화·처방·도구화*(공학 도구/방법론)에 한정.
- Δψ_w = 1.5 V 는 잠정값 (실험 앵커 미확보) → 1.0–2.0 V 밴드로 보고.
- 고정전하 Q_f 는 rigid-charge 가정 → MW 불변(모델 가정). depolarization 포함 시 열화 가능.
- 계면 트랩은 MW 손실항으로만 단순화. MLC 판정은 "필요 윈도우(MW ≥ (N−1)·ΔV)" 만
  반영 — 변동성·read-margin·retention은 미포함.
- 1차원·정적(dc) 근사. 온도·내구성·3D 미포함.

### 참고문헌 (입력 물리)
[1] Miller & McWhorter, *J. Appl. Phys.* 72(12), 5999 (1992) — tanh 분극 모델
[2] Toprasertpong et al., *IEEE TED* 69(12), 7113 (2022) — 해석적 MW
[3] Zagni et al., *Appl. Phys. Lett.* 117(15), 152901 (2020) — MW>0 경계식(동일 계열)
[4] Zhao et al., *IEEE TED* 69, 1561 (2022) — 트랩→MW 손실
        """
    )

st.divider()
st.caption("© 2026 고승민 · 나영은 (한양대 ERICA 전자공학부) · 2026 한국전기전자학회 하계학술대회 "
           "· compact-model 시뮬레이션 도구")
