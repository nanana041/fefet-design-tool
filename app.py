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
    "Compact-model 시뮬레이션 도구임 (소자 제작/측정이 아님).  \n"
    "목표 Memory Window(MW)를 정하면, 그 목표를 만족하는 허용 설계범위(D_it–t_IL)와 D_it 상한 처방을 계산함."
)

# ─────────────────── 본문 컨트롤 (자주 바꾸는 값 4개) ───────────────────
#   ★모바일에서 사이드바는 서랍이라 값 하나 바꿀 때마다 열었다 닫아야 한다.
#   가장 자주 만지는 것 — 설계 노브 2개(t_FE·P_r) + 스펙 2개(손실 허용치·목표 MW) —
#   만 본문 맨 위로 꺼내고, 어쩌다 한 번 바꾸는 것(고급·불확실성·격자)은 사이드바에
#   남긴다. 같은 위젯을 양쪽에 두면 상태가 둘로 갈라지므로 반드시 한쪽에만 둔다.
with st.container(border=True):
    k1, k2 = st.columns(2)
    t_fe = k1.slider(
        "⚙️ t_FE — 강유전체 두께 [nm]", 5.0, 20.0, 10.0, 0.5,
        help="강유전체(HfO₂/HZO) 층의 두께임. 두꺼울수록 메모리 윈도우(MW=두 저장 상태의 "
             "문턱전압 차)가 거의 선형으로 커짐. 목표 MW가 큰 다치 저장에선 이 값을 키워야 함. "
             "(기준 10 nm)")
    pr = k2.slider(
        "⚙️ P_r — 잔류분극 [μC/cm²]", 5.0, 25.0, 15.0, 1.0,
        help="잔류분극 — 전압을 끊어도 강유전체에 남아 있는 분극량임. 클수록 두 상태의 "
             "MW가 커지지만 점점 포화함. 포화분극 P_s는 자동으로 P_r의 1.3배로 잡히고, "
             "P_r=15에선 논문 기준값 P_s=20으로 고정됨. (기준 15)")

# ─────────────────────────── 사이드바 ───────────────────────────
sb = st.sidebar
sb.caption("설계 노브(t_FE · P_r)는 **본문 맨 위**, 두 기준(손실 허용치 · 목표 MW)은 "
           "**지도 바로 아래**에 있음. 여기는 저장 방식과 어쩌다 한 번 바꾸는 값만 둠.")

sb.markdown("### 🎯 스펙 / 목표")
tgt_mode = sb.segmented_control(
    "목표 설정 방식", ["직접 입력", "MLC로 계산"], default="직접 입력",
    help="직접 입력 = 목표 MW를 슬라이더로 바로 지정(0.8–2.0 V). "
         "MLC로 계산 = 저장 레벨 수와 레벨당 마진에서 유도 → 목표 MW = (N−1)×ΔV_level. "
         "두 모드 모두 같은 '목표 MW' 하나로 합류하므로 계산 경로는 동일함.")
tgt_mode = tgt_mode or "직접 입력"
if tgt_mode == "MLC로 계산":
    n_lv = sb.segmented_control(
        "레벨 수 N", [2, 3, 4], default=3, format_func=lambda n: f"{n}단계",
        help="한 셀에 저장하는 단계 수임. N단계를 구분하려면 창(MW)이 (N−1)×레벨마진 "
             "이상 필요 → 목표 MW = (N−1)·ΔV_level. (예: 3단계·마진 1.0V → 목표 2.0 V)")
    n_lv = n_lv or 3
    sb.caption("레벨당 마진 ΔV_level 은 **지도 아래**에 있음")
else:
    n_lv = 2
    sb.caption("목표 MW 슬라이더는 **지도 아래**에 있음")

# ★두 기준 위젯은 지도 아래(범례와 구속 안내 사이)에 그리지만, 그 값은 지도·통과율을
#   계산할 때 이미 필요하다. Streamlit은 위에서 아래로 실행되므로 위젯을 먼저 만들 수
#   없다 → 값을 session_state 에서 미리 읽고, 위젯은 나중에 같은 key 로 그린다.
#   (첫 실행은 setdefault 값, 이후 실행은 사용자가 움직인 값이 스크립트 시작 시점에
#    이미 session_state 에 들어와 있으므로 순서가 어긋나지 않는다.)
#   ★모드를 바꾸면 반대쪽 슬라이더가 그려지지 않는데, 이때 Streamlit 은 그 key 를
#     **지우는 게 아니라 기본값으로 되돌려 놓는다**(실측). 그래서 "key 가 있으면
#     최신값"이라고 믿고 보관값을 갱신하면 되돌려진 기본값이 좋은 값을 덮어쓴다.
#     → 이번 실행에서 **실제로 그려지는** 위젯만 보관값을 갱신하고, 안 그려지는 쪽과
#       방금 모드가 바뀌어 되살아나는 쪽은 보관값으로 덮어쓴다.
_prev_mode = st.session_state.get("_prev_mode")
_switched = _prev_mode != tgt_mode
st.session_state["_prev_mode"] = tgt_mode
_mlc = tgt_mode == "MLC로 계산"
for _wk, _sk, _default, _live in (("loss_max", "_keep_loss", 30, True),
                                  ("dv_level", "_keep_dv", 1.0, _mlc),
                                  ("tgt_direct", "_keep_tgt", 1.00, not _mlc)):
    st.session_state.setdefault(_sk, _default)
    if _live and not _switched:      # 계속 그려지던 위젯 → 사용자가 움직인 값이 최신
        st.session_state[_sk] = st.session_state.get(_wk, st.session_state[_sk])
    else:                            # 안 그려졌거나 방금 되살아남 → 보관값이 최신
        st.session_state[_wk] = st.session_state[_sk]
loss_max = float(st.session_state["loss_max"])
if tgt_mode == "MLC로 계산":
    dv = float(st.session_state["dv_level"])
    # ★수치 계약(tests/test_tool_contract.py): target_MW = (N−1)·ΔV_level.
    #   직접 입력 모드를 더해도 두 모드가 이 한 변수로 합류하므로 관계는 그대로 성립한다.
    target = round((n_lv - 1) * dv, 3)
else:
    dv = None
    target = float(st.session_state["tgt_direct"])
# ── 처음 쓰는 사람을 위한 값 설명 (펼쳐보기) ──
with sb.expander("ℹ️ 각 값이 무슨 뜻인가요?  (처음이면 눌러서 펼쳐보기)"):
    st.markdown(
        "이 도구는 \"이 목표를 만족하려면 계면(D_it·t_IL)을 얼마나 좋게 만들어야 하나\" 를 계산함.\n\n"
        "- ⚙️ 설계 노브 (본문) — 우리가 *만드는* 소자의 구조(강유전체 두께 등). 설계자가 직접 정하는 값.\n"
        "- 🔴⚫ 스펙/목표 (본문) — 우리가 *요구하는* 성능. 목표 MW를 정하면 도구가 허용 범위를 답해줌. "
        "지도의 빨간 실선(🔴 상대 기준)·검은 파선(⚫ 절대 기준)에 각각 대응함.\n"
        "- 📉 불확실성 (Δψ_w) — 실험으로 아직 정확한 값을 못 정한 물리값. 하나로 못 박지 않고, "
        "가능한 범위(1.0–2.0 V)를 그래프에 띠(밴드) 로 함께 그려 불확실성의 폭을 보여줌.\n"
        "- 🔲 격자 — 지도 계산 해상도. 매끄러움·속도만 바뀌고 결과 수치엔 거의 영향 없음.\n\n"
        "각 항목 이름 옆 ❓ 를 누르면 더 자세한 설명이 나옴."
    )

with sb.expander("⚙️ 고급 (기본 고정 권장)"):
    ec = st.number_input(
        "E_c [MV/cm]", 0.5, 3.0, 1.0, 0.1,
        help="항전계 — 분극을 뒤집는 데 필요한 전계. 정규화 기준선 MW_ref=2·E_c·t_FE 에 "
             "들어감. 보통 1.0으로 고정 권장.")
    na = st.number_input(
        "N_a [cm⁻³]", 1e16, 1e18, 1e17, format="%.0e",
        help="p-Si 기판의 도핑(억셉터) 농도. 표면 전위·공핍층 폭에 영향을 줌. 보통 1e17 고정.")

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


# ★배율 축의 ×1 = 표 1(논문) 기준값 — **고정**. 현재 슬라이더 값으로 정규화하면 축의
#   의미가 매번 바뀌어 논문 수치(MW 1.787 / 80.6 %)와 대조가 안 된다.
#   "지금 내가 어디 있나"는 곡선 위 현재 설계점 마커가 대신한다.
REF_TFE, REF_PR, REF_TIL = 10.0, 15.0, 1.0
DIT_CMP_OPTS = [1e11, 5e11, 1e12, 5e12, 1e13]   # 민감도 그림의 비교용 D_it
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _dit_label(d):
    """1e11 → '1×10¹¹' (mathtext 금지 규약이라 유니코드 위첨자로)."""
    e = int(np.floor(np.log10(d)))
    return f"{d / 10 ** e:.0f}×10{str(e).translate(_SUP)}"


@st.cache_data(show_spinner=False)
def compute_sensitivity(t_fe, pr, dpsi, ec, na, dit_cmp):
    """포스터 v4 그림3(정규화 민감도)과 같은 4곡선.

    ★P_r sweep은 P_s > P_r 유지를 위해 sweeps.PS_RATIO(=1.3)로 P_s도 같이 올린다.
      그래서 배율 1.0에서 P_s = 1.3·P_r 이 되어, P_s를 20으로 고정한 기준점과 몇 mV
      어긋난다(P_r=15 기준 1.7907 vs 1.7870 = 3.7 mV). 눈으로는 안 보이지만 검산하면
      드러나므로 캡션에 규약을 적어 둔다.
    ★보라 곡선의 D_it는 포스터처럼 5e12 하드코딩이 아니라 인자로 받는다 — 앱에서는
      이 값을 올려 보며 t_IL 곡선이 갈라지는 것을 눈으로 볼 수 있어야 한다."""
    base = _base(t_fe, pr, dpsi, ec, na)
    tfe = np.linspace(5, 20, 31)
    prv = np.linspace(5, 25, 31)
    til = np.linspace(0.5, 2.0, 25)
    return dict(
        tfe=(tfe, sweep_1d("t_FE", list(tfe), base)["MW"]),
        pr=(prv, sweep_1d("Pr", list(prv), base)["MW"]),
        til=(til, sweep_1d("t_IL", list(til), base)["MW"]),
        til_hi=(til, sweep_1d("t_IL", list(til), {**base, "Dit": dit_cmp})["MW"]),
    )


@st.cache_data(show_spinner=False)
def compute_tab_map(t_fe, pr, dpsi, ec, na):
    """처방표·구속 판정용 고해상도 격자(대표 t_IL 4개)."""
    base = _base(t_fe, pr, dpsi, ec, na)
    return sweep_2d("Dit", np.logspace(11, 13, 200), "t_IL",
                    np.array([0.5, 1.0, 1.5, 2.0]), base)


@st.cache_data(show_spinner=False)
def table_bounds(t_fe, pr, dpsi, ec, na, loss_max, target):
    return dit_upper_bounds(compute_tab_map(t_fe, pr, dpsi, ec, na), loss_max, target)


def binding_of_row(r):
    """이 t_IL 행의 상한을 정한 쪽 — "relative"/"absolute", 도달 불가면 None.

    ★동률(dit_rel == dit_abs)은 상대로 센다. 계약 test_binding_criterion_switches_at_1_5V
      가 '1.4 V까지는 상대'로 고정한 지점이 정확히 rel == abs 인 자리이기 때문."""
    if r["dit_max"] is None:
        return None
    if r["dit_abs"] is None:
        return "relative"
    if r["dit_rel"] is not None and r["dit_rel"] <= r["dit_abs"] * (1 + 1e-9):
        return "relative"
    return "absolute"


def binding_of(rows):
    """지금 목표에서 어느 기준이 구속(binding)인지 — "relative"/"absolute"/"unreachable"."""
    live = [b for b in (binding_of_row(r) for r in rows) if b is not None]
    if not live:
        return "unreachable"
    return "relative" if live.count("relative") * 2 >= len(live) else "absolute"


@st.cache_data(show_spinner=False)
def binding_switch_target(t_fe, pr, dpsi, ec, na, loss_max):
    """구속 기준이 상대 → 절대로 넘어가는 목표 MW [V]. 전환이 없으면 None.
    (기본 스택·손실 30 %에서는 1.5 V — 이 앱의 하이라이트가 되는 지점)"""
    m = compute_tab_map(t_fe, pr, dpsi, ec, na)
    prev = None
    for t in np.arange(0.5, 3.001, 0.05):
        b = binding_of(dit_upper_bounds(m, loss_max, float(t)))
        if prev == "relative" and b != "relative":
            return round(float(t), 2)
        prev = b
    return None


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
    # bind: 이 점의 상한을 정한 기준 → 지도에서 원 테두리 색(빨강/검정)으로 쓰인다
    presc = [{"t_IL": r["t_IL"], "dit_max": r["dit_max"], "bind": binding_of_row(r)}
             for r in trows]
    # ── 헤드라인: Design window 지도 ──
    #   use_container_width=True: 컨테이너 폭에 맞춰 전체를 스케일(잘림·2배 확대 방지)
    #   ★bbox_inches=None 필수. st.pyplot 은 기본이 bbox_inches="tight" 라서 그려진
    #     내용의 경계에 맞춰 그림을 다시 잘라낸다. 그러면 컬러바 눈금·브래킷 글자처럼
    #     축 밖에 있는 것들의 위치가 값에 따라 달라질 때마다 출력 픽셀 크기가 바뀌고
    #     (실측: 1550×920 ↔ 1550×930), 화면에서 그림 크기가 들쭉날쭉해진다.
    #     이 그림은 add_axes 로 좌표를 이미 고정해 뒀으므로 잘라내면 안 된다.
    st.subheader("Design window 지도")
    st.pyplot(plot_tool.plot_designmap(m, target, loss_max, frac, presc, r0["MW_ref"]),
              use_container_width=True, bbox_inches=None)
    # ── 두 기준 슬라이더 — 지도 바로 아래 ──────────────────────────────────
    #   지도를 보고 → 곧바로 손잡이를 움직이고 → 그 아래 범례·구속 안내에서 결과를
    #   읽는 순서. 그림과 손잡이 사이에 설명글이 끼면 조작할 때마다 눈이 건너뛰어야 한다.
    #   값 자체는 스크립트 맨 위에서 session_state 로 이미 읽었다(위 주석 참조).
    cc1, cc2 = st.columns(2)
    cc1.markdown('<span class="anch-loss"></span>', unsafe_allow_html=True)
    cc1.slider(
        "🔴 손실 허용치 MW_loss_max [%]", min_value=10, max_value=50, step=5,
        key="loss_max",
        help="이상적 기준선(MW_ref) 대비 MW가 얼마나 줄어드는 것까지 허용할지(상대 기준). "
             "예: 30%면 'MW가 기준선의 70% 이상이면 합격'. "
             "★30 %는 업계 표준이 아니라 이 도구가 쓰는 예시 기준임 — 슬라이더로 바꿔 볼 것.")
    # ⚫ + anch-dv: 목표 MW는 **절대 기준**이므로 지도의 검정 파선과 같은 표기로 묶는다.
    #   (🔴/anch-loss = 상대 기준 = 지도의 빨간 실선. 이모지는 장식이 아니라 그림의
    #    어느 선에 대응하는지를 가리키는 표시다.)
    cc2.markdown('<span class="anch-dv"></span>', unsafe_allow_html=True)
    if tgt_mode == "MLC로 계산":
        cc2.slider(
            "⚫ 레벨당 마진 ΔV_level [V]", min_value=0.5, max_value=1.5, step=0.1,
            key="dv_level",
            help="인접한 두 저장 레벨 사이에 확보해야 할 최소 문턱전압 간격(읽기 여유)임. "
                 "클수록 안전하지만 요구되는 MW가 커짐.")
        cc2.caption(f"→ 목표 MW = (N−1)×ΔV_level = **{target:.2f} V** (절대 기준)")
    else:
        cc2.slider(
            "⚫ 목표 MW [V]", min_value=0.80, max_value=2.00, step=0.05,
            key="tgt_direct",
            help="이 소자가 만족해야 할 memory window(절대 기준). "
                 "★0.8–1.4 V 구간은 답이 전혀 안 변함 — 그 구간에선 손실 허용치(상대 기준)가 "
                 "먼저 걸리기 때문이며 고장이 아님. 1.5 V부터 절대 기준으로 바통이 넘어가고, "
                 "기본 스택(t_FE 10 nm)에서는 1.8 V 위가 도달 불가임.")

    # ── 범례(MW_loss·MW·allowable D_it)를 그림 아래에 표기 (동적) ──
    mw_rel = r0["MW_ref"] * (1.0 - loss_max / 100.0)
    loss_abs = (r0["MW_ref"] - target) / r0["MW_ref"] * 100.0
    # ★기준선이 지도 밖으로 나가면 왜 없는지 적는다.
    #   contour는 레벨이 데이터 범위 밖이면 아무것도 그리지 않는다. 슬라이더를 끝까지
    #   밀면 실제로 그렇게 된다 — 기본 스택의 MW_loss는 10.5~85.5 %라 손실 허용치
    #   10 %는 격자 아래로 빠지고, MW는 최대 1.79 V라 목표 2.0 V는 격자 위로 빠진다.
    #   선만 조용히 사라지면 고장으로 보이므로, 없는 이유와 그 뜻을 대신 표시한다.
    #   (아래로 벗어남 = 전 구간 미달 / 위로 벗어남 = 전 구간 통과 — 뜻이 정반대다.)
    l_lo, l_hi = float(m["MW_loss"].min()), float(m["MW_loss"].max())
    w_lo, w_hi = float(m["MW"].min()), float(m["MW"].max())
    GRAY = "<span style='color:#777'>"
    if loss_max < l_lo:
        rel_html = (f"{GRAY}빨강 실선 없음</span> — 이 격자의 MW_loss 최소가 "
                    f"{l_lo:.1f} %라 {loss_max:.0f} % 등고선이 지도 아래로 벗어남. "
                    f"<b>전 구간이 상대 기준 미달</b>임.")
    elif loss_max > l_hi:
        rel_html = (f"{GRAY}빨강 실선 없음</span> — MW_loss 최대가 {l_hi:.1f} %라 "
                    f"지도 전체가 상대 기준을 통과함(이 기준은 구속하지 않음).")
    else:
        rel_html = (f"<span style='color:#d1341f;font-weight:700'>━━ 빨강 실선</span>"
                    f" = MW_loss {loss_max:.0f}% (상대) ≡ MW {mw_rel:.1f} V")
    if target > w_hi:
        abs_html = (f"{GRAY}검정 파선 없음</span> — 이 스택의 MW 최대가 {w_hi:.2f} V라 "
                    f"목표 {target:.2f} V에 <b>전 구간 도달 불가</b>.")
    elif target < w_lo:
        abs_html = (f"{GRAY}검정 파선 없음</span> — MW 최소가 {w_lo:.2f} V라 지도 전체가 "
                    f"목표를 넘김(이 기준은 구속하지 않음).")
    else:
        abs_html = (f"<span style='color:#111;font-weight:700'>╌╌ 검정 파선</span>"
                    f" = MW {target:.1f} V (절대) ≡ {loss_abs:.0f}%")
    st.markdown(
        f"<div style='font-size:0.9em;line-height:1.8'>{rel_html}"
        f" <br>{abs_html}"
        f" <br> "
        f"<span style='font-weight:700'>○ 처방점</span> = 허용 D_it 상한 [×10¹² cm⁻²eV⁻¹]"
        f" <br> 색 = MW_loss (5%마다, ≥50% 노랑)"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── 어느 기준이 구속인가 (이 도구의 하이라이트) ────────────────────────
    #   기준이 둘이고 둘 다 만족해야 한다: 상대(MW_loss ≤ loss_max) · 절대(MW ≥ target).
    #   먼저 걸리는 쪽이 구속. 목표가 낮으면 상대가, 올리면 절대가 구속으로 바뀐다.
    #   ★"상대 기준이 항상 먼저 걸린다"는 t_FE·E_c를 고정한 이 지도에서만 참이다 —
    #     앱은 슬라이더로 그걸 실제로 뒤집어 보일 수 있다.
    #   위치: 지도 바로 아래(범례 다음). 두 기준선이 그림에 그려진 직후라야
    #   "그 둘 중 어느 쪽이 지금 걸리는가"가 그림을 보며 읽힌다.
    _bind = binding_of(trows)
    _sw = binding_switch_target(t_fe, pr, dpsi, ec, na, loss_max)
    _swtxt = f" 이 스택에서는 목표 <b>{_sw:.2f} V</b>부터 절대 기준으로 넘어감." if _sw else ""
    if _bind == "relative":
        _msg = (f"<b style='color:#d1341f'>지금은 빨강(상대) 기준이 구속</b> — 손실 허용치"
                f"(MW_loss ≤ {loss_max:.0f} %)가 먼저 걸림. "
                f"이 구간에서는 <b>목표 MW를 올려도 허용 D_it 상한이 변하지 않음</b>"
                f"(고장 아님)." + _swtxt)
        _fg, _bg = "#d1341f", "#fdf0ed"
    elif _bind == "absolute":
        _msg = (f"<b>지금은 검정(절대) 기준이 구속</b> — 목표 MW ≥ {target:.2f} V가 먼저 "
                f"걸림. 여기서부터는 목표를 올릴수록 허용 D_it 상한이 급격히 좁아짐."
                + _swtxt)
        _fg, _bg = "#111111", "#f1f2f4"
    else:
        _msg = (f"<b style='color:#a06a00'>도달 불가</b> — 이 스택은 D_it를 아무리 낮춰도 "
                f"목표 MW {target:.2f} V에 못 미침. 허용 상한이 존재하지 않으므로 아래 "
                f"처방표에 '—'로 표시됨." + _swtxt)
        _fg, _bg = "#a06a00", "#fdf7e8"
    st.markdown(
        f"<div style='background:{_bg};border-left:5px solid {_fg};padding:9px 14px;"
        f"border-radius:4px;font-size:0.93em;line-height:1.65;margin:8px 0 4px'>{_msg}</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    colA, colB = st.columns([1, 1.35])
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
                                              loss_max, target, show_band),
                  bbox_inches=None)   # ↑와 같은 이유 — 크기 일정하게
        st.caption("각 t_IL에서 두 기준을 만족하는 D_it 최대 허용값. 밴드 = Δψ_w 1.0–2.0 V 불확실성.")
    with colB:
        st.subheader("처방 표 (대표 t_IL)")

        def _fmt(x):
            if x is None:
                return "—"
            if x >= 9.9e12:
                return "≥ 1×10¹³"
            return f"{x:.2e}"

        def _neff(x):
            """유효 트랩전하 N_eff = D_it·Δψ_w/2 [cm⁻²].
            ★트랩 항은 이 곱으로만 모델에 들어가므로(축퇴) Δψ_w를 바꾸면 허용 D_it
              상한은 ∝1/Δψ_w 로 움직이지만 이 값은 그대로다 — 그게 '둘을 따로 정할 수
              없다'는 뜻이고, 실제 앵커 대상은 D_it가 아니라 N_eff다.
            ★단 상한이 스윕 격자 끝(1e13)에 걸린 칸은 '≥'로 표시해야 한다. 그 칸의
              진짜 상한은 격자 밖이라 N_eff도 하한값일 뿐인데, 그냥 숫자로 찍으면
              Δψ_w를 바꿨을 때 그 칸만 값이 달라져서 축퇴가 깨진 것처럼 보인다."""
            if x is None:
                return "—"
            return ("≥ " if x >= 9.9e12 else "") + f"{x * dpsi / 2:.2e}"

        tbl = {
            "t_IL [nm]": [f"{r['t_IL']:.1f}" for r in trows],
            "허용 D_it 상한 [cm⁻²eV⁻¹]": [_fmt(r["dit_max"]) for r in trows],
            "→ N_eff 상한 [cm⁻²]": [_neff(r["dit_max"]) for r in trows],
            "상대기준 경계": [_fmt(r["dit_rel"]) for r in trows],
            "절대기준 경계": [_fmt(r["dit_abs"]) for r in trows],
        }
        st.table(tbl)
        st.caption(f"현재 스펙: 목표 MW ≥ {target:.2f} V, 손실 ≤ {loss_max:.0f} %, "
                   f"Δψ_w = {dpsi:.1f} V, t_FE = {t_fe:.1f} nm, P_r = {pr:.0f} μC/cm². "
                   f"'—' = 그 t_IL에서는 목표 도달 불가(상한 없음).")
        st.info(
            "**N_eff = D_it · Δψ_w / 2** — 계면 트랩은 이 **곱으로만** MW를 깎음. "
            "왼쪽 사이드바의 Δψ_w를 움직여 보면 **허용 D_it 상한은 바뀌는데 N_eff 상한은 "
            "그대로**임. 두 값을 따로 정하는 것이 원리적으로 불가능하다는 뜻이고(축퇴), "
            "실험으로 앵커해야 할 대상도 D_it가 아니라 N_eff임. "
            "(‘≥’가 붙은 칸은 상한이 스윕 격자 끝 1×10¹³을 넘어간 경우라 하한값만 "
            "표시된 것 — 그 칸만은 Δψ_w에 따라 표시값이 달라짐.)",
            icon="🔗",
        )

with tab2:
    st.subheader("1D 민감도 (현재 스펙 기준)")
    st.markdown(
        "한 번에 한 변수만 바꿨을 때 memory window(MW)가 어떻게 변하는지 봄. "
        "가로축은 **각 변수를 논문 표 1의 기준값으로 나눈 배율**임 "
        "(×1 = t_FE 10 nm · P_r 15 μC/cm² · t_IL 1.0 nm). 축이 하나뿐이라 곡선끼리 "
        "기울기를 직접 비교할 수 있고, 기준이 고정이라 논문 수치와 언제든 대조됨.\n\n"
        "- **흰 원** = 지금 사이드바 설정이 각 곡선 위에서 어디인지. 슬라이더를 움직이면 "
        "곡선을 따라 미끄러짐.\n"
        "- **검은 파선** = 현재 목표 MW. 곡선이 이 선 위로 올라가는 지점부터 목표 달성임 "
        "(옆 탭 design map의 절대 기준선과 같은 표기).\n\n"
        "- **t_FE** — 강유전체가 두꺼울수록 MW가 거의 선형으로 증가함. 목표 미달이면 이 "
        "곡선으로 t_FE를 얼마나 올려야 하는지 바로 읽힘.\n"
        "- **P_r** — 잔류분극이 클수록 MW가 커지지만 점점 포화함.\n"
        "- **t_IL (빨강)** — 계면층 두께 **단독으로는 MW가 거의 안 변함**.\n"
        "- **t_IL (보라 파선)** — ★같은 t_IL인데 계면 트랩이 많으면 **급격히 깎임**. "
        "아래 슬라이더로 D_it를 올려 보면 두 t_IL 곡선이 벌어지는 게 보임 — "
        "t_IL은 혼자서는 무해하고 **D_it와 얽힐 때만** MW를 문다는 뜻.\n\n"
        "왜 중요한가 — 빨강만 보면 \"t_IL은 중요하지 않다\"로 읽히는데, 옆 탭의 design map은 "
        "t_IL이 두 축 중 하나임. 보라 파선이 그 모순을 풀어 주고, **허용범위를 D_it×t_IL "
        "2차원으로 봐야 하는 근거**가 됨."
    )
    dit_cmp = st.select_slider(
        "보라 파선의 D_it — 올려 보면 t_IL 곡선이 갈라짐", options=DIT_CMP_OPTS,
        value=5e12, format_func=_dit_label,
        help="빨간 t_IL 곡선은 기준 D_it(1×10¹¹)에서의 것임. 이 값을 올리면 같은 t_IL "
             "구간인데도 MW가 급격히 깎이는 보라 곡선이 갈라져 나옴 → '계면층 두께는 "
             "혼자서는 무해하고 트랩과 얽힐 때만 문다'가 눈으로 보임. 정적인 포스터 "
             "그림으로는 못 하는 부분임.")
    s = compute_sensitivity(t_fe, pr, dpsi, ec, na, float(dit_cmp))
    _dl = _dit_label(dit_cmp)
    curves = [
        dict(name="t_FE", x=s["tfe"][0] / REF_TFE, mw=s["tfe"][1],
             color=plot_tool.BLUE, ls="-", lo="5", hi="20 nm",
             cur=(t_fe / REF_TFE, float(np.interp(t_fe, s["tfe"][0], s["tfe"][1])))),
        dict(name="P_r", x=s["pr"][0] / REF_PR, mw=s["pr"][1],
             color=plot_tool.TEAL, ls="-", lo="5", hi="25 μC/cm²",
             cur=(pr / REF_PR, float(np.interp(pr, s["pr"][0], s["pr"][1])))),
        # t_IL은 사이드바 슬라이더가 아니라 지도의 축이라 1D 기준값 1.0 nm에 고정 → 항상 ×1
        dict(name="t_IL", x=s["til"][0] / REF_TIL, mw=s["til"][1],
             color=plot_tool.RED, ls="-", lo="0.5", hi="2 nm",
             cur=(1.0, float(np.interp(REF_TIL, s["til"][0], s["til"][1])))),
        # 두 줄로 쪼갬: 한 줄이면 오른쪽 여백을 넘는다 (세로 분리는 plot_tool이 처리)
        dict(name=f"t_IL\n@ D_it {_dl}", x=s["til_hi"][0] / REF_TIL, mw=s["til_hi"][1],
             color=plot_tool.PLUM, ls=(0, (5, 3)), lo="0.5", hi="", cur=None),
    ]
    st.pyplot(plot_tool.plot_sensitivity(curves, target),
              use_container_width=False, bbox_inches=None)   # ↑와 같은 이유

    def _pct(y):
        return 100.0 * (y[-1] - y[0]) / y[0]

    st.caption(
        f"현재 설정(t_FE {t_fe:.1f} nm, P_r {pr:.0f} μC/cm², Δψ_w {dpsi:.1f} V) 기준. "
        f"t_FE 5→20 nm면 MW {s['tfe'][1][0]:.2f}→{s['tfe'][1][-1]:.2f} V. "
        f"t_IL 0.5→2.0 nm는 기준 D_it(1×10¹¹)에서 {_pct(s['til'][1]):+.1f} % 로 미미하지만, "
        f"D_it = {_dl}에서는 {_pct(s['til_hi'][1]):+.1f} % 로 급격함. "
        "※ P_r 곡선은 P_s > P_r 유지를 위해 P_s = 1.3·P_r 로 함께 올린 결과라, 배율 1.0에서 "
        "기준점(P_s 고정)과 몇 mV 어긋나는 것이 정상임."
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
