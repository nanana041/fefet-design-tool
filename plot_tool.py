# -*- coding: utf-8 -*-
"""
Streamlit 앱용 그림 생성.
- design map (D_it–t_IL, MW_loss 색 + 합격 영역 흰 해치 + 상대/절대 기준선)
- 허용 D_it 상한 처방 곡선 (+ Δψ_w 밴드)
- 1D 민감도 (MW vs t_FE / t_IL)

★중요: 라벨에 matplotlib **mathtext($...$)를 절대 쓰지 않는다.**
  streamlit은 스크립트를 스레드에서 반복 실행하는데, mathtext 파서(pyparsing 전역
  packrat 캐시)가 스레드-반복에 취약해 "ParseException (at char 0)" 를 던진다.
  → 아래첨자는 밑줄(D_it), 위첨자·부등호는 유니코드(cm⁻²eV⁻¹, ≤, ≥) 로만 표기.
  로그축 눈금도 mathtext 대신 유니코드 포매터(10ⁿ)를 쓴다.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.formatter.use_mathtext"] = False
import matplotlib.patheffects as pe
# ★그림은 pyplot 이 아니라 Figure 로 직접 만든다. plt.figure()/plt.subplots() 는 전역
#   레지스트리에 그림을 등록해 두는데, Streamlit 은 매 실행마다 그림을 새로 만들므로
#   슬라이더를 몇 번만 움직여도 그림이 쌓인다(실측: matplotlib 의 "More than 20 figures"
#   경고). 서버에서 렌더만 할 때는 Figure 를 직접 쓰는 것이 공식 권장 방식이고, 전역
#   상태가 없어져 스레드 반복 실행에서 생기는 문제(이 파일 위쪽 mathtext 주석 참조)의
#   여지도 줄어든다.
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter, LogLocator

RED = "#d1341f"
BLUE = "#1f6fd6"
TEAL = "#0f8a7e"    # 포스터 v4 그림3 하우스 색 (P_r 곡선)
PLUM = "#8e44ad"    # 〃 (D_it=5e12 에서의 t_IL — 상호작용 곡선)

# design map 축 여백 — 경계에 찍히는 처방 흰 원(ms=9pt)이 잘리지 않을 만큼만.
# x는 로그축이라 decade 단위, y는 nm 단위.
PAD_DEC, PAD_TIL = 0.05, 0.05

_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _log_unicode(x, pos=None):
    """로그축 눈금을 mathtext 없이 '10ⁿ' 유니코드로 (10의 거듭제곱만)."""
    if x <= 0:
        return ""
    e = int(round(np.log10(x)))
    return "10" + str(e).translate(_SUP)


def _log_unicode_full(x, pos=None):
    """좁은 로그축용: 'm×10ⁿ' 유니코드 (2×10¹² 등도 표기)."""
    if x <= 0:
        return ""
    e = int(np.floor(np.log10(x) + 1e-9))
    mant = x / 10.0 ** e
    mr = int(round(mant))
    if abs(mant - mr) > 0.06:
        return ""
    if mr == 1:
        return "10" + str(e).translate(_SUP)
    if mr == 10:
        return "10" + str(e + 1).translate(_SUP)
    return f"{mr}×10" + str(e).translate(_SUP)


def _set_log_unicode(ax, axis="x", dense=False):
    a = ax.xaxis if axis == "x" else ax.yaxis
    if dense:   # 1 decade 미만 범위: 소수 눈금(2·3·5×10ⁿ)도 라벨
        a.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5)))
        a.set_major_formatter(FuncFormatter(_log_unicode_full))
    else:       # 넓은 범위: 10의 거듭제곱만
        a.set_major_formatter(FuncFormatter(_log_unicode))
    a.set_minor_formatter(NullFormatter())


def _set_pe(cs, lw, fg):
    eff = [pe.withStroke(linewidth=lw, foreground=fg)]
    try:
        cs.set_path_effects(eff)
    except Exception:
        for c in getattr(cs, "collections", []):
            c.set_path_effects(eff)


def plot_designmap(m, target_mw, mw_loss_max, frac, presc, mw_ref):
    """MW_loss design map — 5%마다 이산 색띠, ≥50% 노랑 통일(extend max),
    사선 없음, 빨간 상대선+흰 원마커+동적 빨간 박스숫자(허용 D_it), 검정 파선 절대선.
    축·그림 크기 고정, colorbar 빨간 눈금 + design window 브래킷.
    presc: [{t_IL, dit_max}, ...] (t_IL 0.5/1.0/1.5/2.0),  mw_ref: 정규화 기준선(V)."""
    LEVELS = list(range(10, 51, 5))      # 10,15,...,50
    Z = np.clip(m["MW_loss"], 10.0, None)  # <10은 최하위 색으로(흰 구멍 방지 → 바닥 10 고정)

    # 고정 레이아웃(라벨 길이에 따라 그림이 흔들리지 않도록 add_axes 로 위치 고정)
    fig = Figure(figsize=(8.0, 4.9))
    ax = fig.add_axes([0.095, 0.155, 0.66, 0.80])
    cax = fig.add_axes([0.875, 0.155, 0.028, 0.80])

    cf = ax.contourf(m["X"], m["Y"], Z, levels=LEVELS, cmap="viridis", extend="max")
    cb = fig.colorbar(cf, cax=cax, ticks=LEVELS)
    cb.set_label("MW_loss  [%]", fontsize=12.5, fontweight="bold")
    cb.ax.tick_params(labelsize=10)

    # colorbar 위 빨간 눈금(=상대 기준 loss_max) + design window 브래킷(10~loss_max)
    yt = cax.get_yaxis_transform()   # x: cax 축분율, y: 데이터값(%)
    cax.plot([0, 1], [mw_loss_max, mw_loss_max], transform=yt, color=RED, lw=2.5,
             clip_on=False, zorder=5)
    _loss_abs = (mw_ref - target_mw) / mw_ref * 100.0   # 절대선 등가 손실%
    if 10.0 <= _loss_abs <= 50.0:                        # colorbar에 검정 파선 눈금
        cax.plot([0, 1], [_loss_abs, _loss_abs], transform=yt, color="black", ls="--",
                 lw=1.8, clip_on=False, zorder=5)
    bx = -1.5
    for seg in ([bx, bx], [10, mw_loss_max]), ([bx, bx + 0.4], [10, 10]), ([bx, bx + 0.4], [mw_loss_max, mw_loss_max]):
        cax.plot(seg[0], seg[1], transform=yt, color="#555", lw=1.3, clip_on=False, zorder=4)
    cax.text(bx - 0.7, (10 + mw_loss_max) / 2, "design window", transform=yt, rotation=90,
             ha="center", va="center", fontsize=9, fontweight="bold", color="#222", clip_on=False)
    cax.text(bx + 1.05, (10 + mw_loss_max) / 2, f"{frac:.1f} % of grid", transform=yt, rotation=90,
             ha="center", va="center", fontsize=7.5, color="#888", clip_on=False)

    # 상대 기준선(빨강) at MW_loss = loss_max
    if np.nanmin(m["MW_loss"]) <= mw_loss_max <= np.nanmax(m["MW_loss"]):
        ax.contour(m["X"], m["Y"], m["MW_loss"], levels=[mw_loss_max], colors=RED,
                   linewidths=3.5, zorder=3)
    # 절대 기준선(검정 파선) at MW = target  (배경 밝기 무관하게 흰 후광)
    if m["MW"].min() <= target_mw <= m["MW"].max():
        c_abs = ax.contour(m["X"], m["Y"], m["MW"], levels=[target_mw], colors="black",
                           linestyles="--", linewidths=2.2, zorder=3)
        _set_pe(c_abs, 3.5, "white")

    # 동적 처방: 흰 원마커 + 박스 숫자(×10¹²) — 그림 밖으로 안 나가게 위치 보정
    #   ★색 = 그 점의 상한을 정한 기준. 상대면 빨강(빨간 실선 위), 절대면 검정(검은 파선
    #     위). 어느 쪽도 스윕 범위 안에서 안 걸린 행("none")은 **회색 + "≥10"** 으로
    #     쓴다 — 그 값은 실제 상한이 아니라 격자 끝(D_it 1e13)이라 하한만 아는 상태다.
    #     색으로 기준을 말하는 그림에서 이걸 빨강으로 칠하면 걸리지도 않은 기준이 상한을
    #     정한 것처럼 보인다.
    for p in presc:
        d, til, bind = p.get("dit_max"), p.get("t_IL"), p.get("bind")
        if d is None or d > 1.02e13:
            continue
        bc = {"absolute": "black", "relative": RED}.get(bind, "#6b6b6b")
        ax.plot([d], [til], "o", mfc="white", mec=bc, mew=2.2, ms=9, zorder=6)
        logpos = (np.log10(d) - 11.0) / 2.0                       # 0(1e11)~1(1e13)
        xoff, ha = (16, "left") if logpos < 0.24 else (-16, "right")  # 왼쪽 끝이면 박스 오른쪽
        dy = -13 if til >= 1.9 else (13 if til <= 0.6 else 0)     # 위/아래 끝이면 안쪽으로
        ax.annotate("≥10" if bind == "none" else f"{d/1e12:.1f}",
                    xy=(d, til), xytext=(xoff, dy),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=12, fontweight="bold", color="white",
                    # 상자 배경도 구속 기준의 색으로 — 숫자와 그 숫자를 만든 선이 같은
                    # 색으로 묶인다. 흰 테두리는 어두운 색띠(보라·남색) 위에서 상자
                    # 경계가 사라지지 않게 하려고 남긴다.
                    bbox=dict(boxstyle="round,pad=0.28", fc=bc, ec="white", lw=1.0),
                    zorder=7)

    ax.set_xscale("log")
    _set_log_unicode(ax, "x")
    # ★축을 스윕 범위(D_it 1e11–1e13 · t_IL 0.5–2.0)보다 살짝 넓게 잡는다.
    #   처방 흰 원이 경계값(t_IL 0.5·2.0, D_it 상한이 1e13 근처)에 찍히면 프레임에
    #   반쯤 잘려 나가기 때문 — 확장 폭은 마커 반지름(ms=9pt)이 들어갈 만큼만이다.
    #   (여백은 contourf 밖이라 흰색으로 남는다. 2026-08-04 '디자인맵 수정안' 반영.)
    ax.set_xlim(10 ** (11 - PAD_DEC), 10 ** (13 + PAD_DEC))
    ax.set_ylim(0.5 - PAD_TIL, 2.0 + PAD_TIL)
    ax.set_yticks(np.arange(0.6, 2.01, 0.2))   # 여백 때문에 자동 눈금이 흔들리지 않도록
    ax.set_xlabel("Interface trap density  D_it  [cm⁻²eV⁻¹]", fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Interfacial layer thickness  t_IL  [nm]", fontsize=12.5, fontweight="bold")
    ax.tick_params(labelsize=11)
    return fig     # 범례(MW_loss·MW·allowable D_it)는 앱에서 그림 아래에 표시


def _clean(vals):
    return np.array([np.nan if v is None else float(v) for v in vals], float)


def plot_prescription(til, dit_nom, dit_lo, dit_hi, mw_loss_max, target_mw, show_band):
    """각 t_IL의 허용 D_it 상한 곡선 (+ Δψ_w 밴드). dit_lo=Δψ_w 1.0(높음), dit_hi=Δψ_w 2.0(낮음)."""
    fig = Figure(figsize=(6.2, 4.5))
    ax = fig.subplots()
    til = np.asarray(til, float)
    y = _clean(dit_nom)
    if show_band and dit_lo is not None and dit_hi is not None:
        ylo = _clean(dit_lo)   # Δψ_w = 1.0 V → 상한 높음
        yhi = _clean(dit_hi)   # Δψ_w = 2.0 V → 상한 낮음
        ax.fill_between(til, yhi, ylo, color=BLUE, alpha=0.16,
                        label="Δψ_w = 1.0–2.0 V band")
    ax.plot(til, y, "o-", color=BLUE, lw=3.0, ms=9, mec="white", mew=1.2,
            label="Δψ_w = nominal")
    ax.set_yscale("log")
    _set_log_unicode(ax, "y", dense=True)
    ax.set_xlabel("Interfacial layer  t_IL  [nm]", fontsize=13, fontweight="bold")
    ax.set_ylabel("Allowable  D_it  upper bound  [cm⁻²eV⁻¹]",
                  fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.set_title(f"Prescription:  loss ≤ {mw_loss_max:.0f}%  AND  MW ≥ {target_mw:.2f} V",
                 fontsize=11.5)
    fig.tight_layout()
    return fig


NAVY = "#24406e"    # 소패널 제목 색 (1d민감도 수정안)


def plot_sensitivity(panels, target_mw):
    """1D 민감도 — 변수마다 소패널 1장씩(small multiples). 2026-08-05 '1d민감도 수정안' 형식.

    이전 형식(배율 축 한 장에 4곡선)을 버린 이유: 한 축에 몰아넣으니 (i) 곡선끼리 서로
    가리고, (ii) 가로축이 '배율'이라 t_FE 12 nm 가 몇 배인지 암산해야 했고, (iii) 끝점
    라벨을 오른쪽 여백에 밀어 넣느라 정작 그림이 좁아졌다. 변수별로 축을 나누면 x축에
    **실제 단위**를 쓸 수 있고, y축(MW)만 공유하므로 패널 사이 높이 비교는 그대로 된다.

    ★t_IL 패널의 두 곡선(기준 D_it · 높은 D_it)은 반드시 함께 그린다 — 빨강만 있으면
      "t_IL은 중요하지 않다"로 읽히는데 옆 design map은 t_IL이 두 축 중 하나다.
      t_IL 0.5→2.0 nm의 MW 변화는 D_it에 따라 1×10¹¹ −0.6 % / 1×10¹² −6.0 % /
      5×10¹² −33.4 % 다(계면 트랩과 함께일 때만 문다).

    ★한글 금지 — Streamlit Cloud(Linux)에 한글 글꼴이 없어 배포하면 □로 깨진다.
      수정안 원본의 한글 제목("강유전체 두께 : 창을 키운다")은 영문으로 옮겨 쓴다.

    panels: [{"title", "xlabel", "xticks",
              "series": [{"x","y","color","ls"?,"label"?,"label_va"?}],
              "cur": (x, y) | None,      # 현재 설계점(흰 원)
              "cur_note": str | None}]   # 그 점에 붙일 설명(유도선)
    target_mw: 현재 목표 MW. 앱 전용(수정안 원본엔 없음) — 목표가 슬라이더라 선이 있어야
               "어느 변수를 얼마나 키워야 목표에 닿나"가 읽힌다. 색은 design map의
               절대기준선과 같은 검정 파선으로 통일.
    """
    fig = Figure(figsize=(9.2, 3.9), dpi=140)
    axes = fig.subplots(1, 3, sharey=True)
    # tight_layout 대신 고정 여백 — 라벨 길이가 바뀌어도 패널 폭이 흔들리지 않게(지도와 같은 이유)
    #   wspace 는 넉넉히 — 패널이 붙으면 왼쪽 패널의 마지막 눈금과 오른쪽 패널의 첫 눈금이
    #   글자끼리 겹쳐 "125" 처럼 읽힌다(실측).
    fig.subplots_adjust(left=0.088, right=0.992, bottom=0.215, top=0.870, wspace=0.17)
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]

    ylo, yhi = target_mw, target_mw
    pending = []          # 라벨은 y범위가 정해진 뒤에 붙인다(아래 참조)
    for ax, p in zip(axes, panels):
        for s in p["series"]:
            x = np.asarray(s["x"], float)
            y = np.asarray(s["y"], float)
            ax.plot(x, y, ls=s.get("ls", "-"), color=s["color"], lw=3.4,
                    solid_capstyle="round", dash_capstyle="round", zorder=3)
            ylo, yhi = min(ylo, float(y.min())), max(yhi, float(y.max()))
            if s.get("label"):
                pending.append((ax, s, float(x[-1]), float(y[-1])))

        ax.axhline(target_mw, color="black", ls="--", lw=1.6, zorder=2)

        cur = p.get("cur")
        if cur is not None:
            # 축 끝에 걸려도 반쪽만 그려지지 않도록 clip 해제
            ax.plot([cur[0]], [cur[1]], "o", ms=10.5, mfc="white", mec="#333",
                    mew=2.4, zorder=7, clip_on=False)

        xs = np.concatenate([np.asarray(s["x"], float) for s in p["series"]])
        ax.set_xlim(float(xs.min()), float(xs.max()))
        ax.set_xticks(p["xticks"])
        # 소수 눈금이 하나라도 있으면 전부 한 자리로 (0.5·1·1.5·2 → 0.5·1.0·1.5·2.0)
        _fr = any(abs(t - round(t)) > 1e-9 for t in p["xticks"])
        ax.set_xticklabels([f"{t:.1f}" if _fr else f"{t:g}" for t in p["xticks"]])
        ax.set_xlabel(p["xlabel"], fontsize=13, fontweight="bold")
        ax.set_title(p["title"], fontsize=11.5, color=NAVY, pad=9)
        ax.tick_params(labelsize=11.5)
        ax.set_axisbelow(True)
        ax.grid(True, color="0.90", lw=1.0)
        for sp in ax.spines.values():
            sp.set_linewidth(1.6)
            sp.set_color("#1a1a1a")

    axes[0].set_ylabel("Memory window  MW  [V]", fontsize=13, fontweight="bold")
    pad = 0.09 * max(yhi - ylo, 0.2)
    axes[0].set_ylim(ylo - pad, yhi + pad)      # sharey → 세 패널 공통
    y0, y1 = axes[0].get_ylim()

    # ── 곡선 이름표·현재점 설명은 y범위가 정해진 뒤에 붙인다 ────────────────
    #   위아래 어느 쪽에 놓을지는 그 곡선이 패널 안에서 어디쯤 있느냐에 달렸다.
    #   t_FE를 키우면 세 패널의 곡선이 통째로 위로 올라가는데, 그때도 "선 위"를
    #   고집하면 이름표가 제목 위로 튀어나간다(실측: t_FE 18 nm).
    def _frac(v):
        return (v - y0) / (y1 - y0)

    for ax, s, ex, ey in pending:
        # 곡선 오른쪽 끝에 그 곡선 색으로 직접 붙인다 — 범례 상자를 따로 두면 좁은
        # 패널을 더 잡아먹고 눈이 그림↔범례를 왕복해야 한다.
        va = s.get("label_va", "bottom")
        f = _frac(ey)
        if va == "bottom" and f > 0.86:     # 위로 붙일 자리가 없으면 아래로
            va = "top"
        elif va == "top" and f < 0.14:      # 아래로 붙일 자리가 없으면 위로
            va = "bottom"
        ax.annotate(s["label"], xy=(ex, ey), xytext=(-3, 10 if va == "bottom" else -10),
                    textcoords="offset points", ha="right", va=va,
                    fontsize=12, color=s["color"], fontweight="bold",
                    zorder=6, path_effects=halo)

    for ax, p in zip(axes, panels):
        cur, note = p.get("cur"), p.get("cur_note")
        if cur is None or not note:
            continue
        xa, xb = ax.get_xlim()
        right = (cur[0] - xa) / (xb - xa) > 0.55      # 오른쪽에 있으면 왼쪽으로 뺀다
        up = _frac(cur[1]) < 0.72                     # 위쪽에 있으면 아래로 단다
        ax.annotate(note, xy=cur,
                    xytext=(-14 if right else 14, 34 if up else -34),
                    textcoords="offset points",
                    ha="right" if right else "left", va="center",
                    fontsize=11, color="#555", zorder=7, path_effects=halo,
                    arrowprops=dict(arrowstyle="-", color="#999", lw=1.0,
                                    shrinkA=2, shrinkB=6))

    # 목표선은 세 패널을 가로지르므로 이름표는 맨 왼쪽에 한 번만 단다
    axes[0].annotate(f"target {target_mw:.2f} V", xy=(0.02, target_mw),
                     xycoords=("axes fraction", "data"), xytext=(0, 4),
                     textcoords="offset points", ha="left", va="bottom",
                     fontsize=11, color="black", fontweight="bold", zorder=6,
                     path_effects=halo)
    return fig
