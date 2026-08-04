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

    # 동적 처방: 흰 원마커 + 흰 박스 숫자(×10¹²) — 그림 밖으로 안 나가게 위치 보정
    #   ★원 테두리 색 = 그 점의 상한을 정한 기준의 색. 상대 기준이면 빨강(빨간 실선 위),
    #     절대 기준이면 검정(검은 파선 위). 점이 어느 선을 따라가고 있는지가 색으로 보인다.
    for p in presc:
        d, til = p.get("dit_max"), p.get("t_IL")
        if d is None or d > 1.02e13:
            continue
        bc = "black" if p.get("bind") == "absolute" else RED   # 구속 기준의 색
        ax.plot([d], [til], "o", mfc="white", mec=bc, mew=2.2, ms=9, zorder=6)
        logpos = (np.log10(d) - 11.0) / 2.0                       # 0(1e11)~1(1e13)
        xoff, ha = (16, "left") if logpos < 0.24 else (-16, "right")  # 왼쪽 끝이면 박스 오른쪽
        dy = -13 if til >= 1.9 else (13 if til <= 0.6 else 0)     # 위/아래 끝이면 안쪽으로
        ax.annotate(f"{d/1e12:.1f}", xy=(d, til), xytext=(xoff, dy),
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


def plot_sensitivity(curves, target_mw):
    """정규화 1D 민감도 — 포스터 v4 그림3과 같은 형식(배율 축 한 장).

    x = 설계 변수 / 그 변수의 현재 기준값. 축이 하나뿐이라 곡선끼리 기울기를 비교하는
    것이 정당해진다. (v3의 트윈축[아래 t_FE·위 t_IL]은 같은 가로 거리가 변수마다 다른
    양을 뜻해서, 정작 그 그림의 메시지였던 '기울기 비교'가 성립하지 않았다.)

    ★보라 파선(D_it = 5×10¹²에서의 t_IL)은 반드시 함께 그린다 — 빼면 "t_IL은 중요하지
      않다"로 읽히는데, 옆 design map은 t_IL이 두 축 중 하나다. 포스터·앱이 스스로
      모순되는 그림 두 장을 걸게 된다. t_IL 0.5→2.0 nm의 MW 변화는 D_it에 따라
      1×10¹¹ −0.6 % / 1×10¹² −6.0 % / 5×10¹² −33.4 % 다(계면 트랩과 함께일 때만 문다).

    ★×1 = **표 1 기준값 고정**(t_FE 10 nm · P_r 15 · t_IL 1.0 nm). 현재 슬라이더 값으로
      정규화하면 슬라이더를 움직일 때마다 축의 의미가 바뀌어, 앱이 논문 수치
      (MW 1.787 / 80.6 % / 처방 8.6·4.9·3.4·2.6)와 대조가 안 된다. 대신 "지금 내가
      어디 있나"는 곡선 위 현재 설계점 마커(`cur`)가 맡는다.

    curves: [{"name", "x"(배율), "mw", "color", "ls", "lo", "hi", "off"?, "cur"?}]
            cur = (x, mw) 현재 설계점. 없으면 안 찍는다.
    target_mw: 현재 목표 MW. 앱 전용(포스터엔 없음) — 목표가 슬라이더라 선이 있어야
               "어느 변수를 얼마나 키워야 목표에 닿나"가 바로 읽힌다. 색은 design map의
               절대기준선과 같은 **검정 파선**으로 통일(빨강은 t_IL 곡선 색과 충돌).
    """
    fig = Figure(figsize=(6.9, 4.3))
    ax = fig.add_axes([0.125, 0.170, 0.790, 0.795])

    # 라벨은 곡선·목표선 위를 지날 수밖에 없다(목표가 슬라이더라 위치가 계속 바뀜)
    # → 흰 후광을 둘러 겹쳐도 읽히게 한다. 고정 오프셋으로는 회피가 불가능하다.
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]

    xlo, xhi = 1.0, 1.0
    ylo, yhi = target_mw, target_mw
    ends = []
    for c in curves:
        x = np.asarray(c["x"], float)
        y = np.asarray(c["mw"], float)
        ax.plot(x, y, ls=c.get("ls", "-"), color=c["color"], lw=3.0, zorder=3)
        ax.annotate(c["lo"], xy=(x[0], y[0]), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=9.5, color=c["color"], zorder=6, path_effects=halo)
        ends.append((float(x[-1]), float(y[-1]),
                     f"{c['name']}  {c['hi']}".rstrip(), c["color"]))
        xlo, xhi = min(xlo, float(x.min())), max(xhi, float(x.max()))
        ylo, yhi = min(ylo, float(y.min())), max(yhi, float(y.max()))

    # 목표 MW — 포스터엔 없지만 앱은 목표가 슬라이더라 여기 있는 게 맞다
    ax.axhline(target_mw, color="black", ls="--", lw=1.7, zorder=2)
    ax.annotate(f"target {target_mw:.2f} V", xy=(xlo, target_mw), xytext=(2, 4),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=10, color="black", fontweight="bold", zorder=6,
                path_effects=halo)

    # ×1 = 표 1 기준값(고정). 축이 흔들리지 않아야 논문 수치와 대조가 된다.
    ax.axvline(1.0, color="0.6", lw=1.4, ls=(0, (4, 3)), zorder=2)
    # (한글 금지 — Streamlit Cloud(Linux)에 한글 글꼴이 없어 □로 깨진다)
    ax.annotate("×1 = paper baseline", xy=(1.0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-5, -12), textcoords="offset points", ha="right", va="top",
                fontsize=9, color="0.35", zorder=6, path_effects=halo)

    # 현재 설계점 — 슬라이더를 움직이면 이 점이 곡선을 따라 미끄러진다
    for c in curves:
        if c.get("cur") is None:
            continue
        cx, cy = c["cur"]
        ax.plot([cx], [cy], "o", ms=11, mfc="white", mec=c["color"], mew=2.6, zorder=7)

    xs, ys = xhi - xlo, max(yhi - ylo, 0.2)
    ax.set_xlim(xlo - 0.06 * xs, xhi + 0.42 * xs)     # 오른쪽 = 끝점 라벨 자리
    ax.set_ylim(ylo - 0.10 * ys, yhi + 0.10 * ys)
    xt = [t for t in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0) if xlo - 1e-9 <= t <= xhi + 1e-9]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"×{t:g}" for t in xt])       # mathtext 금지 → 유니코드 ×
    # ★포스터 그림3은 이 축을 한글("설계 변수 / 그 변수의 기준값")로 쓰지만, 앱은 영문으로
    #   둔다 — Streamlit Cloud(Linux)에 한글 글꼴이 없어 배포하면 전부 □로 깨진다.
    ax.set_xlabel("Design variable  /  paper baseline (Table 1)",
                  fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Memory window  MW  [V]", fontsize=12.5, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.set_axisbelow(True)
    ax.grid(True, color="0.90", lw=1.0)

    # ── 끝점 라벨을 오른쪽 여백에 세로로 겹치지 않게 배치 ────────────────────
    #   포스터(_시안_민감도_qfdit.py)는 곡선마다 위치를 손으로 잡아 뒀지만, 앱은
    #   슬라이더로 곡선이 움직여서 그 방식이 통하지 않는다. 곡선끝 y가 서로 붙으면
    #   (예: P_r·t_IL이 둘 다 기준점 근처로 수렴) 라벨이 그대로 포개진다.
    #   → 공통 x(오른쪽 여백)에 모으고, y만 최소간격으로 밀어 분리한 뒤 유도선을 잇는다.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    gut = (xhi + 0.05 * xs - x0) / (x1 - x0)          # 라벨 왼쪽 끝(axes 분율)
    LH = 11.0 / (fig.get_size_inches()[1] * 72 * 0.795)   # 한 줄 높이(axes 분율)
    items = sorted(
        ({"f": (ey - y0) / (y1 - y0), "n": 1 + t.count("\n"), "t": t, "c": col,
          "ex": ex, "ey": ey} for ex, ey, t, col in ends),
        key=lambda d: d["f"])
    top = -1.0
    for it in items:                                   # 아래→위로 밀어 올리며 분리
        half = it["n"] * LH / 2 + 0.012
        it["f"] = max(it["f"], top + half)
        top = it["f"] + half
    over = top - 1.0
    if over > 0:                                       # 위로 넘치면 통째로 내린다
        for it in items:
            it["f"] -= over
    for it in items:
        ly = y0 + it["f"] * (y1 - y0)
        if abs(ly - it["ey"]) > 0.01 * (y1 - y0):      # 옮긴 만큼 유도선을 그어 준다
            ax.plot([it["ex"], x0 + gut * (x1 - x0)], [it["ey"], ly],
                    color=it["c"], lw=0.9, ls=(0, (2, 2)), alpha=0.75, zorder=4)
        ax.text(gut + 0.012, it["f"], it["t"], transform=ax.transAxes,
                ha="left", va="center", fontsize=11, color=it["c"],
                fontweight="bold", zorder=6, path_effects=halo)
    return fig
