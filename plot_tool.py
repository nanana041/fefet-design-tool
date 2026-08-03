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
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter, LogLocator

RED = "#d1341f"
BLUE = "#1f6fd6"

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
    fig = plt.figure(figsize=(8.0, 4.9))
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

    # 동적 처방: 흰 원마커 + 빨간 박스 숫자(×10¹²) — 그림 밖으로 안 나가게 위치 보정
    for p in presc:
        d, til = p.get("dit_max"), p.get("t_IL")
        if d is None or d > 1.02e13:
            continue
        ax.plot([d], [til], "o", mfc="white", mec=RED, mew=2.2, ms=9, zorder=6)
        logpos = (np.log10(d) - 11.0) / 2.0                       # 0(1e11)~1(1e13)
        xoff, ha = (16, "left") if logpos < 0.24 else (-16, "right")  # 왼쪽 끝이면 박스 오른쪽
        dy = -13 if til >= 1.9 else (13 if til <= 0.6 else 0)     # 위/아래 끝이면 안쪽으로
        ax.annotate(f"{d/1e12:.1f}", xy=(d, til), xytext=(xoff, dy),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=12, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.28", fc=RED, ec="white", lw=0.8),
                    zorder=7)

    ax.set_xscale("log")
    _set_log_unicode(ax, "x")
    ax.set_xlim(1e11, 1e13)     # ★고정
    ax.set_ylim(0.5, 2.0)       # ★고정
    ax.set_xlabel("Interface trap density  D_it  [cm⁻²eV⁻¹]", fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Interfacial layer thickness  t_IL  [nm]", fontsize=12.5, fontweight="bold")
    ax.tick_params(labelsize=11)
    return fig     # 범례(MW_loss·MW·allowable D_it)는 앱에서 그림 아래에 표시


def _clean(vals):
    return np.array([np.nan if v is None else float(v) for v in vals], float)


def plot_prescription(til, dit_nom, dit_lo, dit_hi, mw_loss_max, target_mw, show_band):
    """각 t_IL의 허용 D_it 상한 곡선 (+ Δψ_w 밴드). dit_lo=Δψ_w 1.0(높음), dit_hi=Δψ_w 2.0(낮음)."""
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
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


def plot_sensitivity(tfe_vals, tfe_mw, til_vals, til_mw, target_mw, cur_tfe):
    """1D 민감도: (좌) MW vs t_FE, (우) MW vs t_IL. 목표선·현재 t_FE 표시."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axL.plot(tfe_vals, tfe_mw, "o-", color=BLUE, lw=3.0, ms=8, mec="white", mew=1.0)
    axL.axhline(target_mw, color=RED, ls="--", lw=2.2, label=f"target {target_mw:.2f} V")
    axL.axvline(cur_tfe, color="#666", ls=":", lw=1.8, label=f"current t_FE {cur_tfe:.1f} nm")
    axL.set_xlabel("Ferroelectric thickness  t_FE  [nm]", fontsize=12.5, fontweight="bold")
    axL.set_ylabel("Memory window  MW  [V]", fontsize=12.5, fontweight="bold")
    axL.grid(True, alpha=0.3); axL.legend(fontsize=10); axL.tick_params(labelsize=11)

    axR.plot(til_vals, til_mw, "s--", color=RED, lw=3.0, ms=8, mec="white", mew=1.0)
    axR.set_xlabel("Interfacial layer thickness  t_IL  [nm]", fontsize=12.5, fontweight="bold")
    axR.set_ylabel("Memory window  MW  [V]", fontsize=12.5, fontweight="bold")
    axR.grid(True, alpha=0.3); axR.tick_params(labelsize=11)
    axR.set_title("(t_IL alone: negligible effect)", fontsize=10)
    fig.tight_layout()
    return fig
