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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter, LogLocator

RED = "#d1341f"
BLUE = "#1f6fd6"
TEAL = "#0f8a7e"    # 포스터 v4 그림3 하우스 색 (P_r 곡선)
PLUM = "#8e44ad"    # 〃 (D_it=5e12 에서의 t_IL — 상호작용 곡선)

# ── design map 색띠 (2026-08-05 확정본 '스크린샷(251)'에서 픽셀로 추출) ──────
#   matplotlib 기본 컬러맵이 아니다 — 가장 가까운 것(gist_earth)과도 평균 |ΔRGB| 36
#   이라 이름으로 못 부른다. 그래서 값으로 박아 둔다. 10–15 % 부터 5 % 간격 8칸.
LOSS_BANDS = ["#1c4f7e", "#215f9a", "#0070c0", "#3d9987",
              "#80c262", "#b2d729", "#feda02", "#f8b342"]
LOSS_OVER = "#e35d4f"      # > 50 % (컬러바 위쪽 삼각형)

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
    """MW_loss design map — 2026-08-05 확정 디자인('스크린샷(251)').

    5 % 간격 이산 색띠(위 LOSS_BANDS) + >50 % 삼각형, 사선 없음.
    상대 기준선 = **굵은 검은 실선**, 절대 기준선 = **흰 파선**, 처방점 = 흰 원(검은
    테두리) + 흰 상자에 검은 숫자, 범례는 **그림 안 왼쪽 아래**.
    컬러바: 빨간 눈금(상대) · 검은 눈금(절대) · design window 브래킷.

    presc: [{t_IL, dit_max, bind}, ...] (t_IL 0.5/1.0/1.5/2.0),  mw_ref: 정규화 기준선(V).
    """
    LEVELS = list(range(10, 51, 5))      # 10,15,...,50
    Z = np.clip(m["MW_loss"], 10.0, None)  # <10은 최하위 색으로(흰 구멍 방지 → 바닥 10 고정)

    # 고정 레이아웃(라벨 길이에 따라 그림이 흔들리지 않도록 add_axes 로 위치 고정)
    fig = Figure(figsize=(8.0, 4.9))
    ax = fig.add_axes([0.095, 0.155, 0.66, 0.80])
    cax = fig.add_axes([0.875, 0.155, 0.028, 0.80])

    cmap = ListedColormap(LOSS_BANDS)
    cmap.set_over(LOSS_OVER)
    cf = ax.contourf(m["X"], m["Y"], Z, levels=LEVELS, cmap=cmap, extend="max")
    cb = fig.colorbar(cf, cax=cax, ticks=LEVELS)
    cb.set_label("MW_loss  [%]", fontsize=12.5, fontweight="bold")
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_linewidth(1.4)
    # >50 삼각형이 무슨 뜻인지 삼각형 옆에 직접 쓴다(눈금이 아니라 구간이라서)
    cax.text(1.45, 1.055, "> 50", transform=cax.transAxes, ha="left", va="center",
             fontsize=10, clip_on=False)

    # colorbar 눈금: 빨강 = 상대 기준(loss_max), 검정 = 절대 기준의 등가 손실%
    yt = cax.get_yaxis_transform()   # x: cax 축분율, y: 데이터값(%)
    cax.plot([-0.1, 1.1], [mw_loss_max, mw_loss_max], transform=yt, color=RED, lw=3.0,
             clip_on=False, zorder=5)
    _loss_abs = (mw_ref - target_mw) / mw_ref * 100.0   # 절대선 등가 손실%
    if 10.0 <= _loss_abs <= 50.0:
        cax.plot([-0.1, 1.1], [_loss_abs, _loss_abs], transform=yt, color="black",
                 lw=3.0, clip_on=False, zorder=5)
    # design window 브래킷 — 왼쪽부터 [통과율(회색) · design window(굵게) · 브래킷 · 바]
    #   ★브래킷 끝은 loss_max 가 아니라 **두 기준 중 더 빡센 쪽**이다. 둘 다 만족해야
    #     통과이므로, 절대 기준이 더 빡세면(목표를 올리면) 통과 구간은 검은 눈금에서
    #     끊긴다. loss_max 로 고정해 두면 브래킷이 옆의 통과율 숫자와 어긋난다
    #     (실측: 목표 1.60 V → 브래킷 10–30 % 인데 실제 통과는 20 % 까지).
    bx = -1.4
    win_top = min(mw_loss_max, _loss_abs) if _loss_abs >= 10.0 else 10.0
    if win_top > 11.0:
        for seg in ([bx, bx], [10, win_top]), ([bx, bx + 0.4], [10, 10]), ([bx, bx + 0.4], [win_top, win_top]):
            cax.plot(seg[0], seg[1], transform=yt, color="#555", lw=1.3, clip_on=False, zorder=4)
        cax.text(bx - 0.9, (10 + win_top) / 2, "design window", transform=yt, rotation=90,
                 ha="center", va="center", fontsize=9.5, fontweight="bold", color="#222", clip_on=False)
        cax.text(bx - 2.1, (10 + win_top) / 2, f"{frac:.1f} % of the swept grid",
                 transform=yt, rotation=90, ha="center", va="center", fontsize=8.5,
                 color="#888", clip_on=False)
    else:
        # 통과 구간이 색띠 바닥(10 %)보다 아래 → 브래킷을 그릴 자리가 없다. 숫자만 남긴다.
        cax.text(bx - 0.9, 30, f"{frac:.1f} % of the swept grid", transform=yt, rotation=90,
                 ha="center", va="center", fontsize=8.5, color="#888", clip_on=False)

    # 상대 기준선 = 굵은 검은 실선 at MW_loss = loss_max
    has_rel = np.nanmin(m["MW_loss"]) <= mw_loss_max <= np.nanmax(m["MW_loss"])
    if has_rel:
        ax.contour(m["X"], m["Y"], m["MW_loss"], levels=[mw_loss_max], colors="black",
                   linewidths=4.0, zorder=3)
    # 절대 기준선 = 흰 파선 at MW = target
    #   ★확정본은 이 선이 빨강·주황 위에 있어 순백으로 충분하지만, 앱은 슬라이더로
    #     선이 노랑(#feda02) 띠 위로 옮겨 갈 수 있고 거기서는 흰 선이 사라진다.
    #     → 어두운 후광을 얇게 둘러 어느 띠 위에서든 남게 한다(모양은 흰 파선 그대로).
    has_abs = m["MW"].min() <= target_mw <= m["MW"].max()
    if has_abs:
        # ★contour 의 linestyles 는 대시 튜플을 그대로 못 받는다(리스트로 감싸야 한다)
        c_abs = ax.contour(m["X"], m["Y"], m["MW"], levels=[target_mw], colors="white",
                           linestyles=[(0, (3.4, 2.4))], linewidths=2.8, zorder=3)
        _set_pe(c_abs, 4.8, "#3a3a3a")

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
    for sp in ax.spines.values():        # 확정본의 굵은 검은 테두리
        sp.set_linewidth(1.6)
        sp.set_color("#1a1a1a")

    # ── 범례 = 그림 안 왼쪽 아래 (확정본 배치) ────────────────────────────
    #   두 기준선은 "무엇과 같은가"까지 적는다: 상대 기준 30 %는 이 스택에서 MW 1.4 V
    #   와 같은 말이고, 절대 기준 1.0 V는 손실 50 %와 같은 말이다. 이 등가가 없으면
    #   두 선이 왜 따로 필요한지가 안 보인다.
    #   ★그려지지 않은 선은 범례에도 넣지 않는다(슬라이더를 끝까지 밀면 선이 격자
    #     밖으로 나간다 — 없는 선의 이름표만 남으면 고장으로 읽힌다).
    hs, ls_ = [], []
    if has_rel:
        hs.append(Line2D([], [], color="black", lw=4.0))
        ls_.append(f"MW_loss = {mw_loss_max:.0f} %  (relative)  ≡  "
                   f"MW = {mw_ref * (1.0 - mw_loss_max / 100.0):.2f} V")
    if has_abs:
        # 흰 파선은 흰 범례 바탕에서 사라지므로, 지도에서처럼 어두운 선을 깔고 그 위에
        # 흰 파선을 얹어 두 개를 한 칸에 겹쳐 그린다(HandlerTuple).
        # ★matplotlib은 대시 길이에 선폭을 곱한다(lines.scale_dashes) — 그대로 4.5/3.0을
        #   주면 굵은 선에서 대시 한 칸이 범례 손잡이보다 길어져 통짜 막대로 보인다.
        #   → 선폭으로 나눈 값을 넣어 실제 4.5pt/3.0pt 가 되게 한다.
        hs.append((Line2D([], [], color="#3a3a3a", lw=5.4),
                   Line2D([], [], color="white", lw=3.0, ls=(0, (1.5, 1.0)))))
        ls_.append(f"MW = {target_mw:.2f} V  (absolute)  ≡  {_loss_abs:.0f} %")
    hs.append(Line2D([], [], ls="none", marker="o", mfc="white", mec="black",
                     mew=2.6, ms=10))
    ls_.append("allowable D_it  [×10¹² cm⁻²eV⁻¹]")
    # ★HandlerTuple(ndivide=1) 이라야 두 선이 **겹쳐** 그려진다. ndivide=None 은 손잡이
    #   칸을 튜플 개수만큼 나눠 나란히 그린다(실측: 왼쪽 절반 검정 + 오른쪽 절반 흰 조각).
    leg = ax.legend(hs, ls_, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
                    loc="lower left", fontsize=10.5, framealpha=1.0, facecolor="white",
                    edgecolor="black", borderpad=0.6, labelspacing=0.6, handlelength=2.4,
                    handletextpad=1.0)
    leg.get_frame().set_linewidth(1.4)
    leg.set_zorder(8)

    # ── 처방점(흰 원) + 허용 D_it 숫자 상자 ────────────────────────────────
    #   확정본은 점·상자를 흰 바탕 + 검은 테두리로 통일한다(어느 기준이 구속인지는 지도
    #   아래 안내 띠가 말한다). "none"(스윕 끝까지 두 기준 다 안 걸림)만 회색으로 남긴다
    #   — 그 값은 진짜 상한이 아니라 격자 끝이라, 검정으로 칠하면 걸리지도 않은 기준이
    #   상한을 정한 것처럼 보인다.
    #   ★범례가 그림 안에 있으므로 상자가 그 뒤로 숨을 수 있다(실측: 목표 1.60 V에서
    #     t_IL 0.5 행의 숫자가 범례에 완전히 가려졌다). 자리를 [왼쪽 → 오른쪽 → 위쪽]
    #     순서로 시도해 범례와도 축 밖과도 겹치지 않는 첫 자리에 놓는다.
    FigureCanvasAgg(fig)                     # 범례 크기를 재려면 렌더러가 필요하다
    lb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
    W = fig.get_size_inches()[0] * 0.66 * 72.0      # 축 폭 [pt]
    H = fig.get_size_inches()[1] * 0.80 * 72.0      # 축 높이 [pt]
    for p in presc:
        d, til, bind = p.get("dit_max"), p.get("t_IL"), p.get("bind")
        if d is None or d > 1.02e13:
            continue
        ec = "#6b6b6b" if bind == "none" else "black"
        ax.plot([d], [til], "o", mfc="white", mec=ec, mew=3.0, ms=11, zorder=6)

        txt = "≥10" if bind == "none" else f"{d / 1e12:.1f}"
        fx = (np.log10(d) - (11 - PAD_DEC)) / (2 + 2 * PAD_DEC)     # 마커 위치[축 분율]
        fy = (til - (0.5 - PAD_TIL)) / (1.5 + 2 * PAD_TIL)
        bw, bh = (len(txt) * 8.2 + 12) / W, 26.0 / H                # 상자 크기[축 분율]
        dy0 = -15 if til >= 1.9 else (15 if til <= 0.6 else 0)      # 위/아래 끝이면 안쪽으로
        pick = None
        for xoff, dy, ha in ((-18, dy0, "right"), (18, dy0, "left"),
                             (-18, 30, "right"), (18, 30, "left"), (0, 34, "center")):
            cx = fx + xoff / W + (0 if ha == "center" else
                                  (-bw / 2 if ha == "right" else bw / 2))
            cy = fy + dy / H
            x0, x1, y0, y1 = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2
            if x0 < 0.004 or x1 > 0.996 or y0 < 0.004 or y1 > 0.996:
                continue                                            # 축 밖으로 나감
            if not (x1 < lb.x0 or x0 > lb.x1 or y1 < lb.y0 or y0 > lb.y1):
                continue                                            # 범례와 겹침
            pick = (xoff, dy, ha)
            break
        xoff, dy, ha = pick or (-18, dy0, "right")
        ax.annotate(txt, xy=(d, til), xytext=(xoff, dy),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=13, fontweight="bold", color=ec,
                    bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=ec, lw=2.0),
                    zorder=7)
    return fig


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
