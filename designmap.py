"""
designmap.py — 2D MW_loss design map 위의 허용 설계 영역(design window) + 경계값. [Person B]

이중 기준(상대+절대):
    상대: MW_loss <= mw_loss_max (%)   기본 30
    절대: MW_real >= mw_real_min (V)    binary 1.0 / multilevel 3.0
경계값(F8): 각 t_IL에 대해 두 기준을 모두 만족하는 Dit의 상한(Dit_max)을 구한다.
(MW_loss는 Dit↑에 증가, MW는 Dit↑에 감소 → 통과영역은 Dit <= Dit_max 형태.)
"""
import numpy as np


def window_mask(m, mw_loss_max=30.0, mw_real_min=1.0):
    """2D map dict m에서 design window 통과 여부 bool 배열(shape=MW)."""
    return (m["MW_loss"] <= mw_loss_max) & (m["MW"] >= mw_real_min)


def _crossing_logx(x, y, level):
    """y가 level을 처음 교차하는 x를 log10(x) 선형보간으로 반환. 없으면 None."""
    lx = np.log10(x)
    for k in range(len(x) - 1):
        y0, y1 = y[k], y[k + 1]
        if y0 == level:
            return x[k]
        if (y0 < level <= y1) or (y0 > level >= y1):
            t = (level - y0) / (y1 - y0)
            return 10.0 ** (lx[k] + t * (lx[k + 1] - lx[k]))
    return None


def dit_upper_bounds(m, mw_loss_max=30.0, mw_real_min=1.0):
    """
    x축=Dit, y축=t_IL 인 2D map에서 각 t_IL의 Dit 상한 경계.
    반환 list of dict: t_IL, dit_rel(상대기준 경계), dit_abs(절대기준 경계), dit_max(둘 중 binding).
    """
    assert m["xname"] == "Dit", "dit_upper_bounds는 x축이 Dit인 map을 가정"
    dit = m["xvals"]
    rows = []
    for i, til in enumerate(m["yvals"]):
        mwl = m["MW_loss"][i, :]
        mw = m["MW"][i, :]
        d_rel = _crossing_logx(dit, mwl, mw_loss_max)   # MW_loss=30 교차
        d_abs = _crossing_logx(dit, mw, mw_real_min)    # MW=min 교차
        # ★ 가장 작은 D_it(=이 행에서 가장 유리한 점)에서 이미 어느 한 기준이라도
        #   깨지면 그 행은 통째로 탈락 → 상한은 None.
        #   (목표 미달인데 상대기준 경계를 상한으로 보고하던 버그 방지;
        #    2026-08-03 승민 회신 반영. dit는 오름차순 → dit[0]이 최소.)
        feasible = (mwl[0] <= mw_loss_max) and (mw[0] >= mw_real_min)
        if not feasible:
            dit_max = None
        else:
            cands = [d for d in (d_rel, d_abs) if d is not None]
            dit_max = min(cands) if cands else dit[-1]  # 둘 다 미교차면 전 구간 통과 → 최댓값
        rows.append(dict(t_IL=float(til), dit_rel=d_rel, dit_abs=d_abs, dit_max=dit_max))
    return rows


def format_boundary_table(rows, mw_real_min=1.0):
    """경계표를 사람이 읽는 문자열로(F8). Dit 단위 cm^-2eV^-1."""
    lines = []
    lines.append(f"  design window 경계 (MW_loss<=30% AND MW>={mw_real_min:.1f}V)")
    lines.append(f"  {'t_IL[nm]':>9} | {'Dit_max(상대)':>13} | {'Dit_max(절대)':>13} | {'Dit_max(binding)':>16}")
    lines.append("  " + "-" * 60)
    for r in rows:
        def fmt(x):
            return f"{x:.2e}" if x is not None else "   none   "
        lines.append(f"  {r['t_IL']:>9.2f} | {fmt(r['dit_rel']):>13} | {fmt(r['dit_abs']):>13} | {fmt(r['dit_max']):>16}")
    return "\n".join(lines)
