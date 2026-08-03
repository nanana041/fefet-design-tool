"""
ferro.py — Miller analytic 강유전체 분극 모델 (★ 내부 전부 SI 단위).

E [V/m], P [C/m^2], Ec [V/m], Pr/Ps [C/m^2].
step1(run_step1_pe.py)의 편의단위(MV/cm, uC/cm^2) 버전을 SI로 이식한 것.

두 가지(branch) — 부호 규약(중요):
    P_down(E): E 감소(+→-) 방향,  P_down(0) = +Pr   ← program(+Pr) 상태가 사는 가지
    P_up(E)  : E 증가(-→+) 방향,  P_up(0)   = -Pr   ← erase(-Pr) 상태가 사는 가지
검증: P_up(0)=-Pr, P_down(0)=+Pr, P(±∞)=±Ps.
"""

import numpy as np
from scipy.optimize import brentq
from units import EPS0


def miller_delta(Pr, Ps, Ec):
    """δ = Ec / ln[(1+Pr/Ps)/(1-Pr/Ps)]. (단위 무관 비율이므로 입력 단위계 자유)"""
    r = Pr / Ps
    assert 0.0 < r < 1.0, "Pr/Ps 는 (0,1) 이어야 한다 (Ps>Pr)."
    return Ec / np.log((1.0 + r) / (1.0 - r))


def P_up(E, Pr, Ps, Ec, delta=None):
    """E 증가 가지. P_up(0) = -Pr."""
    if delta is None:
        delta = miller_delta(Pr, Ps, Ec)
    return Ps * np.tanh((E - Ec) / (2.0 * delta))


def P_down(E, Pr, Ps, Ec, delta=None):
    """E 감소 가지. P_down(0) = +Pr."""
    if delta is None:
        delta = miller_delta(Pr, Ps, Ec)
    return Ps * np.tanh((E + Ec) / (2.0 * delta))


def branch_func(state):
    """state='program'(+Pr) -> P_down ; 'erase'(-Pr) -> P_up."""
    if state == "program":
        return P_down
    if state == "erase":
        return P_up
    raise ValueError(f"unknown state {state!r} (program/erase 만 허용)")


def solve_EFE(D, state, eps_FE, Pr, Ps, Ec, E_bracket=1.0e9):
    """
    변위 연속  D = ε0·ε_FE·E_FE + P_branch(E_FE)  를 E_FE에 대해 역산.

    f(E) = ε0·ε_FE·E + P(E) - D 는 E에 대해 단조증가(두 항 모두 증가)
    → 유일근. brentq로 [-E_bracket, +E_bracket]에서 찾는다.
    인자 단위: D [C/m^2], eps_FE 무차원, Pr/Ps/E [SI]. 반환 E_FE [V/m].
    """
    P = branch_func(state)
    delta = miller_delta(Pr, Ps, Ec)

    def f(E):
        return EPS0 * eps_FE * E + P(E, Pr, Ps, Ec, delta) - D

    fa, fb = f(-E_bracket), f(+E_bracket)
    if fa * fb > 0.0:
        raise ValueError(
            f"solve_EFE: bracket 안에 근 없음 (D={D:.3e}, f(-)={fa:.3e}, f(+)={fb:.3e}). "
            "E_bracket 키우거나 D/단위 점검."
        )
    return brentq(f, -E_bracket, E_bracket, xtol=1e-2, rtol=1e-12)


def pe_loop(Pr, Ps, Ec, Emax=None, n=400):
    """플롯용 P-E loop. 반환 (E[V/m], P_up[C/m^2], P_down[C/m^2])."""
    if Emax is None:
        Emax = 4.0 * Ec
    E = np.linspace(-Emax, Emax, n)
    d = miller_delta(Pr, Ps, Ec)
    return E, P_up(E, Pr, Ps, Ec, d), P_down(E, Pr, Ps, Ec, d)
