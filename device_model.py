"""
device_model.py — FeFET MFIS compact model (★ 옵션 B: state-dependent frozen trap).

핵심 외부 인터페이스: solve_device(params) -> dict   (시그니처 변경 금지)

[모델 요약 — 옵션 B]
- 게이트 적층 1D: Gate / FE(t_FE) / IL(t_IL) / p-Si.
- Vth는 반도체 threshold 조건 ψ_s = 2φ_F 에서 게이트 전압식을 직접 평가해 추출한다
  (threshold에서 ψ_s가 고정이라 외곽 ψ_s 루트파인드 불필요; 분극 역산만 brentq).
- 계면 트랩 Dit는 두 부분:
    (1) read-following  -q·Dit·(ψ_s-ψ_s0): threshold(ψ_s=ψ_s0)에서 0 → MW엔 영향X,
        Id-Vg stretch-out(F6 시각화)용. 본 함수의 MW 계산엔 들어오지 않는다.
    (2) frozen state-dependent  ∓q·Dit·(Δψ_w/2): program/erase 반대 부호로 '얼어붙은' 전하.
        ← 이 항이 MW를 줄인다(F7 design map의 Dit축). 이게 옵션 B의 핵심.
  순효과:  MW = MW_pol - q·Dit·Δψ_w/C_IL   (+ D-결합 소항)
  ※ Dit/Δψ_w는 effective 단순화이며 novelty 아님(선행연구 입력 물리 재현). 논문에 명시.

[표기 규약]
- MW_ref = 2·Ec·t_FE 는 '정규화 기준선'이며 물리적 상한이 아니다. MW_loss<0(기준선 상회) 정상.
- 내부 계산은 SI. 입력만 편의단위(units.py에서 변환). 인라인 변환 금지.
"""

import numpy as np
from scipy.optimize import brentq
from units import (Q, EPS0, EPS_SI, NI_SI, kT_q,
                   MVcm_to_Vm, Vm_to_MVcm, nm_to_m,
                   uCcm2_to_Cm2, Cm2_to_uCcm2,
                   percm2_to_perm2, percm3_to_perm3,
                   dit_percm2eV_to_perm2eV)
from ferro import solve_EFE, pe_loop


def mw_ref(Ec_MVcm, tFE_nm):
    """정규화 기준선 MW_ref = 2*Ec*t_FE  [V]. (Ec: MV/cm, tFE: nm 입력) — 상한 아님."""
    Ec = MVcm_to_Vm(Ec_MVcm)      # V/m
    tFE = nm_to_m(tFE_nm)         # m
    return 2.0 * Ec * tFE         # V


def solve_device(params: dict) -> dict:
    """
    FeFET 한 점(설계 변수 1조합)의 MW 등을 계산. (옵션 B)

    params keys (편의단위 입력):
        t_FE [nm], t_IL [nm], eps_FE, eps_IL, Pr [uC/cm^2], Ps [uC/cm^2],
        Ec [MV/cm], Na [cm^-3], Vfb [V], Qf [cm^-2], Dit [cm^-2 eV^-1],
        dpsi_w [V] (write-bias band-bending swing; 옵션 B 모델 파라미터)

    returns dict:
        Vth_prog, Vth_erase, MW, MW_ref, MW_loss(%),
        PE_loop=(E_MVcm, Pup_uCcm2, Pdown_uCcm2),
        IdVg_prog, IdVg_erase = (Vg[V], Id[A]) 튜플 또는 None.
            params["want_idvg"]=True 일 때만 계산(기본 False; sweep 성능 보호).
    """
    p = _with_defaults(params)
    d = _derive(p)

    Vth_prog, _ = _vth_state("program", p, d)
    Vth_erase, _ = _vth_state("erase", p, d)
    MW = Vth_erase - Vth_prog

    MWr = mw_ref(p["Ec"], p["t_FE"])
    MW_loss = (MWr - MW) / MWr * 100.0 if MWr > 0 else float("nan")  # 음수 가능(정상)

    E, Pup, Pdn = pe_loop(d["Pr_SI"], d["Ps_SI"], d["Ec_SI"])
    PE_loop = (Vm_to_MVcm(E), Cm2_to_uCcm2(Pup), Cm2_to_uCcm2(Pdn))

    IdVg_prog = _idvg_state("program", p, d) if p["want_idvg"] else None
    IdVg_erase = _idvg_state("erase", p, d) if p["want_idvg"] else None

    return {
        "Vth_prog": Vth_prog,
        "Vth_erase": Vth_erase,
        "MW": MW,
        "MW_ref": MWr,
        "MW_loss": MW_loss,
        "PE_loop": PE_loop,
        "IdVg_prog": IdVg_prog,
        "IdVg_erase": IdVg_erase,
    }


def in_design_window(res: dict, mw_loss_max=30.0, mw_real_min=1.0):
    """
    이중 기준 design window 판정.
      - 상대: MW_loss <= mw_loss_max (%)        (기본 30%)
      - 절대: MW_real >= mw_real_min (V)         (binary 1.0V / multilevel 3.0V)
    근거값은 예시 기준(문헌 보고 MW 범위 참고). 표준 규격 아님.
    """
    return (res["MW_loss"] <= mw_loss_max) and (res["MW"] >= mw_real_min)


def calibrate_dpsi_w(dMW_meas, Dit, base=None, bracket=(0.05, 8.0)):
    """
    측정된 MW 감소 1점에 Δψ_w를 앵커링한다 (가짜수치 금지 — 입력 dMW_meas/Dit/t_IL은
    반드시 검증된 1차 문헌에서 읽어온 값이어야 한다).

    정의: dMW_meas = MW(Dit=0) - MW(Dit, Δψ_w)  를 만족하는 Δψ_w 를 full model 역산.
      - 1차 근사식 q·Dit_SI·Δψ_w/C_IL 은 D-결합 때문에 ~16% 빗나가므로 brentq로 정확히 푼다.

    인자:
        dMW_meas [V] : 그 Dit에서 trap이 깎은 MW 감소량(양수). (문헌 측정값)
        Dit [cm^-2 eV^-1] : dMW_meas가 측정된 조건의 Dit.
        base : 그 측정 소자에 맞춘 params(t_IL, t_FE, Pr, Ec ...). 특히 t_IL이 C_IL을 정한다.
    반환: 앵커된 Δψ_w [V].

    ⚠ Δψ_w·Dit 축퇴: MW 감소는 곱 (Dit·Δψ_w)에만 의존한다(=유효 트랩 전하).
       따라서 Δψ_w 단독값은 'Dit를 그 측정조건 값으로 고정했을 때'의 의미만 갖는다.
    """
    p0 = _with_defaults(base)
    mw0 = solve_device({**p0, "Dit": 0.0})["MW"]

    def f(dp):
        mw = solve_device({**p0, "Dit": Dit, "dpsi_w": dp})["MW"]
        return (mw0 - mw) - dMW_meas

    fa, fb = f(bracket[0]), f(bracket[1])
    if fa * fb > 0.0:
        raise ValueError(
            f"calibrate_dpsi_w: bracket {bracket} 안에 해 없음 "
            f"(이 Dit로 도달 가능한 MW 감소 범위를 dMW_meas={dMW_meas}가 벗어남). "
            "Dit/측정조건을 점검하라.")
    return brentq(f, bracket[0], bracket[1], xtol=1e-4)


# ---------------- 내부 ----------------

def _with_defaults(params):
    d = dict(t_FE=10.0, t_IL=1.0, eps_FE=25.0, eps_IL=3.9,
             Pr=15.0, Ps=20.0, Ec=1.0, Na=1e17, Vfb=0.0,
             Qf=0.0, Dit=1e11,
             dpsi_w=1.5,          # ⚠ 잠정 가정값. 검증문헌 1점에 앵커 예정(calibrate_dpsi_w).
                                  #   주의: MW 감소는 곱 (Dit·Δψ_w)에만 의존(축퇴).
             # --- Id-Vg(long-channel) 옵션 (MW 계산엔 영향 없음) ---
             want_idvg=False,    # solve_device가 IdVg를 채울지 (sweep 성능 보호용 기본 False)
             mu_n=0.02,          # 전자 이동도 [m^2/Vs] (≈200 cm^2/Vs, 그림용 스케일)
             WL=10.0,            # 채널 W/L (무차원, 그림용 스케일)
             Vds_lin=0.1,        # 선형영역 Vds [V] (transfer curve용)
             n_idvg=240)         # Id-Vg ψ_s 격자점 수
    d.update(params or {})
    assert d["Ps"] > d["Pr"], "Ps must exceed Pr (tanh 모델 가정)"
    return d


def _derive(p):
    """편의단위 params -> SI 파생량 dict. 모든 단위 변환은 여기 한 곳."""
    Na_SI = percm3_to_perm3(p["Na"])
    phi_F = kT_q() * np.log(Na_SI / NI_SI)          # Fermi potential [V]
    psi_s0 = 2.0 * phi_F                            # threshold 표면전위 [V]
    tIL_m = nm_to_m(p["t_IL"])
    C_IL = EPS0 * p["eps_IL"] / tIL_m               # IL 면적 커패시턴스 [F/m^2]
    return dict(
        Na_SI=Na_SI, phi_F=phi_F, psi_s0=psi_s0,
        tFE_m=nm_to_m(p["t_FE"]), C_IL=C_IL,
        Pr_SI=uCcm2_to_Cm2(p["Pr"]), Ps_SI=uCcm2_to_Cm2(p["Ps"]),
        Ec_SI=MVcm_to_Vm(p["Ec"]),
        Qf_SI=percm2_to_perm2(p["Qf"]),
        Dit_SI=dit_percm2eV_to_perm2eV(p["Dit"]),   # m^-2 eV^-1 (★ q는 식에서 1번만)
        dpsi_w=p["dpsi_w"],
    )


def _q_dep(psi_s, Na_SI):
    """공핍 전하 Q_dep = -q·Na·W,  W = sqrt(2·ε0·εSi·ψ_s/(q·Na))  [C/m^2] (ψ_s>0)."""
    W = np.sqrt(2.0 * EPS0 * EPS_SI * psi_s / (Q * Na_SI))
    return -Q * Na_SI * W


def _q_semi(psi_s, Na_SI):
    """
    표준 MOS 표면 전하식(p형, ψ_s>0). 공핍+반전 모두 포함.
        Q_s = -sqrt(2·ε0·εSi·Na·kT)·sqrt[ (βψ_s+e^{-βψ_s}-1) + (ni/Na)^2·(e^{βψ_s}-βψ_s-1) ]
    반환 (Q_s_total [C/m^2], Q_inv [C/m^2]) — 둘 다 음수. Q_inv는 반전 기여(전체-공핍).
    (Id-Vg 전용. Vth/MW 엔진은 _q_dep만 쓰며 이 함수에 의존하지 않는다.)
    """
    kT = Q * kT_q()                      # [J]
    beta = 1.0 / kT_q()                  # [1/V]
    pref = np.sqrt(2.0 * EPS0 * EPS_SI * Na_SI * kT)   # [C/m^2]
    x = beta * psi_s
    x_inv = min(x, 300.0)                # exp 오버플로 가드(float64 한계 709 훨 아래; 실제 x_max~42라 미발동)
    dep = x + np.exp(-x) - 1.0
    inv = (NI_SI / Na_SI) ** 2 * (np.exp(x_inv) - x_inv - 1.0)
    Qs = -pref * np.sqrt(dep + inv)
    Qd = -pref * np.sqrt(dep)
    return Qs, (Qs - Qd)


def _idvg_state(state, p, d):
    """
    한 분극 상태의 Id-Vg(long-channel, 선형영역) transfer curve. 반환 (Vg[V], Id[A]).

    [구성] ψ_s를 독립변수로 격자 스윕하며 각 점에서:
      - Q_s(ψ_s): 공핍+반전 (반전전하 Q_inv가 전류를 운반)
      - 계면트랩: frozen(상태부호, 상수) + read-following -q·Dit·(ψ_s-ψ_s0)
                  → read 항이 ψ_s에 따라 자라며 Vg를 늘려 'stretch-out'을 만든다(F6).
      - Vg(ψ_s) = Vfb + ψ_s - Qtot/C_IL + E_FE·t_FE - q·Qf/C_IL  (Qf는 상수 평행이동)
      - Id = (W/L)·μ·|Q_inv|·Vds_lin           (선형영역, 그림용 상대 스케일)
    ψ_s=ψ_s0에서 read 항=0이라 Vg(ψ_s0)≈Vth (엔진의 _vth_state와 정합).
    """
    psi0 = d["psi_s0"]
    psi = np.linspace(0.10, psi0 + 0.25, int(p["n_idvg"]))
    sgn = +1.0 if state == "program" else -1.0
    Q_frozen = -Q * d["Dit_SI"] * (sgn * d["dpsi_w"] / 2.0)   # 상태별 상수

    Vg = np.empty_like(psi)
    Id = np.empty_like(psi)
    for k, ps in enumerate(psi):
        Q_s, Q_inv = _q_semi(ps, d["Na_SI"])
        Q_read = -Q * d["Dit_SI"] * (ps - psi0)              # read-following(stretch-out)
        Qtot = Q_s + Q_read + Q_frozen
        D = -Qtot
        E_FE = solve_EFE(D, state, p["eps_FE"], d["Pr_SI"], d["Ps_SI"], d["Ec_SI"])
        Vg[k] = (p["Vfb"] + ps - Qtot / d["C_IL"] + E_FE * d["tFE_m"]
                 - Q * d["Qf_SI"] / d["C_IL"])
        Id[k] = p["WL"] * p["mu_n"] * abs(Q_inv) * p["Vds_lin"]
    return Vg, Id


def idvg(params, state):
    """공개 헬퍼: params(편의단위)+state('program'/'erase') -> (Vg[V], Id[A]). 그림 스크립트용."""
    p = _with_defaults(params)
    d = _derive(p)
    return _idvg_state(state, p, d)


def _vth_state(state, p, d):
    """
    한 분극 상태(program=+Pr / erase=-Pr)의 Vth를 ψ_s=2φ_F threshold에서 직접 계산.
    반환 (Vth [V], E_FE [V/m]).
    """
    psi_s = d["psi_s0"]
    Q_s = _q_dep(psi_s, d["Na_SI"])                 # 공핍전하(음수), 두 상태 동일

    # frozen state-dependent trap 전하: program(+Pr,강반전)→음전하, erase(-Pr,축적)→양전하
    sgn = +1.0 if state == "program" else -1.0
    Q_it = -Q * d["Dit_SI"] * (sgn * d["dpsi_w"] / 2.0)

    Qtot = Q_s + Q_it
    D = -Qtot                                       # 변위 연속 D = -(Q_s+Q_it)

    E_FE = solve_EFE(D, state, p["eps_FE"], d["Pr_SI"], d["Ps_SI"], d["Ec_SI"])

    V_IL = -Qtot / d["C_IL"]
    V_FE = E_FE * d["tFE_m"]
    Vth = p["Vfb"] + psi_s + V_IL + V_FE - Q * d["Qf_SI"] / d["C_IL"]
    return Vth, E_FE


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # 한글 콘솔(cp949) 크래시 방지
    except Exception:
        pass
    r = solve_device({})
    print("=== baseline self-test (옵션 B) ===")
    for k in ["Vth_prog", "Vth_erase", "MW", "MW_ref", "MW_loss"]:
        print(f"  {k:10s} = {r[k]:+.4f}")
    if r["MW"] > r["MW_ref"] * 1.05:
        print("  [WARN] MW가 MW_ref를 크게 초과 → 단위/부호 점검 권장")
    else:
        print("  OK: MW <= MW_ref (정상 범위)")
    print("=== Dit sweep (MW가 줄어드는지) ===")
    for dit in [0.0, 1e11, 1e12, 5e12]:
        rr = solve_device({"Dit": dit})
        print(f"  Dit={dit:8.1e}  MW={rr['MW']:+.4f}  MW_loss={rr['MW_loss']:6.2f}%")
