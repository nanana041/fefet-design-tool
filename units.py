"""
units.py — 단위 변환 단일 출처 (single source of truth).

원칙: 내부 계산은 SI (V, m, C/m^2, V/m). 입출력만 편의단위.
모든 변환은 여기 모은다. 다른 모듈에서 인라인 변환 금지.
"""

# ---- 물리 상수 (SI) ----
Q = 1.602176634e-19      # 전하 [C]
EPS0 = 8.8541878128e-12  # 진공 유전율 [F/m]
KB = 1.380649e-23        # 볼츠만 상수 [J/K]
T_ROOM = 300.0           # 상온 [K]
NI_SI = 1.0e16           # Si 진성 캐리어 농도 [m^-3] (~1e10 cm^-3)
EPS_SI = 11.7            # Si 비유전율 (상대값, 무차원)

def kT_q(T=T_ROOM):
    """열전압 kT/q [V]."""
    return KB * T / Q

# ---- 길이 ----
def nm_to_m(x):  return x * 1e-9
def m_to_nm(x):  return x * 1e9

# ---- 전계 ----
def MVcm_to_Vm(x):  return x * 1e8     # 1 MV/cm = 1e8 V/m
def Vm_to_MVcm(x):  return x * 1e-8

# ---- 분극 / 면전하 ----
def uCcm2_to_Cm2(x):  return x * 1e-2  # 1 μC/cm^2 = 1e-2 C/m^2
def Cm2_to_uCcm2(x):  return x * 1e2

# ---- 면밀도 (cm^-2 -> m^-2) ----
def percm2_to_perm2(x):  return x * 1e4
def perm2_to_percm2(x):  return x * 1e-4

# ---- 도핑 (cm^-3 -> m^-3) ----
def percm3_to_perm3(x):  return x * 1e6

# ---- Dit (cm^-2 eV^-1 -> m^-2 eV^-1) ----
# [규약] frozen trap 전하는  Q_it[C/m^2] = q * Dit[m^-2 eV^-1] * Δψ[V]  로 계산한다.
# (Δψ를 V로 두면 eV로 둔 것과 수치적으로 동일하므로 q는 한 번만 곱한다.)
# 따라서 변환은 ×1e4 만 한다. 절대 /Q 하지 말 것 → 하면 q배(≈1.6e-19) 오차.
def dit_percm2eV_to_perm2eV(x):  return x * 1e4   # cm^-2eV^-1 -> m^-2 eV^-1  (★이걸 써라)

# (deprecated·폐기) m^-2 J^-1 변환. frozen-trap 식엔 절대 쓰면 안 됨(q배 오차) → 호출 시 예외로 차단.
def dit_to_SI(x):
    raise NotImplementedError(
        "dit_to_SI는 폐기됨(q배 오차 유발). frozen-trap 전하는 dit_percm2eV_to_perm2eV(×1e4)를 써라.")
