"""
sweeps.py — solve_device를 격자로 호출해 MW / MW_loss 곡선·맵을 만든다. [Person B]

규약:
- Pr을 sweep할 때는 Ps>Pr을 유지해야 하므로 Ps = ps_ratio*Pr 로 같이 올린다(기본 1.3배).
- 반환은 numpy 배열 담은 dict. 그림은 scripts/ 에서 그린다(여기선 데이터만).
"""
import numpy as np
from device_model import solve_device

PS_RATIO = 1.3   # Ps = PS_RATIO * Pr (Pr sweep 시)


def _apply(base, overrides):
    """base params에 overrides 적용. Pr이 바뀌면 Ps도 비례 조정."""
    p = dict(base or {})
    p.update(overrides)
    if "Pr" in overrides:
        p["Ps"] = PS_RATIO * p["Pr"]
    return p


def sweep_1d(varname, values, base=None):
    """단일 변수 sweep. 반환 dict(var, values, MW[], MW_loss[])."""
    MW, MWL = [], []
    for v in values:
        r = solve_device(_apply(base, {varname: v}))
        MW.append(r["MW"])
        MWL.append(r["MW_loss"])
    return dict(var=varname, values=np.asarray(values, float),
                MW=np.asarray(MW), MW_loss=np.asarray(MWL))


def sweep_2d(xname, xvals, yname, yvals, base=None):
    """
    2D sweep. X=xvals(가로), Y=yvals(세로). 반환 dict로 MW, MW_loss 2D 배열(shape (ny,nx)).
    """
    xvals = np.asarray(xvals, float)
    yvals = np.asarray(yvals, float)
    X, Y = np.meshgrid(xvals, yvals)            # (ny, nx)
    MW = np.empty_like(X)
    MWL = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r = solve_device(_apply(base, {xname: X[i, j], yname: Y[i, j]}))
            MW[i, j] = r["MW"]
            MWL[i, j] = r["MW_loss"]
    return dict(xname=xname, yname=yname, xvals=xvals, yvals=yvals,
                X=X, Y=Y, MW=MW, MW_loss=MWL)
