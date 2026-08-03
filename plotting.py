"""
plotting.py — 공통 그림 스타일/저장 헬퍼.

그림 라벨은 영문으로(한글 폰트 깨짐 방지). 저장은 results/F<n>_<desc>.png.
"""
import os
import matplotlib
matplotlib.use("Agg")          # 헤드라인/콘솔 환경에서도 저장만 (창 안 띄움)
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
})


def savefig(fig, name):
    """results/ 에 저장하고 절대경로 반환."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.normpath(path)
