"""Common imports and plot styling shared by every notebook and script here.

Imported with `from preamble import *` at the top of each module, together with
`natural_units_GeV` (units) and `macro_lens_functions` (lensing geometry). It sets
matplotlib to serif + LaTeX rendering, so a TeX installation with `latex` and
`dvipng` on the PATH is required to reproduce the figures.
"""
import numpy as np
import numpy.testing as npt
import pandas as pd
import scipy as sp
from scipy import stats
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.integrate import odeint
from tqdm import tqdm, tqdm_notebook
import random
from time import time as tictoc
from scipy.optimize import fmin
import lenstronomy
import copy

from astropy.coordinates import SkyCoord

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
from matplotlib.animation import FFMpegWriter, FuncAnimation  # needs ffmpeg on the PATH

# ---------------------------------------------------------------------------
# Color palettes (Tsitsul "normal/bright/dark/fancy/tarnish" qualitative sets).
# `xgfs_fancy6` is the palette used throughout the paper figures
# ---------------------------------------------------------------------------
xgfs_normal6 = 255**-1 * np.asarray([(64, 83, 211), (221, 179, 16), (181, 29, 20), (0, 190, 255), (251, 73, 176), (0, 178, 93), (202, 202, 202)])
xgfs_normal12 = 255**-1 * np.asarray([(235, 172, 35), (184, 0, 88), (0, 140, 249), (0, 110, 0), (0, 187, 173), (209, 99, 230), (178, 69, 2), (255, 146, 135), (89, 84, 214), (0, 198, 248), (135, 133, 0), (0, 167, 108), (189, 189, 189)])
xgfs_bright6 = 255**-1 * np.asarray([(239, 230, 69), (233, 53, 161), (0, 227, 255), (225, 86, 44), (83, 126, 255), (0, 203, 133), (238, 238, 238)])
xgfs_dark6 = 255**-1 * np.asarray([(0, 89, 0), (0, 0, 120), (73, 13, 0), (138, 3, 79), (0, 90, 138), (68, 53, 0), (88, 88, 88)])
xgfs_fancy6 = 255**-1 * np.asarray([(86, 100, 26), (192, 175, 251), (230, 161, 118), (0, 103, 138), (152, 68, 100), (94, 204, 171), (205, 205, 205)])
xgfs_tarnish6 = 255**-1 * np.asarray([(39, 77, 82), (199, 162, 166), (129, 139, 112), (96, 78, 60), (140, 159, 183), (121, 104, 128), (192, 192, 192)])

plt.rcdefaults()
fontsize = 12
from matplotlib import font_manager
from matplotlib import rcParams
from matplotlib import rc
rcParams['font.family'] = 'serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':fontsize})
rc('text', usetex=True)
custom_preamble = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}"
    }
plt.rcParams.update(custom_preamble)
