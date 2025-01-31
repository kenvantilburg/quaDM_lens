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
from matplotlib.animation import FFMpegWriter, FuncAnimation

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