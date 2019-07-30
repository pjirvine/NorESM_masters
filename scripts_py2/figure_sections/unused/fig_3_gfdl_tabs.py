
"""
Figure 4 - plot overlays
"""

out_dir = '/n/home03/pjirvine/projects/fraction_better_off/figures/fig_4/'

in_dir = '/n/home03/pjirvine/projects/fraction_better_off/tables/'
df = pd.read_csv(in_dir+'anom_SREX.csv')

import matplotlib.pyplot as plt
plt.rcdefaults()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection

def label(xy, text):
    y = xy[1] - 0.1  # shift y-value for label so that it's below the artist
    plt.text(xy[0], y, text, ha="center", family='sans-serif', size=14)

def markers_4(xy,results, abbv):
    # xy gives central point
    # results = [[T,F], ... ]
    # first = greater / lesser (red / green)
    # second = significant / not (filled / empty)
    # ORDER = SATmax, P5, SAT, P-E
    markers = ['X','^','s','o']
    
    colors = ['b' if x[0] else 'r' for x in results] # read first element of each result to determine color
    fills = ['full' if x[1] else 'none' for x in results] # read second element of each result to determine fill
    
    dxy = 0.03 # Shift points off centre by this amount
    shifts = dxy * np.mgrid[1:-1:-2j, 1:-1:-2j].reshape(2, -1).T # [1,1],..[-1,-1]
    
    for JDX in range(4):
        plt.plot(grid[IDX][0] + shifts[JDX][0], grid[IDX][1] + shifts[JDX][1], 
                 marker=markers[JDX], color=colors[JDX], fillstyle=fills[JDX], markersize=18)
        label(grid[IDX], abbv)

def labeled_boxes_4(xy,results, abbv):
    # xy gives central point
    # results = [[T,F], ... ]
    # first = greater / lesser (red / green)
    # second = significant / not (filled / empty)
    # ORDER = SATmax, P5, SAT, P-E
    markers = ['s','s','s','s']
    sub_labels = ['Tx','Px','T','PE']
    
    colors = [blue if x[0] else 'r' for x in results] # read first element of each result to determine color
    alphas = [1.0 if x[1] else std_alpha for x in results] # read second element of each result to determine fill
    
    dxy = 0.03 # Shift points off centre by this amount
    shifts = dxy * np.mgrid[1:-1:-2j, 1:-1:-2j].reshape(2, -1).T # [1,1],..[-1,-1]
    
    for JDX in range(4):
        plt.plot(grid[IDX][0] + shifts[JDX][0], grid[IDX][1] + shifts[JDX][1], 
                 marker=markers[JDX], color=colors[JDX], alpha=alphas[JDX], markersize=18)
        plt.text(grid[IDX][0] + shifts[JDX][0], grid[IDX][1] + shifts[JDX][1], sub_labels[JDX], ha="center", va='center', family='sans-serif', size=12)
        label(grid[IDX], abbv)

def color_labels(xy,results, abbv):
    # xy gives central point
    # results = [[T,F], ... ]
    # first = greater / lesser (red / green)
    # second = significant / not (filled / empty)
    # ORDER = SATmax, P5, SAT, P-E
    markers = ['s','s','s','s']
    sub_labels = ['Tx','Px','T','PE']
    
    colors = [blue if x[0] else red for x in results] # read first element of each result to determine color
    alphas = [1.0 if x[1] else 0.3 for x in results] # read second element of each result to determine fill
    
    dxy = 0.03 # Shift points off centre by this amount
    shifts = dxy * np.mgrid[1:-1:-2j, 1:-1:-2j].reshape(2, -1).T # [1,1],..[-1,-1]
    
    for JDX in range(4):
        plt.plot(grid[IDX][0] + shifts[JDX][0], grid[IDX][1] + shifts[JDX][1], 
                 marker='o', color='w', alpha=0.5, markersize=18)
        plt.text(grid[IDX][0] + shifts[JDX][0], grid[IDX][1] + shifts[JDX][1], sub_labels[JDX], weight='bold', color=colors[JDX], alpha=alphas[JDX], ha="center", va='center', family='sans-serif', size=12)
        label(grid[IDX], abbv)
        
fig, ax = plt.subplots()
# create 6x6 grid to plot the artists

grid = np.mgrid[0.1:0.9:6j, 0.1:0.9:6j].reshape(2, -1).T

patches = []
lines = []

for IDX in range(26):
    
    abbv = df.loc[IDX][0]
    
    pe = ((abs(df.loc[IDX][1]) > abs(df.loc[IDX][2])), (df.loc[IDX][3] > 0.5))
    precip = ((abs(df.loc[IDX][4]) > abs(df.loc[IDX][5])), (df.loc[IDX][6] > 0.5))
    precip5max = ((abs(df.loc[IDX][7]) > abs(df.loc[IDX][8])), (df.loc[IDX][9] > 0.5))
    tas = ((abs(df.loc[IDX][10]) > abs(df.loc[IDX][11])), (df.loc[IDX][12] > 0.5))
    tasmax = ((abs(df.loc[IDX][13]) > abs(df.loc[IDX][14])), (df.loc[IDX][15] > 0.5))
    
    results = [tasmax,precip5max,tas,pe]
    color_labels(grid[IDX],results, abbv)
#     labeled_boxes_4(grid[IDX],results, abbv)
#     markers_4(grid[IDX],results, abbv)


plt.subplots_adjust(left=0.05, right=1, bottom=0.05, top=1)
plt.axis('equal')
plt.axis('off')

# plt.savefig(out_dir+'fig4_labels.png', format='png', dpi=480)
# plt.savefig(out_dir+'fig4_labels.eps', format='eps', dpi=480)

plt.savefig(out_dir+'fig4_text.png', format='png', dpi=480)
plt.savefig(out_dir+'fig4_text.eps', format='eps', dpi=480)

plt.show()