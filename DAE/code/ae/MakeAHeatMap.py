import matplotlib
import matplotlib.pyplot as plt
import numpy as np

a = [[33.81, 36.17, 34.88, 37.5, 45.5],
     [26.27, 30.25, 33.42, 32.48, 34],
     [22.69, 25.64, 28.96, 33.42, 33.08],
     [21.02, 25.98, 24.9, 30.31, 34.52],
     [18.67, 22.83, 25.96, 32.31, 33.13],
     [18.53, 22.13, 26.78, 30.68, 29.78]]

plt.imshow(a, cmap='hot', interpolation='none')#, vmin=0, vmax=1, aspect='equal')

cbar = plt.colorbar()
cbar.set_label('Validation error (mm)',size=28)
# access to cbar tick labels:
cbar.ax.tick_params(labelsize=25)

ax1 = plt.gca();
plt.xlabel('Depth', fontsize=28)
plt.ylabel('Width', fontsize=28)
ax1.set_xticklabels(['Depth', 1,2,3,4,5], {'fontsize': 30})
ax1.set_yticklabels(['Width', 32,64,128,256,512,1024], {'fontsize': 30})

ax1.title.set_fontsize(2000)

plt.show()