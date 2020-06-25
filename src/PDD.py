#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:37:51 2020

@author: danieldcecchi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
from statistics import mean
#%matplotlib inline

fig, ax = plt.subplots()

#Reads both the topas and EGS files to convert to 3D array
topas1 = pd.read_csv("/Users/danieldcecchi/2019fallresearch/xray_sim/src/2DDose45Deg.csv",skiprows = 7)
#topas2 = pd.read_csv("5WaterDOSXYZ.csv",skiprows = 7)
#raw_data_dose = open("../DataFiles/cubeCo5x5.txt")
#lines =  raw_data_dose.readlines()
a = []
#b = np.zeros(len(lines), dtype = 'object')
j = 0
#for line in lines:
    #a = [float(i) for i in line.split()]

    #b[j] = a
    #j+=1


#TOPAS Values
#doses = topas1.values
#max_dose_topas = np.mean(doses)
#print(max_dose_topas)
#doses = [d for d in doses]
#doses = np.asarray(doses)
#BEAM = doses.reshape(20,20,1)
#BEAM = np.rot90(BEAM,2,axes = (0,2))
#maxB = np.amax(BEAM)

#diff = 100 - np.divide(EGS,TOPAS)

# plt.figure(1)
# i = 0
# l = plt.imshow(BEAM[i],cmap=plt.cm.BuPu_r)


# axcolor = 'lightgoldenrodyellow'
# ax.set_title("PDD BEAM Mean Energy")


# axind = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
# sfreq = Slider(axind, 'Index', 1, 19, valinit=i, valstep=1)

# cax = plt.axes([0.82, 0.1, 0.075, 0.8])
# cbar = fig.colorbar(l,cax=cax)
# cbar.set_label('Relative Intensity')

# def update(val):
#     frame = np.around(sfreq.val)
#     l.set_data(BEAM[int(frame)])

# sfreq.on_changed(update)






doses = topas1.values
max_dose_topas = np.mean(doses)
#print(max_dose_topas)
#doses = [d for d in doses]
doses = np.asarray(doses)
DOSXYZ = doses.reshape(50,50,50)
DOSXYZ = np.rot90(DOSXYZ,1,axes=(0,2))
maxD = np.amax(DOSXYZ)


#plt.figure(2)
j = 0
k = plt.imshow(DOSXYZ[j],cmap=plt.cm.BuPu_r)


axcolor = 'lightgoldenrodyellow'
ax.set_title("2D Film 3 -45 Deg")


axind = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
freq = Slider(axind, 'Index', 1, 19, valinit=j, valstep=1)

cax = plt.axes([0.82, 0.1, 0.075, 0.8])
cbar = fig.colorbar(k,cax=cax)
cbar.set_label('Relative Intensity')

def update(val):
    frame = np.around(freq.val)
    k.set_data(DOSXYZ[int(frame)])

freq.on_changed(update)

# avgd = []



# for i in range(50):
#     avgd.append((DOSXYZ[i][24]))


# dist = [x for x in range(50)]
# plt.plot(dist, avgd)
# plt.title("BEAM Energy across Middle Row 24th Slice")
# plt.xlabel('Distance Across Slices [cm]')
# plt.gca().set_xticklabels(['']*10)
# plt.ylabel('Dose [Gy/Particle]')
# plt.gca().invert_xaxis()

plt.show()

