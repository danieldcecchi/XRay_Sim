#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:11:54 2019

@author: danieldcecchi
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

file = pd.read_csv('/Users/danieldcecchi/2019fallresearch/xray_sim/src/total17M034.csv')

total = file.values
total = np.asarray(total)
dosevalues = []
for i in range(len(total)):
    if i%6 == 4:
        dosevalues.append(total[i])
dose2 = []
for i in dosevalues:
    i = i.tolist()
    i = np.squeeze(i).astype(float)
    dose2.append(i)

dose = np.asarray(dose2)    
    
deg = np.arange(75,105.5,0.5)

plt.ylabel('Dose To Scint')
plt.xlabel('Degree of Rotation')

plt.plot(deg,dose)
plt.show()