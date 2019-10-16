#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:43:04 2019

@author: danieldcecchi
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#pulls the raw data to create the scintillator


def scintillator(raw_data_scint,input_file):
    #makes the scintillator data into a 3D array.
    for row in raw_data_scint:
        r = np.array(list(row)).astype(int)
        try:
            matrix1 = np.vstack((r,matrix1))
        except NameError:
            matrix1 = r
    
    n = 20
    n_matrices = int(raw_data_scint.shape[0] / n)
    scint = matrix1.reshape(n,n,n_matrices)
    
    #finds the location of all the '3's
    threelocs = np.argwhere(scint == 3)
    #[matrix,row,column]
    
    #Grabs the dose data and puts into a 3 by 3 array.
    raw_data_dose = open(input_file)
    lines =  raw_data_dose.readlines()
    a = []
    b = np.zeros(len(lines), dtype = 'object')
    j = 0
    for line in lines:
        a = [float(i) for i in line.split()]
    
        b[j] = a
        j+=1
    
    dose_column = np.array(b[4])
    
    dose_data = dose_column.reshape(n,n,n)
    
    dose_scint = []
    
    for i in range(len(threelocs)):
        dose_scint.append(dose_data[threelocs[i][0]][threelocs[i][1]][threelocs[i][2]])

    avgdose = np.mean(dose_scint)

    #print('The mean dose measured by the scintillator is: ' + str(avgdose))

    return avgdose





if __name__ == "__main__":
    #max_distance = sys.argv[1]
    directory = '../DataFiles/'
    raw_data_scint = np.loadtxt(directory + 'scintillator.txt', dtype = 'str',skiprows = 12,max_rows = 420,usecols = 0)
    input_files = []
    dose_at_d = []
    for filename in os.listdir(directory):
        if 'keV' in filename:
            input_files.append(directory + filename)
        else:
            continue
    input_files = sorted(input_files)
    for i in input_files:
        dose = scintillator(raw_data_scint, i)
        dose_at_d.append(dose)
    distances = [i for i in range(1,3)]
    plt.plot(distances, dose_at_d)
    plt.xlabel('Distance From X-Ray Source [cm]')
    plt.ylabel('Measured Dose [units]')
    plt.show()













