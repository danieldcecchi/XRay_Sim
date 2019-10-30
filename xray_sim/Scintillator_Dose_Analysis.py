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
    directory = '/Users/danieldcecchi/code/xray_sim/DataFiles/'
    raw_data_scint = np.loadtxt(directory + 'scintillator.txt', dtype = 'str',skiprows = 12,max_rows = 420,usecols = 0)
    input_files = []
    emax = int(sys.argv[3])
    emin = int(sys.argv[2])
    steps = int(sys.argv[4])
    nrgrange = [i for i in range(emin,emax+steps,steps)]
    #puts all the files into a list

    for i in range(emin,emax+steps,steps):
        for filename in os.listdir(directory):
            if f'{i}keV' in filename:
                input_files.append(directory + filename)
            else:
                continue
        emin+=steps
    m = int(len(input_files)/3)
    n = int(len(nrgrange))
    data_files = [[0] * m for i in range(n)]
    nrg_index = 0
    dose_at_d = [[0] * m for i in range(n)]
    #filters the list so that it creates a new list for each specific energy
    for j in range(emin,emax + steps,steps):
        d_index = 0
        for i in range(len(input_files)):
            if f'{j}' in input_files[i]:
                data_files[nrg_index][d_index] = input_files[i]
                d_index+=1
            else:
                continue
        nrg_index +=1
    dose_index=0
    max_distance = 10
    for i in range(int(n)):
    
        data_files[i].sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for j in range(max_distance):
            dose = scintillator(raw_data_scint,data_files[i][j])
            dose_at_d[i][j] = dose

    distances = [i for i in range(1,max_distance + 1)]
    for i in range(n):
        plt.plot(distances,dose_at_d[i],label=f'{nrgrange[i]}keV')
    # plt.plot(distances, dose_at_d[0],label='80keV')
    # plt.plot(distances,dose_at_d[1],label='100keV')
    # plt.plot(distances,dose_at_d[2],label='120keV')
    plt.xlabel('Distance From X-Ray Source [cm]')
    plt.ylabel('Measured Dose [units]')
    plt.title('Dosage vs. Distance')
    plt.legend(loc='best')
    plt.show()













