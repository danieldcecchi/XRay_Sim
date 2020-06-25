#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:43:04 2019

@author: danieldcecchi
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#pulls the raw data to create the scintillator


def scintillator(raw_data_scint,input_file):
    #makes the scintillator data into a 3D array.
    for row in raw_data_scint:
        r = np.array(list(row)).astype(int)
        try:
            matrix1 = np.vstack((r,matrix1))
        except NameError:
            matrix1 = r

    #Creates a 3D array of the scintillator
    n = 40
    n_matrices = int(raw_data_scint.shape[0] / n)
    scint = matrix1.reshape(n,n_matrices,n)
    scint = np.flip(scint,1)
    # #finds the location of all the '2's
    twolocs = np.argwhere(scint == 2)

    #[matrix,row,column]
    # #Grabs the dose data and puts into a 3 by 3 array.
    raw_data_dose = open(input_file)
    lines =  raw_data_dose.readlines()
    a = []
    b = np.zeros(len(lines), dtype = 'object')
    j = 0
    for line in lines:
        a = [float(i) for i in line.split()]
    
        b[j] = a
        j+=1

    #grabs dose and uncertainty data
    dose_column = np.array(b[4])
    unc_column = np.array(b[5])

    for i in range(len(dose_column)):
        unc_column[i] = unc_column[i]*dose_column[i]
    dose_data = dose_column.reshape(n,n_matrices,n)
    unc_data = unc_column.reshape(n,n_matrices,n)
    dose_scint = []
    unc_scint = []
    for i in range(len(twolocs)):
        dose_scint.append(dose_data[twolocs[i][0]][twolocs[i][1]][twolocs[i][2]])


    for i in range(len(twolocs)):
        unc_scint.append(unc_data[twolocs[i][0]][twolocs[i][1]][twolocs[i][2]])

    avgdose = np.mean(dose_scint)
    avgunc = np.mean(unc_scint)

    #print('The mean dose measured by the scintillator is: ' + str(avgdose))

    return avgdose, avgunc



if __name__ == "__main__":
    directory = '/Users/danieldcecchi/2019fallresearch/xray_sim/DataFiles/'
    raw_data_scint = np.loadtxt(directory + 'scintillator_new.txt', dtype = 'str', \
                                skiprows = 16,max_rows = 160,usecols = 0)
    input_files = []
    emax = 120
    emin = 80
    steps = 20
    nrgrange = [i for i in range(emin,emax+steps,steps)]
    #puts all the files into a list
    
    for i in range(emin,emax+steps,steps):
        for filename in os.listdir(directory):
            if f'{i}keV' in filename:
                input_files.append(directory + filename)
            else:
                continue
        emin+=steps

    m = int(len(input_files)/(3))
    n = int(len(nrgrange))
    data_files = [[0] * m for i in range(n)]
    nrg_index = 0
    dose_at_d = [[0] * m for i in range(n)]
    unc_at_d = [[0] * m for i in range(n)]
    #filters the list so that it creates a new list for each specific energy
    for j in range(80,140,20):
        d_index = 0
    
        for i in range(len(input_files)):
            if f'{j}' in input_files[i]:
                data_files[nrg_index][d_index] = input_files[i]
    
                d_index+=1
            else:
                continue
        nrg_index +=1
    
    dose_index=0
    max_distance = 1
    # max_distance = float(max_distance)
    # if max_distance < 1:
    #     max_distance = int(float(str(max_distance-int(max_distance))[-1:]))
    # else:
    #     max_distance = int(max_distance)

    for i in range(n):

        data_files[i].sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for j in range(max_distance):
            dose, uncertainty = scintillator(raw_data_scint,data_files[i][j])
            dose_at_d[i][j] = dose
            unc_at_d[i][j] = uncertainty
    dose_at_d = np.array(dose_at_d)
    unc_at_d = np.array(unc_at_d)/np.sqrt(316)
    distances = [i for i in range(1,max_distance + 1)]

    plt.plot(nrgrange,dose_at_d,'o',label='Energy')
    for i in range(len(dose_at_d)):
        plt.text(nrgrange[i],dose_at_d[i], f'{dose_at_d[i]}')
    
    plt.errorbar(nrgrange,dose_at_d, yerr = unc_at_d,capsize = 10,barsabove = True,fmt = 'none',label = 'Error')
    plt.xlabel('Energy [KeV]')
    plt.ylabel('Dose')
    plt.legend()
    plt.show()


    # un-comment here if you want a Dose vs Distance graph.
    # distances = [i for i in range(1,max_distance + 1)]
    # for i in range(n):
    #     plt.plot(distances,dose_at_d[i],label=f'{nrgrange[i]}keV',marker = '.')
    #     plt.errorbar(distances,dose_at_d[i], yerr = unc_at_d[i],capsize = 10,barsabove = True,fmt = 'none',label = 'Error')

    # plt.xlabel('Distance From X-Ray Source [cm]')
    # plt.ylabel('Measured Dose [units]')
    # plt.title('Dosage vs. Distance')
    # plt.legend(loc='best')
    # plt.show()








