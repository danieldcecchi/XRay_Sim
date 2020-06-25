#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:39:38 2019

@author: danieldcecchi
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


#Based on your data's noise, you may need to increase or decrease the mimimum
#size of your peaks.  This is found in line 38.  You will also need to change
#the array of your tube currents if you change it from the usual as seen in
#line 71

csv_file = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_03_20/120kV_5min_30s.csv'
csv_file2 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_18_20/114935_Dose1min038.csv'
csv_file3 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_18_20/115539_Dose1min038.csv'
csv_file4 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_18_20/115740_Dose1min038.csv'

def main_analysis(csv_file):
    results = np.loadtxt(csv_file, usecols = [0,1],skiprows = 1)
    results = results.T
    time = results[0]
    DoseRate = results[1]
    
    #Calculates the Slope
    diffDR = np.diff(DoseRate)
    der = abs(diffDR/2)
    
    #New Time array that has a size equal to der
    time2 = (time[:-1] + time[1:])/2
    #plot of the slope

    #Finds the max and minimum of the peaks
    peaklocs, a = find_peaks(der,height = 18000,distance = 5)
    firstpeak = peaklocs[::2]
    print(firstpeak)
    '''Use this following plot when switching between files to check if you
    #are doing it right.'''
    #plt.plot(time2[peaklocs],der[peaklocs],'x',time,DoseRate,'b',time2,der,'g')
    #plt.plot(time2[firstpeak + 3], DoseRate[firstpeak + 3],'o')
    '''Grabs the average distance between the two times of the peaks '''
    avd = []
    i = 0
    while i < len(peaklocs):
        avd.append(int(abs((time2[peaklocs[i]]-1) + (time2[peaklocs[i+1]]-1))/2))
        i+= 2
    '''Since the array of time is even numbers, this makes any odd number even'''
    avd = np.array(avd)
    avd[avd%2==1] -= 1

    avd_time_locs = []
    for i in avd:
        a = np.where(time == i)[0][0]
        avd_time_locs.append(a)
    avd_time_locs = np.array(avd_time_locs)
    '''calculates the mean of the DoseRate data between bounds.
    4 to the right and left was chosen arbitarily. 
    Also calculates the standard deviation in the values'''
    mean_DoseRate = np.array([])
    std_DoseRate = np.array([])
    for i in avd_time_locs:
        mean_DoseRate = np.append(mean_DoseRate, DoseRate[i-4 : i+4].mean())
        std_DoseRate = np.append(std_DoseRate, DoseRate[i-4:i+4].std())

    
    '''plots average Dose Rate against tube current'''
    tube_current = np.array([4,5,7.5,10,15,20,25])
    plt.title('Dose Rate vs Tube Current')
#    
    plt.errorbar(tube_current,mean_DoseRate, yerr = std_DoseRate,capsize = 10,barsabove = True,fmt = 'none',label = 'Error')
    plt.xlabel('Tube Current [mA]')
    plt.ylabel('Average Dose Rate from Raw Data [Gy/s]')
    plt.plot(tube_current,mean_DoseRate,'x',label = 'Data Points')
    
    z = np.polyfit(tube_current,mean_DoseRate,1)
    
    '''Next few lines take the data and fit a y = mx slope'''
    degrees = [1]
    matrix = np.stack([tube_current**d for d in degrees], axis=-1)
    coeff = np.linalg.lstsq(matrix, mean_DoseRate,rcond=None)[0]
    
    fit = np.dot(matrix, coeff)
    plt.plot(tube_current,fit,'g-',label = f'Forced zero Y-intercept: y = {np.round(coeff,2)}x')
    
    best_fit_line = best_fit(tube_current,z[0],z[1])
    #plots the best fit line
    plt.plot(tube_current,best_fit_line,'b--',label = f"Best Fit Line: y = {round(z[0],2)}x + {round(z[1],2)}")
    plt.legend()
    
    #average mean dose rate
    avgMDR = mean_DoseRate.mean()


    #calculates variance
    fitdif_data = mean_DoseRate - fit
    fitdif = 0
    for i in fitdif_data:
        fitdif = fitdif + i*i
    
    #calculates variance in Y
    meandif_data = avgMDR - mean_DoseRate

    meandif = 0
    for i in meandif_data:
        meandif = meandif + i*i
        
    #calculates the r squared value
    r_squared = (meandif - fitdif)/meandif
    print(f'R Squared Value Calculated: R^2 = {r_squared}')
    plt.show(block = True)
def best_fit(x,m,b):
    return m*x + b

main_analysis(csv_file)

def second_analysis(csv_file):
    '''This function grabs the max dose of the 5mA and above peaks to append
    to the list of the average dose from 1-4mA'''
    #def function(csv_file):
    results = np.loadtxt(csv_file, usecols = [0,1],skiprows = 1)
    results = results.T
    time = results[0]
    DoseRate = results[1]
    
    #Calculates the Slope
    diffDR = np.diff(DoseRate)
    der = abs(diffDR/2)
    
    #New Time array that has a size equal to der
    time2 = (time[:-1] + time[1:])/2
    #plot of the slope

    #Finds the max and minimum of the peaks
    '''This part grabs the 5mA and above highest values'''
    peaklocs, a = find_peaks(der,height = 30000,distance = 5)
    firstpeak = peaklocs[::2]
    firstpeak = firstpeak[3:]
    

    
    
    '''Use this following plot when switching between files to check if you
    #are doing it right.'''
    #plt.plot(time2[peaklocs],der[peaklocs],'x',time,DoseRate,'b',time2,der,'g')
    #plt.plot(time2[firstpeak + 3], DoseRate[firstpeak + 3],'o')
    '''Grabs the average distance between the two times of the peaks '''
    avd = []
    i = 0
    while i < len(peaklocs):
        avd.append(int(abs((time2[peaklocs[i]]-1) + (time2[peaklocs[i+1]]-1))/2))
        i+= 2
    '''Since the array of time is even numbers, this makes any odd number even'''
    avd = np.array(avd)
    avd[avd%2==1] -= 1

    avd_time_locs = []
    for i in avd:
        a = np.where(time == i)[0][0]
        avd_time_locs.append(a)
    avd_time_locs = np.array(avd_time_locs)
    '''calculates the mean of the DoseRate data between bounds.
    4 to the right and left was chosen arbitarily. 
    Also calculates the standard deviation in the values'''
    mean_DoseRate = np.array([])
    std_DoseRate = np.array([])
    for i in avd_time_locs:
        mean_DoseRate = np.append(mean_DoseRate, DoseRate[i-4 : i+4].mean())
        std_DoseRate = np.append(std_DoseRate, DoseRate[i-4:i+4].std())

    mean_DoseRate = mean_DoseRate[0:3]
    avg_DoseRate = np.append(mean_DoseRate, DoseRate[firstpeak + 3])

    '''plots average Dose Rate against tube current'''
    tube_current = np.array([4,5,7.5,10,15,20,25])
    plt.title('Dose Rate vs Tube Current')
#    
#   plt.errorbar(tube_current,mean_DoseRate, yerr = std_DoseRate,capsize = 10,barsabove = True,fmt = 'none',label = 'Error')
    plt.xlabel('Tube Current [mA]')
    plt.ylabel('Average Dose Rate from Raw Data [cGy/s]')
    plt.plot(tube_current,avg_DoseRate,'x',label = 'Data Points')
    
    z = np.polyfit(tube_current,avg_DoseRate,1)
    
    '''Next few lines take the data and fit a y = mx slope'''
    degrees = [1]
    matrix = np.stack([tube_current**d for d in degrees], axis=-1)
    coeff = np.linalg.lstsq(matrix, avg_DoseRate,rcond=None)[0]
    
    fit = np.dot(matrix, coeff)
    plt.plot(tube_current,fit,'g-',label = f'Forced zero Y-intercept: y = {np.round(coeff,2)}x')
    
    best_fit_line = best_fit(tube_current,z[0],z[1])
    #plots the best fit line
    plt.plot(tube_current,best_fit_line,'b--',label = f"Best Fit Line: y = {round(z[0],2)}x + {round(z[1],2)}")
    plt.legend()
    
    #average mean dose rate
    avgMDR = avg_DoseRate.mean()


    #calculates variance
    fitdif_data = avg_DoseRate - fit
    fitdif = 0
    for i in fitdif_data:
        fitdif = fitdif + i*i
    
    #calculates variance in Y
    meandif_data = avgMDR - avg_DoseRate

    meandif = 0
    for i in meandif_data:
        meandif = meandif + i*i
        
    #calculates the r squared value
    r_squared = (meandif - fitdif)/meandif
    print(f'R Squared Value Calculated: R^2 = {r_squared}')
    plt.show(block = True)

#second_analysis(csv_file)



