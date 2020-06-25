#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:46:15 2020

@author: danieldcecchi
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator
from textwrap import wrap

'''The many CSV Files that I have used for plotting and analyzing data'''

csv_file = '/Users/danieldcecchi/2019fallresearch/dose_measurements/01_14_20/080kVp_30s_incol.csv'
csv_file1 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/01_14_20/100kVp_30s_incol.csv'
csv_file2 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/01_14_20/120kVp_30s_incol.csv'
csv_file3 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_18_20/113725Dose1min024.csv'
csv_file4 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_18_20/114020Dose1min024.csv'
csv_file5 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_18_20/80kV_5min_30s.csv'
csv_file6 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_18_20/114935Dose1min038.csv'
csv_file7 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_18_20/115539Dose1min038.csv'
csv_file8 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_18_20/115740Dose1min038.csv'
csv_file9 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/104413Dose1min024.csv'
csv_file10 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/105307Dose1min024.csv'
csv_file11 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/110235Dose1min024.csv'
csv_file12= '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/110652Dose1min024.csv'
csv_file13= '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/111102Dose1min024.csv'
csv_file14= '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/111712Dose1min024.csv'
csv_file15= '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/111923Dose1min024.csv'
csv_file16= '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/112134Dose1min024.csv'
csv_file17= '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/112343Dose1min024.csv'
csv_file18= '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/112551Dose1min024.csv'
csv_file19= '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/114111Dose5min024.csv'
csv_file20= '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/122242Dose1min038.csv'
csv_file21= '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/122451Dose1min038.csv'
csv_file22 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/122714Dose1min038.csv'
csv_file23 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/122910Dose1min038.csv'
csv_file24 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/123239Dose1min038.csv'
csv_file25 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/124759Dose5min038.csv'
csv_file26 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/125436Dose1min038.csv'
csv_file27 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/125732Dose1min031.csv'
csv_file28 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/125943Dose1min031.csv'
csv_file29 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/130202Dose1min031.csv'
csv_file30 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/130403Dose1min031.csv'
csv_file31 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/130621Dose1min031.csv'
csv_file32 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_19_20/132119Dose5min031.csv'
csv_file33 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/12_19_19/080kVp_1-30mA_30s.csv'
csv_file34 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/12_19_19/100kVp_1-30mA_30s.csv'
csv_file35 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/12_19_19/120kVp_1-30mA_30s.csv'
csv_file36 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_17_20/120kVp1_5min_30s.csv'
csv_file37 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_17_20/120kVp2_5min(2)_30s.csv'
csv_file38 = '/Users/danieldcecchi/2019fallresearch/dose_measurements/02_17_20/120kVp3_5min(3)_30s.csv'
def slope_analysis(csv_file):
    """This function compares the slope while the probe is getting irradiate
    and how it changes from experiment to experiment"""

    slopes = []
    
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
    peaklocs, a = find_peaks(der,height = 15000,distance = 5)
    firstpeak = peaklocs[::2]


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
    
    slope_dose_vals = []
    time_vals = []
    
    for i in avd_time_locs:
        dose = []
        time = []
        dose.append(DoseRate[i-4 : i+4])
        time.append(time2[i-4: i+4])
        time_vals.append(time)
        slope_dose_vals.append(dose)

      
    
    

    
    for i in slope_dose_vals:
        a = linregress(time,i)
        slopes.append(a[0])
    
    
        
    tube_current = np.array([4,5,7.5,10,15,20,25,30])
    plt.plot(tube_current, slopes)
    z = np.polyfit(tube_current,slopes,1)
    
    degrees = [1]
    matrix = np.stack([tube_current**d for d in degrees], axis=-1)
    coeff = np.linalg.lstsq(matrix, slopes,rcond=None)[0]
    
    fit = np.dot(matrix, coeff)
    #plt.plot(tube_current,fit,'g-',label = f'Forced zero Y-intercept: y = {np.round(coeff,2)}x')
    
    best_fit_line = best_fit(tube_current,z[0],z[1])
    
    plt.plot(tube_current,best_fit_line,'b--',label = f"Best Fit Line: y = {round(z[0],2)}x + {round(z[1],2)}")
    plt.xlabel('Tube Current')
    plt.ylabel('Slope of Peak [cGy/min]')
    plt.title('120kVp 024 1 min Irradiation: rest 5 min')
    plt.legend()
    
#slope_analysis(csv_file2)
    
def slope_analysis_compare(csv_file1, csv_file2, csv_file3,csv_file4, csv_file5):
    """This function compares the slope while the probe is getting irradiate
    and how it changes from irradiation to irradiation"""
    csv_files = [csv_file1,csv_file2,csv_file3,csv_file4,csv_file5]
    slopes = []
    for i in csv_files:
        results = np.loadtxt(i, usecols = [0,1],skiprows = 1)
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
        peaklocs, a = find_peaks(der,height = 20000,distance = 5)
        firstpeak = peaklocs[::2]

    
        '''Use this following plot when switching between files to check if you have the right height 
        in peaklocs.'''
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
        
        slope_dose_vals = []
        time_vals = []
        
        for i in avd_time_locs:
            dose = []
            time = []
            dose.append(DoseRate[i-4 : i+4])
            time.append(time2[i-4: i+4])
            time_vals.append(time)
            slope_dose_vals.append(dose)
    
          
        
        
    
        
        for i in slope_dose_vals:
            a = linregress(time,i)
            slopes.append(a[0])
    
    
        
    irr_val = np.linspace(0,len(csv_files),len(csv_files))  
    plt.plot(irr_val, slopes)
    z = np.polyfit(irr_val,slopes,1)
    
    degrees = [1]
    matrix = np.stack([irr_val**d for d in degrees], axis=-1)
    coeff = np.linalg.lstsq(matrix, slopes,rcond=None)[0]
    
    fit = np.dot(matrix, coeff)
    #plt.plot(tube_current,fit,'g-',label = f'Forced zero Y-intercept: y = {np.round(coeff,2)}x')
    
    #best_fit_line = best_fit(irr_val,z[0],z[1])
    
    #plt.plot(irr_val,best_fit_line,'b--',label = f"Best Fit Line: y = {round(z[0],2)}x + {round(z[1],2)}")
    plt.xlabel('Irradiation Index')
    plt.ylabel('Slope of Peak [cGy/min]')
    plt.title('031 1 min Irradiation: rest 1 min')
    plt.legend()
    plt.show()
    
    
#slope_analysis_compare(csv_file27,csv_file28,csv_file29,csv_file30,csv_file31)
    
def compare_analysis(csv_file1, csv_file2,csv_file3):
    """This function compares irradiation dose rates from one experiment to the other when doing
    multiple exposures with varying tube current. """
    csv_files = [csv_file1,csv_file2,csv_file3]
    
    for p in csv_files:
        results = np.loadtxt(p, usecols = [0,1],skiprows = 1)
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
        peaklocs, a = find_peaks(der,height = 25000,distance = 5)
    
        '''Use this following plot when switching between files to check if you
        #are doing it right.'''
        #plt.plot(time2[peaklocs],der[peaklocs],'x',time,DoseRate,'b',time2,der,'g')

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
        
        slope_dose_vals = []
        time_vals = []
        
        for i in avd_time_locs:
            dose = []
            time = []
            dose.append(DoseRate[i-18 : i+18])
            time.append(time2[i-18: i+18])
            time_vals.append(time)
            slope_dose_vals.append(dose)
    
        """Following for loop creates a dose and time list that has the same length as
        all other files"""
        
        dose = []
    
        for i in range(len(slope_dose_vals)):
            
            a = slope_dose_vals[i]
    
            a = a[0].tolist()
            dose.append(a)
            for k in range(30):
                dose.append(0)
                
        dose = np.hstack(dose)
        print(max(dose))
        dose = [x/3103137 for x in dose]
        title = p[65:71]
        
        t = np.linspace(0,len(dose)/20,len(dose), endpoint = True)
        plt.plot(t,dose,label = title)
        plt.legend(loc = 'best', fontsize = 12)
        #plt.title("\n".join(wrap('024 Inside Collimator, 5 min spaced Irradiations')))
        plt.xlabel('Tube Current [mA]', size = 22)
        #t = [t.min(), round(t.max()+0.3)]
        plt.xticks([1,4.4,7.6,10.9,14.2,17.5,20.8, 24.1, 27.4, 30.7],[1,2,3,4,5,10,15, 20, 25, 30])
        #plt.xticks([1,4.4,7.6,10.9,14.2,17.5,20.8],[4,5,7.5,10,15,20,25])
        #plt.yticks([0,0.2,0.4,0.6,0.8,1],[0.000,0.200,0.400,0.600,0.800,1.000])
        #plt.yticks.set_major_formatter(plt.ticker.mtick.FormatStrFormatter('%.3'))
        plt.ylabel('Scintillator Output [a.u.]', size = 19)
    plt.grid(True)
    plt.show()    
    
#compare_analysis(csv_file36, csv_file37,csv_file38)
#compare_analysis(csv_file33, csv_file34,csv_file35)

def compare_analysis2(csv_file1, csv_file2,csv_file3,csv_file4, csv_file5):
    '''This function compares single exposure from various experiments with different files'''
    csv_files = [csv_file1,csv_file2,csv_file3,csv_file4,csv_file5]
    k = 1
    for i in csv_files:
        results = np.loadtxt(i, usecols = [0,1],skiprows = 1)
        results = results.T
        time = results[0]
        time = time.tolist()
        DoseRate = results[1]
    
        #Calculates the Slope
        diffDR = np.diff(DoseRate)
        der = abs(diffDR/2)
        
        #New Time array that has a size equal to der
        #plot of the slope
    
        #Finds the max and minimum of the peaks
        peaklocs, a = find_peaks(der,height = 10000,distance = 5)
    
    
        '''Use this following plot when switching between files to check if you
        are doing it right.'''
        #plt.plot(time,DoseRate,'b')
    
        dose = []
        realtime = time[peaklocs[0]+3:peaklocs[1]-3]
        realtime = [x - realtime[0] for x in realtime]
        
        dose = DoseRate[peaklocs[0]+3:peaklocs[1]-3]
        m = max(dose)
        print(m)
        dose = [x/89151 for x in dose]
        title = i[65:82]
        plt.grid(True)
        plt.plot(realtime,dose,label = f"Irradiation {k}")
        #plt.title('038 1min Irradiation: rest 1min')
        plt.legend(loc = 'best', fontsize = 16)
        #plt.xticks([], fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.xlabel('Time [s]', size = 22)
        plt.ylabel('Scintillator Output [a.u.]', size = 19)
        k += 1
    plt.show()
   

#031 Irradiation
#compare_analysis2(csv_file27,csv_file28,csv_file29,csv_file30,csv_file31)
#038 Irradiation
#compare_analysis2(csv_file20,csv_file21,csv_file22,csv_file23,csv_file24)

def main_analysis(csv_file1, csv_file2, csv_file3, csv_file4, csv_file5):
    csv_files = [csv_file1, csv_file2, csv_file3, csv_file4, csv_file5]
    mean_DoseRate = np.array([])
    std_DoseRate = np.array([])
    for i in csv_files:
        results = np.loadtxt(i, usecols = [0,1],skiprows = 1)
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
        peaklocs, a = find_peaks(der,height = 10000,distance = 5)
        firstpeak = peaklocs[::2]

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
        for i in avd_time_locs:
            mean_DoseRate = np.append(mean_DoseRate, DoseRate[i-4 : i+4].mean())
            std_DoseRate = np.append(std_DoseRate, DoseRate[i-4:i+4].std())


    m = max(mean_DoseRate)
    s = max(std_DoseRate)
    mean_DoseRate = np.array([x/m for x in mean_DoseRate])
    std_DoseRate = np.array([x/m for x in std_DoseRate])
    '''plots average Dose Rate against tube current'''
    irr_index = np.array([1,2,3,4,5])

#    
#   plt.errorbar(tube_current,mean_DoseRate, yerr = std_DoseRate,capsize = 10,barsabove = True,fmt = 'none',label = 'Error')
    
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Irradiation', size = '22')
    ax.set_ylabel('Scintillator Output [a.u]', size = 19)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    #ax.set_title('DR vs Irr 038 1 min: rest 1min')
    plt.plot(irr_index,mean_DoseRate,'x',label = 'Data Points')
    
    z = np.polyfit(irr_index,mean_DoseRate,1)
    
    '''Next few lines take the data and fit a y = mx slope'''
    degrees = [1]
    matrix = np.stack([irr_index**d for d in degrees], axis=-1)
    coeff = np.linalg.lstsq(matrix, mean_DoseRate,rcond=None)[0]
    
    fit = np.dot(matrix, coeff)
    #plt.plot(irr_index,fit,'g-',label = f'Forced zero Y-intercept: y = {np.round(coeff,2)}x')
    
    best_fit_line = best_fit(irr_index,z[0],z[1])
    #plots the best fit line and error bars with values
    plt.errorbar(irr_index,mean_DoseRate, yerr = std_DoseRate,capsize = 10,barsabove = True,\
                 fmt = 'none',label = 'Error')
    #plt.plot(irr_index,best_fit_line,'b--',label = f"Best Fit Line: y = {round(z[0],4)}x + {round(z[1],2)}")
    
    #Fits an exponential to the data

    
    plt.grid(True)
    popt, pcov = curve_fit(func, irr_index, mean_DoseRate)
    plt.plot(irr_index, func(irr_index, *popt), 'g-',label='Exponential fit')
    plt.legend(fontsize = 11)
    plt.figure(figsize=(6,6))
    #average mean dose rate
    #avgMDR = mean_DoseRate.mean()
    
    

#    #calculates variance
#    fitdif_data = mean_DoseRate - fit
#    fitdif = 0
#    for i in fitdif_data:
#        fitdif = fitdif + i*i
#    
#    #calculates variance in Y
#    meandif_data = avgMDR - mean_DoseRate
#
#    meandif = 0
#    for i in meandif_data:
#        meandif = meandif + i*i
#        
#    #calculates the r squared value
#    r_squared = (meandif - fitdif)/meandif
#    print(f'R Squared Value Calculated: R^2 = {r_squared}')
    
    plt.show(block = True)
    
def best_fit(x,m,b):
    return m*x + b

def func(x, a, b, c):
    return a * np.exp(-b * x) + c


#024 Irradiation rest 3-6 min
#main_analysis(csv_file9,csv_file10,csv_file11,csv_file12,csv_file13)
#024 Irradiation
#main_analysis(csv_file14,csv_file15,csv_file16,csv_file17,csv_file18)
    
#038 Irradiation 
#main_analysis(csv_file20,csv_file21,csv_file22,csv_file23,csv_file24)

#031 Irradiation
#main_analysis(csv_file27,csv_file28,csv_file29,csv_file30,csv_file31)



def compare_analysis3(csv_file1, csv_file2,csv_file3):
    """This function compares irradiation dose rates from one experiment to the other when doing
    multiple exposures with varying tube current. """
    csv_files = [csv_file1,csv_file2,csv_file3]
    colours = ['blue','orange','green']
    k = 0
    for p in range(len(csv_files)):
        
        results = np.loadtxt(csv_files[p], usecols = [0,1],skiprows = 1)
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
        peaklocs, a = find_peaks(der,height = 25000,distance = 5)
        oddpeaks = []
        evenpeaks = []
        for i in range(len(peaklocs)):
            if i%2 == 0:
                oddpeaks.append(peaklocs[i])
            else:
                evenpeaks.append(peaklocs[i])
        '''Use this following plot when switching between files to check if you
        #are doing it right.'''
        #plt.plot(time2[evenpeaks],der[evenpeaks],'x',time,DoseRate,'b',time2,der,'g')
        
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
        
        slope_dose_vals = []
        time_vals = []
        
        
        for i in avd_time_locs:
            dose = []
            time = []
            dose.append(DoseRate[i-18 : i+18])
            time.append(time2[i-18: i+18])
            time_vals.append(time)
            slope_dose_vals.append(dose)
    
        max_dose_values = []
        for i in oddpeaks:
            max_dose_values.append(DoseRate[i+4])
        min_dose_values = []
        for j in evenpeaks:
            min_dose_values.append(DoseRate[j-1])
    
        """Following for loop creates a dose and time list that has the same length as
        all other files"""
        
        dose = []
    
        for i in range(len(slope_dose_vals)):
            
            a = slope_dose_vals[i]
    
            a = a[0].tolist()
            dose.append(a)
            for k in range(30):
                dose.append(0)
                
        dose = np.hstack(dose)
        
        max_dose_values = [x/4253736 for x in max_dose_values]
        min_dose_values = [x/4253736 for x in min_dose_values]
        title = csv_files[p][65:72]
        #t = np.linspace(0,len(dose)/20,len(dose), endpoint = True)
        x = [i for i in range(7)]
        plt.grid(True)
        plt.plot(x,max_dose_values,'-',label = f"{title} Interval Max", color = colours[p])
        plt.plot(x,min_dose_values, '--',label = f"{title} Interval Min", color = colours[p])
        plt.legend(loc = 'best', fontsize = 12)
        #plt.title("\n".join(wrap('024 Inside Collimator, 5 min spaced Irradiations')))
        plt.xlabel('Tube Current [mA]', size = 22)
        #t = [t.min(), round(t.max()+0.3)]
        plt.xticks([0,1,2,3,4,5,6],[4,5,7.5,10,15,20,25])
        plt.ylabel('Scintillator Output [a.u.]', size = 19)
        k += 1 
    plt.show()
#compare_analysis3(csv_file36, csv_file37, csv_file38)



    