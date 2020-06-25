#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:23:06 2019

@author: danieldcecchi
"""

from tkinter import *
from tkinter import ttk
from tkinter.ttk import Frame
import time as t
from datetime import *
import os
from shutter import Shutter
import threading
from datetime import datetime




'''Directory is where the diagnostics page saves its information once you quit the program.
You can change it to your own. '''
directory = '/Users/danieldcecchi/2019Fallresearch/Sample_Diagnostics'



'''Boolean values for button presses.'''
STATUS = True 
QBPress = False
FPress = False
HPress = False
SPress = False
EPress = False


'''Grabs the current time without the month or the year'''
now = datetime.now()

'''Grabs the date and write it to the diagnostics text file'''
today = date.today()
output = f"{today}" + ".txt"
path = os.path.join(directory, output)
with open(path, 'w') as f:
    f.write('Diagnostic Data: ' + "\n")







class Application(Tk):
    
    '''This functions makes the GUI that controls the shutter for the COMET xray tube at the
    University of Victoria'''


    def __init__(self, shutterclass):
        super(Application, self).__init__()
        
        '''The next several lines creates the user interface that you see when you start the program. 
        It creates two pages, one for the commands and one for the print screen that shows what commands 
        have already been executed and with what parameters. The screen will also show any error
        messages that may arise.'''
        self.nb = ttk.Notebook(self)
        self.nb.grid(row=1, column=0,rowspan=100,columnspan=500)
        self.nb.page1 = ttk.Frame(self.nb)
        self.nb.add(self.nb.page1, text='Controls')
        self.nb.page2 = ttk.Frame(self.nb)
        self.nb.add(self.nb.page2, text = 'Diagnostics')

        self.T = Text(self.nb.page2, height=62, width=62, bg = 'grey', fg = 'blue')
        self.T.pack(fill = X)

        self.connect()

        self.title('Shutter GUI')
        self.nb.pack(expan = 1, fill = 'both')
        self.geometry("700x500")

        '''Defines the Choices Possible for CP and ET
        At the Moment they do not add any function.  But I was thinking that maybe
        we could use them for later testing to set more specific ET and CP.'''
        #timechoices = {'min', 's', 'ms', 'micros'}
        #comchoice = [i for i in range(10)]
        #self.exptime = StringVar(self)
        #self.exptime.set('ms')
        self.comport = StringVar(self)
        self.comport.set('CP')
    
        
        '''Creates entry box for exposure time'''
        self.entry = Entry(self.nb.page1)
        self.entry.place(relx = 0.5,rely = 0.07,anchor = CENTER)
        
        
        '''Inserts Title for ET and Commands'''
        label = ttk.Label(self.nb.page1, text = 'Exposure Time')
        label.place(relx = 0.5,rely = 0.015,anchor = CENTER)

        label3 = ttk.Label(self.nb.page1, text = 'Commands',font = ("Helvetica",24))
        label3.place(relx = 0.5,rely = 0.15,anchor = CENTER)
        
        '''Changes Dropdown menus that aren't being used right now. '''
        #self.exptime.trace('w', self.change_dropdown)
        #self.comport.trace('w',self.change_dropdown)


        '''Now comes the Buttons for the commands for Process().'''
        
    
        '''Button for Exposure Time'''
        et = Button(self.nb.page1,text='Enter ET',command= self.exposuretime, \
               activeforeground = 'Yellow',highlightthickness = 0,relief = 'ridge', bd = 0, \
                   width = 10, height = 3)
               
        et.place(relx = 0,rely = 0.25, anchor = W)
        et.config(font = ("Courier",15))
        
        '''Button for FIRE'''
        fire = Button(self.nb.page1,text='FIRE', command = self.fire, activeforeground = 'Yellow', \
               highlightthickness = 0, relief = 'ridge',width = 10, height = 3)
        fire.place(relx = 0.5,rely = 0.25, anchor = CENTER)
        fire.config(font = ("Courier",15))


        '''Stop Button'''
        stopbutton = Button(self.nb.page1,text = 'Stop', command = self.stop, \
            height = 2, width = 15)
        stopbutton.place(relx = 0.5,rely = 0.75, anchor=CENTER)
        stopbutton.config(font = ("Courier",15),state=DISABLED)
        
        
        '''Button for Comport Number that is not being used right now'''
#        Button(self.nb.page1, text = 'Enter CP#', command = self.comportnumber, \
#               activeforeground = 'Yellow',highlightthickness = 0).place(relx = 0.2, rely = \
#                                                                 0.4)


        '''Button to HOME Shutter'''
        home = Button(self.nb.page1,text = 'HOME',command = self.home, activeforeground = 'Yellow', \
               highlightthickness = 0, relief = 'ridge',width = 10, height = 3)
        home.place(relx = 1,rely = 0.25, anchor = E)
        home.config(font = ("Courier",15))
        
        
        '''Button to Quit program'''
        quitButton = Button(self.nb.page1, text="Quit",command=self.function_end, \
                            height = 3, width = 20)
        quitButton.place(relx = 0.5, rely = 0.9, anchor = CENTER)
        quitButton.config(font = ("Courier",20))
        
        self.protocol("WM_DELETE_WINDOW", self.function_end)
        
        '''Threading allows the class to call another class Shutter in shutter_skelly periodically
        using the run function below. This runs in the background while the user can use all the buttons
        themself. '''
        thread = threading.Thread(target=self.run,args=())
        thread.daemon = True 
        thread.start()
        
    def run(self):
        while True: 
            if f"{self.shutterclass.Process()}" != "None":
                self.T.insert(END,"\n" + f"{self.shutterclass.Process()}")
            t.sleep(0.25)
        
        
        
    def home(self):
        HPress = True
        self.isclicked(FPress,HPress,QBPress,SPress,EPress)

    def connect(self):
        '''Prints out first status message when program connects'''
        self.shutterclass = shutterclass
        self.T.insert("1.0", f"{self.shutterclass.GetStatusMsg()}")


    def fire(self):
        FPress = True
        self.isclicked(FPress,HPress,QBPress,SPress,EPress)

    def exposuretime(self):
        EPress = True
        self.isclicked(FPress,HPress,QBPress,SPress,EPress)
        # if self.entry.get() != '':
        #     self.shutterclass.Process(command = 'TEXP', parameter = int(self.entry.get()))
        #     self.T.insert(END,"\n" + f"{self.shutterclass.GetStatusMsg()}")

    #def change_dropdown(self, *args):

        """Lets you set the exposure time and change comport number"""

        #print( self.exptime.get() )
        #print(self.comport.get())


    def stop(self):
        SPress = True
        self.isclicked(FPress,HPress,QBPress,SPress,EPress)
    
    def function_end(self):

        """called when the user quits the program"""
        QBPress = True
        self.isclicked(FPress,HPress,QBPress,SPress,EPress)


    def time_of_year(self):
        """Finds the time of year and day that 
        the program is operating at and sets
        certain functions because of it"""
        
    def time_of_day(self):
        '''Finds the time of day to show special messages'''

        
    def isclicked(self,FPress,HPress,QBPress,SPress,EPress):
        '''Function to tell if a button is pressed.'''
        if FPress == True:
            
            self.shutterclass.Process(command = 'FIRE')
            self.T.insert(END,"\n" + now.strftime("%H:%M:%S"))
            self.T.insert(END, "\n" + f"{self.shutterclass.GetStatusMsg()}")
            self.stopbutton['state'] = NORMAL
        
        elif QBPress == True:
            self.T.insert(END,"\n" + now.strftime("%H:%M:%S"))
            #if messagebox.askokcancel("Quit", "You're Done? *sniff* "):
                #input = self.T.get("1.0", END)
                #with open(path, 'a') as f:
                    #f.write(input)
            self.destroy()
        elif HPress == True:
            self.T.insert(END,"\n" + now.strftime("%H:%M:%S"))
            self.shutterclass.Process(command = 'HOME')
            self.T.insert(END,"\n" + f"{self.shutterclass.GetStatusMsg()}")
            
        elif SPress == True:
            self.T.insert(END,"\n" + now.strftime("%H:%M:%S"))
            self.shutterclass.Process(command = 'STOP')
            self.T.insert(END,'\n' + f'{self.shutterclass.GetStatusMsg()}')   
        elif EPress == True:
            self.T.insert(END,"\n" + now.strftime("%H:%M:%S"))
            if self.entry.get() != '':
                self.shutterclass.Process(command = 'TEXP', parameter = float(self.entry.get()))
                self.T.insert(END,"\n" + f"{self.shutterclass.GetStatusMsg()}")
              
    

class PopUp(Tk):

    def __init__(self):
        super(PopUp, self).__init__()
        
        '''Extra window that pops up when you first load the GUI. '''
        self.nb1 = ttk.Notebook(self)
        self.nb1.page = ttk.Frame(self.nb1)

        self.title('WELCOME')
        self.nb1.pack(expan = 1, fill = 'both')
        self.geometry("200x200")

        label3 = ttk.Label(self.nb1.page, text = 'Commands',font = ("Helvetica",24))
        label3.place(relx = 0.5,rely = 0.15,anchor = CENTER)


if __name__ == '__main__':

    #By default, set port to 1 just because.
    port = '1'
    shutterclass = Shutter(port)
    #window = PopUp()
    #window.mainloop()
    app = Application(shutterclass)

    app.mainloop()




