import numpy as np
import pandas as pd
import tkinter
import tkinter as tk
from tkinter import messagebox
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image  
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


class Project:
    def initial_gui(self):
        
        win =tk.Tk()
        win.title("Himanish")
        win.geometry('1200x703') 
        #win.minsize(400,400)
        #win.maxsize(1000,1000)
        win.config(background="green")
        image1 = Image.open("/Users/himanish/Documents/mustard.jpeg")
        test = ImageTk.PhotoImage(image1)

        label0 = tkinter.Label(image=test,bd=15)
        label0.image = test
       
        # Position image
        label0.place(x=10, y=3)
        lbl1= Label(win,text="Mustard Crop Yield Predictor",bg="blue",fg="white",height=3)
        lbl1.pack()

        lb2= Label(win,text="Enter Soil Parameters ",bg="black",fg="white",width=30,height=2)
        lb2.place(x=30,y=70)

        global ent1,ent2,ent3,ent4,ent5,ent6,ent7,ent8,ent9,ent10,ent11,ent14,ent15,ent16,ent17
        
       
        ent1=StringVar()
        ent2=StringVar()
        ent3=StringVar()
        ent4=StringVar()
        ent5=StringVar()
        ent6=StringVar()
        ent7=StringVar()
        ent8=StringVar()
        ent9=StringVar()
        ent10=StringVar()
        ent11=StringVar()
        ent14=StringVar()
        ent15=StringVar()
        ent16=StringVar()
        ent17=StringVar()
#
        lb3= Label(win,text="pH",bg="black",fg="white",width=2,height=1)
        lb3.place(x=40,y=120)
        entry=Entry(win,bg="pink",fg="black",textvariable= ent1)
        entry.place(x=75,y=120)
#2
        lb4= Label(win,text="EC",bg="black",fg="white",width=2,height=1)
        lb4.place(x=40,y=150)
        entry1=Entry(win,bg="white",fg="black",bd=1,textvariable=ent2)
        entry1.place(x=75,y=150)
#3
        lbl5= Label(win,text="OC",bg="black",fg="white",width=2,height=1)
        lbl5.place(x=40,y=180)
        entry3=Entry(win,bg="white",fg="black",bd=1,textvariable=ent3)
        entry3.place(x=75,y=180)
#4
        lbl6= Label(win,text="N",bg="black",fg="white",width=2,height=1)
        lbl6.place(x=40,y=210)
        entry4=Entry(win,bg="white",fg="black",bd=1,textvariable=ent4)
        entry4.place(x=75,y=210)
#5
        lbl7= Label(win,text="P",bg="black",fg="white",width=2,height=1)
        lbl7.place(x=40,y=240)
        entry5=Entry(win,bg="white",fg="black",bd=1,textvariable=ent5)
        entry5.place(x=75,y=240)
#6
        lbl8= Label(win,text="K",bg="black",fg="white",width=2,height=1)
        lbl8.place(x=40,y=270)
        entry6=Entry(win,bg="white",fg="black",bd=1,textvariable=ent6)
        entry6.place(x=75,y=270)
#7
        lbl9= Label(win,text="S",bg="black",fg="white",width=2,height=1)
        lbl9.place(x=40,y=300)
        entry7=Entry(win,bg="white",fg="black",bd=1,textvariable=ent7)
        entry7.place(x=75,y=300)
#8
        lbl10= Label(win,text="Zn",bg="black",fg="white",width=2,height=1)
        lbl10.place(x=40,y=330)
        entry8=Entry(win,bg="white",fg="black",bd=1,textvariable=ent8)
        entry8.place(x=75,y=330)
#9
        lbl11= Label(win,text="Fe",bg="black",fg="white",width=2,height=1)
        lbl11.place(x=40,y=360)
        entry9=Entry(win,bg="white",fg="black",bd=1,textvariable=ent9)
        entry9.place(x=75,y=360)
#10
        lbl12= Label(win,text="Cu",bg="black",fg="white",width=2,height=1)
        lbl12.place(x=40,y=390)
        entry10=Entry(win,bg="white",fg="black",bd=1,textvariable=ent10)
        entry10.place(x=75,y=390)
#11
        lbl13= Label(win,text="Mn",bg="black",fg="white",width=2,height=1)
        lbl13.place(x=40,y=420)
        entry11=Entry(win,bg="white",fg="black",bd=1,textvariable=ent11)
        entry11.place(x=75,y=420)
        
        lbl13= Label(win,text="PREDICTION BY ML CLASSIFIERS",bg="black",fg="white",width=50,height=2)
        lbl13.place(x=40,y=500)
        
        btn1=Button(win,text="K-Nearest Neighbour",bg="white",fg="black",command=self.knn_output)  
        btn1.place(x=40,y=550)
        entry14=Entry(win,bg="white",fg="black",textvariable=ent14)
        entry14.place(x=250,y=550)
        
        btn2=Button(win,text="Decision Tree",bg="white",fg="black",command=self.dt_output)  
        btn2.place(x=40,y=585)
        entry15=Entry(win,bg="white",fg="black",bd=1,textvariable=ent15)
        entry15.place(x=250,y=585)
        
        btn3=Button(win,text="Support Vector Machine",bg="white",fg="black",command=self.svc_output)  
        btn3.place(x=40,y=620)
        entry16=Entry(win,bg="pink",fg="black",bd=1,textvariable=ent16)
        entry16.place(x=250,y=620)
        
        btn4=Button(win,text="Reset",width=20,fg="black",bg="red",command=self.Reset)
        btn4.place(x=500,y=200)
        
        # btn4=Button(win,text="Final prediction",width=20,fg="black",bg="red",command=self.final_prediction)
        # btn4.place(x=500,y=400)
       
        
        # entry17=Entry(win,bg="pink",fg="black",bd=7,textvariable=ent17)
        # entry17.place(x=500,y=500)
        
        
        
        win.mainloop()


  
       
       
       
    def dt_output(self):
        
        
      

       pH = float(ent1.get())
       EC = float(ent2.get())
       OC = float(ent3.get())
       N = float(ent4.get())
       P = float(ent5.get())
       K = float(ent6.get())
       S = float(ent7.get())
       Zn = float(ent8.get())
       Fe = float(ent9.get())
       Cu = float(ent10.get())
       Mn = float(ent11.get())
       tupl= [ pH , EC , OC , N , P , K , S , Zn , Fe , Cu , Mn ]
       print(tupl)
       #crop=int(self..get())
       #print()
       filename="dt_dtmodel.sav"
       f = np.array(tupl)
       f=f.reshape(1,-1)
       loaded_model = pickle.load(open(filename, 'rb'))
       result = loaded_model.predict(f)
       print(result)
       if (result == [0]) :
           ent15.initialize("Low")
       elif (result == [1]) :
           ent15.initialize("Medium")
       else:
           ent15.initialize("High")

    def knn_output(self):
     

     pH = float(ent1.get())
     EC = float(ent2.get())
     OC = float(ent3.get())
     N = float(ent4.get())
     P = float(ent5.get())
     K = float(ent6.get())
     S = float(ent7.get())
     Zn = float(ent8.get())
     Fe = float(ent9.get())
     Cu = float(ent10.get())
     Mn = float(ent11.get())
     tupl= [ pH , EC , OC , N , P , K , S , Zn , Fe , Cu , Mn ]
     print(tupl)
     #crop=int(self..get())
     #print()
     filename="dt_model.sav"
     f = np.array(tupl)
     f=f.reshape(1,-1)
     loaded_model = pickle.load(open(filename, 'rb'))
     result = loaded_model.predict(f)
     print(result)
     if (result == [0]) :
         ent15.initialize("Low")
     elif (result == [1]) :
         ent14.initialize("Medium")
     else:
         ent14.initialize("High")

    def svc_output(self):
        
        pH = float(ent1.get())
        EC = float(ent2.get())
        OC = float(ent3.get())
        N = float(ent4.get())
        P = float(ent5.get())
        K = float(ent6.get())
        S = float(ent7.get())
        Zn = float(ent8.get())
        Fe = float(ent9.get())
        Cu = float(ent10.get())
        Mn = float(ent11.get())
        tupl= [ pH , EC , OC , N , P , K , S , Zn , Fe , Cu , Mn ]
        print(tupl)
        #crop=int(self..get())
        #print()
        filename="dt_svmmodel.sav"
        f = np.array(tupl)
        f=f.reshape(1,-1)
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.predict(f)
        print(result)
        if (result == [0]) :
            ent15.initialize("Low")
        elif (result == [1]) :
            ent16.initialize("Medium")
        else:
            ent16.initialize("High")

    def Reset(self):
                ent1.set("")
                ent2.set("")
        
                
                ent3.set("")
                ent4.set("")
                ent5.set("")
                ent6.set("")
                ent7.set("")
                ent8.set("")
                ent9.set("")
                ent10.set("")
                ent11.set("")
                ent14.set("")
                ent15.set("")
                ent16.set("")
                ent17.set("")
                

obj1=Project()
obj1.initial_gui()



