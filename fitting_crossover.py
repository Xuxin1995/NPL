import numpy as np
from scipy.optimize import curve_fit
from scipy import special   
import matplotlib.pyplot as plt 
import os
import pandas as pd

def func(z,a,b):
   return 0.5+0.5*special.erf((z-a)/b)
   
def funRev(z,a,b):
   return 0.5-0.5*special.erf((z-a)/b)

# list = os.listdir(path)
# N=len(list)
for i in range(19,20):
    plt.cla()
    Y11 = np.loadtxt('20outputs.txt')
    Y12 = np.loadtxt('20outputs2.txt')
    Y21 = np.loadtxt('40outputs.txt')
    Y22 = np.loadtxt('40outputs2.txt')
    # Y31 = np.loadtxt(path3+'outputs.txt')
    # Y32 = np.loadtxt(path3+'outputs2.txt')
    # Y41 = np.loadtxt(path4+'outputs.txt')
    # Y42 = np.loadtxt(path4+'outputs2.txt')
    # Y51 = np.loadtxt(path5+'outputs.txt')
    # Y52 = np.loadtxt(path5+'outputs2.txt')
    # Y61 = np.loadtxt(path6+'outputs.txt')
    # Y62 = np.loadtxt(path6+'outputs2.txt')
    # Y71 = np.loadtxt(path7+'outputs.txt')
    # Y72 = np.loadtxt(path7+'outputs2.txt')

    # Y1 = (Y11+Y21+Y31+Y41+Y51)/5
    # Y2 = (Y12+Y22+Y32+Y42+Y52)/5
    
# np.savetxt('Y1.txt',Y1)
# np.savetxt('Y2.txt',Y2)
    # # print(type(Y11))
    X = np.loadtxt('T.txt')    
    # popt, pcov = c
    # urve_fit(func, coorAll[:-1,0], pA[iter,:])
    popt1, pcov1 = curve_fit(funRev, X, Y11)
    popt2, pcov2 = curve_fit(func, X, Y12)
    popt3, pcov3 = curve_fit(funRev, X, Y21)
    popt4, pcov4 = curve_fit(func, X, Y22)
    # popt5, pcov5 = curve_fit(funRev, X, Y31)
    # popt6, pcov6 = curve_fit(func, X, Y32)
    # popt7, pcov7 = curve_fit(funRev, X, Y41)
    # popt8, pcov8 = curve_fit(func, X, Y42)
    # popt9, pcov9 = curve_fit(funRev, X, Y51)
    # popt10, pcov10 = curve_fit(func, X, Y52)

    a1 = popt1[0]   
    b1= popt1[1] 
    y1vals = funRev(X,a1,b1) 
    a2 = popt2[0]   
    b2= popt2[1] 
    y2vals = func(X,a2,b2)
    a3 = popt3[0]   
    b3= popt3[1] 
    y3vals = funRev(X,a3,b3)  
    a4 =popt4[0]
    b4= popt4[1] 
    y4vals = func(X,a4,b4)
    X1 = (X-a1)*20
    X2 = (X-a3)*40

    plt.figure(figsize=(6,8))
    plot1 = plt.plot(X1, y1vals, 'o-',color='b',label='L=20')  
    # plot2 = plt.plot(X, y1vals, 'r',label='polyfit values')
    plot3 = plt.plot(X1, y2vals, 'o-',color='b')  
    plot5 = plt.plot(X2, y3vals, 's-',color='r',label='L=40')  
    plot6 = plt.plot(X2, y4vals, 's-',color='r')  
    # plot7 = plt.plot(X, Y31, 'v-',color='r',label='6',alpha=0.5)  
    # plot8 = plt.plot(X, Y32, 'v-',color='r',alpha=0.5)  
    # plot9 = plt.plot(X, Y41, 'p-',color='c',label='8',alpha=0.5)  
    # plot10 = plt.plot(X, Y42, 'p-',color='c',alpha=0.5)  
    # plot11 = plt.plot(X, Y51, '*-',color='m',label='10',alpha=0.5)  
    # plot12 = plt.plot(X, Y52, '*-',color='m',alpha=0.5)  
    # plot13 = plt.plot(X, Y61, '+-',color='y',label='12',alpha=0.5)  
    # plot14 = plt.plot(X, Y62, '+-',color='y',alpha=0.5)  
    # plot15 = plt.plot(X, Y71, 'd-',color='k',label='14',alpha=0.5)  
    # plot16 = plt.plot(X, Y72, 'd-',color='k',alpha=0.5)  
    plt.xlim(-15,15)
    # plot4 = plt.plot(X, y2vals, 'r',label='polyfit values') 
    # plt.title('rate1e-2'+'    '+str((i+1)*1000)+'    '+'popt:'+str(popt1))
    plt.legend()
    # plt.savefig(New_path+str((i+1)*1000)+'1e-2.png')  
    plt.savefig('crossover3.png') 

    plt.show()  
