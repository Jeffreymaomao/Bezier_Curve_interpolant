import numpy as np
from matplotlib import pyplot as plt
from decimal import Decimal #精于计算
from decimal import getcontext #保留的小数位数，自己设置
################################################################
################################################################
def C(n, m):
    N = Decimal (np.math.factorial(n))
    M = Decimal (np.math.factorial(m))
    nm = int(n-m)
    NM = Decimal (np.math.factorial(nm))
    return int(N/(M*NM))
################################################################
################################################################
def M(n):
    M = np.ones((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            M[j,i] = C(n,i) * (j**i) * ((n-j)**(n-i))
    return (M).astype(np.float64)# need to divide n**n
################################################################
################################################################
def IM(n):
    M = np.ones((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            M[j,i] = C(n,i) * (j**i) * ((n-j)**(n-i))
    M = (M/(n**n)).astype(np.float64)
    Det = np.linalg.det(M)
    if Det == 0.0:
        print('The matrix is sigular, using Bezier curve instead.')
        IM = np.eye(n+1)
    else:
        IM = np.linalg.inv(M)
    return IM
################################################################
################################################################
def Bezier(Px, Py, t):
    n = max(len(Px),len(Py))-1
    x,y = 0,0
    for i in range(n+1):
        x = x + C(n,i) * (t**i) * (1-t)**(n-i) * Px[i]
        y = y + C(n,i) * (t**i) * (1-t)**(n-i) * Py[i]
    return x,y
################################################################
################################################################
def Bezier_interpolant(rx, ry, t):
    # n > 60  Ｍ之反矩陣無法求 #
    # n > 145 矩陣太大球不出來 # 
    n = max(len(rx),len(ry))-1
    x,y = 0,0
    iM = IM(n)
    mx = iM.dot(rx)
    my = iM.dot(ry)
    for i in range(n+1):
        x = x + C(n,i) * (t**i) * (1-t)**(n-i) * mx[i]
        y = y + C(n,i) * (t**i) * (1-t)**(n-i) * my[i]
    return x,y
################################################################
################################################################
def Bezier_interpolant_segment_point(rx, ry, t, N_points):
    N = N_points
    n = max(len(rx),len(ry))
    s = int(np.floor((n-1)/(N-1)))  #How mamy segmentation(+rest)
    R = s*(N-1)                     #How many interval 
    r = n - R -1                    #How many interval (rest)
    T1 = (R/(R+r))/s            #Segment length 
    T2 = (r/(R+r))              #Segment length (rest)
    I = int(np.floor(t/T1))     #Which segment (1,2,...)
    i = int(I*(N-1))          #Which initial point to interpolant
    if T2 == 0:
        T = t/T1 - I
        x,y = Bezier_interpolant(rx[i:i+N], ry[i:i+N], T)
    else:
        if I < s:
            T = t/T1 - I
            x,y = Bezier_interpolant(rx[i:i+N], ry[i:i+N], T)
        else:
            T = (t-s*T1)/T2 # rest
            x,y = Bezier_interpolant(rx[i:], ry[i:], T)
    return x,y
################################################################
################################################################
def Bezier_interpolant_segment(rx, ry, t, N_points):
    N = N_points # how many interpolant points in a segmant 
    #-If the N is too small choose N = 2--------------------
    if N<2:
        if N==1:
            print('You can not seperate the points by {:} point.'.format(N))
            print('Let N_points = 2')
        else:
            print('You can not seperate the points by {:} points.'.format(N))
            print('Let N_points = 2')
        N = 2
    #-------------------------------------------------------
    n = min(len(rx),len(ry))
    #--N is equal or biger then the data--------------------
    if N >= n:
        print('There is no need to seperate the curve.')
        x,y = Bezier_interpolant(rx, ry, t)
    #-------------------------------------------------------
    else:
        if np.size(t) ==1:
            x,y = Bezier_interpolant_segment_point(rx, ry, t, N)
        else:
            x,y = t*0,t*0
            for i in range(len(t)):
                x[i],y[i] = Bezier_interpolant_segment_point(rx, ry, t[i], N)
    return x,y
################################################################
################################################################

def Plot(List,Label = '',color = 'r'):
    List 
    x = np.arange(0,len(List))
    plt.title('Graph {}'+Label)
    plt.plot(x,List,color)
    plt.scatter(x,List,c = color,s=5,label = Label)
    
################################################################
################################################################

def plot_Bezier(x_int, y_int, X, Y):
    x = x_int
    y = y_int
    annotations=[]
    for i in range(len(x)):
        annotations.append("P{}".format(i+1))
    plt.figure(figsize=(8, 6))
    plt.title('Interpolant {:} points by Bezier curve'.format(len(x)),
              fontsize = 20)
    plt.plot(X,Y,'b',
             label = 'Bezier curve',
             zorder=1,
             linewidth = 1)
    '''
    plt.scatter(X,Y,c='b',
                label = 'Bezier points',
                zorder=1,
                s = 5)
    '''
    plt.scatter(x,y,c='r',
                label = 'Interpolant points',
                zorder=2,
                s = 10)

    S = 50
    sx = np.abs((max(x)-min(x))/S)
    sy = np.abs((max(y)-min(y))/S)
    for i, label in enumerate(annotations):
        plt.annotate(label,
                     (x[i]+sx, y[i]-sx),
                     verticalalignment='top')
    L = 10
    plt.xlim((min(x)-sx*L, max(x)+sx*L))
    plt.ylim((min(y)-sy*L, max(y)+sy*L))
    plt.legend()
    plt.show()


