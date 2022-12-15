import matplotlib.pyplot as plt
import numpy as np
from recommendation import choice1
from recommendation import choice2
from sklearn import preprocessing
from matplotlib import pyplot as ply
normal=[27, 20, 25, 24, 10, 6, 31, 16, 11, 3, 33, 17, 29, 35, 37]
multi=[2, 6, 8, 10, 12, 17, 18, 20, 21, 25, 27, 30, 33, 35, 37]
#[5, 7, 9, 11, 12, 14, 16, 20, 22, 24, 29, 31, 34, 35, 37]
#[3, 6, 8, 10, 12, 14, 16, 17, 20, 21, 25, 27, 30, 31, 32, 35]
#[2, 6, 8, 10, 12, 17, 18, 20, 21, 25, 27, 30, 33, 35, 37]
#[6, 8, 10, 12, 14, 17, 20, 23, 25, 27, 29, 30, 32, 35, 37]
x=[]
y=[]
X=[]
Y=[]
for each in normal:
    x.append(choice1[each])
    y.append(choice2[each])
print(x)
print(y)
for each in multi:
    X.append(choice1[each])
    Y.append(choice2[each])
print(X)
print(Y)

print(np.std(x))
print(np.std(X))
print(np.std(y))
print(np.std(Y))
group1=np.array(x)/np.array(y)
group2=np.array(X)/np.array(Y)
print(np.std(group1))
print(np.std(group2))


plt.scatter(range(len(group2)),group2,alpha=0.65,marker='v',label='Single-objective')
plt.scatter(range(len(group1)),group1,alpha=0.65,marker='^',label='Multi-objective')
plt.title('Advantage of Multi-objective recommendation: rating/popularity')
plt.xlabel('Book index')
plt.ylabel('Rating/popularity')
plt.grid()
plt.legend()
'''
plt.scatter(y,x,alpha=0.65,label='Single-objective',marker='v') #normal
plt.scatter(Y,X,alpha=0.65,label='Multi-objective',marker='^') #multi
plt.title('Recommended Books')
plt.xlabel('Popularity')
plt.ylabel('Rating')
plt.grid()
plt.legend()
'''
plt.show()