from sklearn import linear_model
import pickle
from matplotlib import pyplot as plt
import numpy as np

with open('dic_book.pickle', 'rb') as f:
    book = pickle.load(f)


num=[]
vector=[]
for key, value in book.items():
    num.append(value[0])
    vector.append(value[1])

training_num=num
training_vector=vector

testing_num=num[-10:]
testing_vector=vector[-10:]

clf=linear_model.Lasso(alpha=0.0005,normalize=False,max_iter=5000)
clf.fit(training_vector,training_num)
cof2=clf.coef_
inter2=clf.intercept_
print(clf.score(training_vector,training_num))

choice0=np.dot(np.array(training_vector),np.array(cof2))+inter2
mse = ((choice0 - training_num)**2).mean()
print(mse)

choice=np.dot(np.array(testing_vector),np.array(cof2))+inter2
mse = ((choice - testing_num)**2).mean()
print(mse)
'''
plt.scatter(range(len(choice)),choice/10,label='Predicted',marker='v',alpha=0.8)
plt.scatter(range(len(testing_num)),np.array(testing_num)/10,label='Ground truth',marker='^',alpha=0.8)
plt.legend()
plt.grid()
plt.title('Comparison between popularity ground truth and prediced values')
plt.xlabel('Book index')
plt.ylabel('Number of Ratings')
plt.show()


from sklearn.metrics import r2_score

coefficient_of_dermination = r2_score(testing_num,choice)
print(coefficient_of_dermination)
'''