from sklearn import linear_model
import pickle
from matplotlib import pyplot as plt
import numpy as np

with open('dic_user_rating.pickle', 'rb') as f:
    rating = pickle.load(f)

with open('dic_user_vector.pickle', 'rb') as f:
    vectors = pickle.load(f)

training_rating=rating
training_vectors=vectors

testing_rating=rating[:12]
testing_vectors=vectors[:12]
clf=linear_model.Lasso(alpha=0.000001,normalize=False,max_iter=5000)
clf.fit(training_vectors,training_rating)
cof1=clf.coef_
inter1=clf.intercept_
print(clf.score(training_vectors,training_rating))

choice0=np.dot(np.array(training_vectors),np.array(cof1))+inter1
mse = ((choice0 - training_rating)**2).mean()
print(mse)

choice=np.dot(np.array(testing_vectors),np.array(cof1))+inter1
mse = ((choice - testing_rating)**2).mean()
print(mse)

'''
plt.scatter(range(12),choice,label='Predicted',marker='v',alpha=0.8)
plt.scatter(range(12),testing_rating,label='Ground truth',marker='^',alpha=0.8)
plt.legend()
plt.title('Comparison between rating ground truth and prediced values')
plt.xlabel('Book index')
plt.ylabel('Rating')
plt.grid()
plt.show()


from sklearn.metrics import r2_score

coefficient_of_dermination = r2_score(testing_rating,choice)
print(coefficient_of_dermination)
'''

