import json
import numpy as np
from nltk.corpus import stopwords
import pickle
EngStopWords = set(stopwords.words('english'))
from loss1 import cof1, inter1
from loss2 import cof2, inter2
'''
f3=open('glove.6B.50d.txt','r')

#Generate the glove dictionary for mapping
glove={}
for lines in f3:
    l=lines.split()
    word=l[0]
    list0=[]
    for each in l[1:]:
        list0.append(float(each))
    glove[word]=np.array(list0)

def load_data(file_name, head):
    count = 0
    data = []
    with open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            count += 1
            data.append(d)
            # break if reaches the 100th line
            if (head is not None) and (count > head):
                break
    return data

books = load_data('goodreads_books_poetry.json',head=2000)


#save the book vector, book id, rating times to the dictionary training set: 1900 testing set: 100
book_dic={}
itering=0

for each in range(1100,1200):
    content=books[each]['description']
    try:
        idnum=books[each]['book_id']
        num_rating=float(books[each]['ratings_count'])

        if content!='':
            v = np.zeros(50)
            count = 0
            for word in content.split():
                if word in EngStopWords:
                    pass
                else:
                    if word in list(glove.keys()):
                        v = v + glove[word]
                        count += 1
            if count != 0:
                v /= count
                book_dic[idnum]=[num_rating,v]
                if itering%100==0:
                    print(itering)
                itering+=1
    except ValueError:
        print('wrong line:')
        print(each)
with open('dic_book_rec.pickle', 'wb') as handle:
    pickle.dump(book_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)'''
with open('dic_book_rec.pickle', 'rb') as f:
    book_dic = pickle.load(f)
num=[]
vector=[]
for key, value in book_dic.items():
    num.append(value[0])
    vector.append(value[1])

choice1=np.dot(np.array(vector),np.array(cof1))+inter1 #student
choice2=np.dot(np.array(vector),np.array(cof2))+inter2 #popularity


choice2+=200
choice2/=100


sorted_index_array = np.argsort(choice1[:50])
sorted_array = choice1[:50][sorted_index_array]
print(sorted_index_array)
print(sorted_array)