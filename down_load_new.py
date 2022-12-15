import json
import numpy as np
from nltk.corpus import stopwords
import pickle
EngStopWords = set(stopwords.words('english'))

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

books = load_data('goodreads_books_poetry.json',head=36514)


#save the book vector, book id, rating times to the dictionary training set: 1900 testing set: 100
book_dic={}
itering=0

for each in range(3000):
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
with open('dic_book.pickle', 'wb') as handle:
    pickle.dump(book_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Load the interaction
interaction= load_data('goodreads_interactions_poetry.json',head=5000)
dic={}
for each in range(5000):
    if interaction[each]['is_read']==True:
        try:
            dic[interaction[each]['user_id']].append([interaction[each]['book_id'],float(interaction[each]['rating'])])
        except:
            dic[interaction[each]['user_id']]=[[interaction[each]['book_id'],float(interaction[each]['rating'])]]

i=0
key_initial='0'
result_initial=[]
for key, value in dic.items():
    if len(value)>=i:
        i=len(value)
        key_initial=key
        result_initial=value
dic0={}
dic0[key_initial]=result_initial
with open('dic_user.pickle', 'wb') as handle1:
    pickle.dump(dic0, handle1, protocol=pickle.HIGHEST_PROTOCOL)

training_user={}
vectors=[]

for value in dic0.values():
    for each in value:
        training_user[each[0]]=each[1]

ratings=[]
for i, k in training_user.items():
    for j in books:
        if j['book_id']==i:
            content = j['description']
            if content != '':
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
                    vectors.append(v)
                    ratings.append(k)

with open('dic_user_rating.pickle', 'wb') as handle2:
    pickle.dump(ratings, handle2, protocol=pickle.HIGHEST_PROTOCOL)
with open('dic_user_vector.pickle', 'wb') as handle3:
    pickle.dump(vectors, handle3, protocol=pickle.HIGHEST_PROTOCOL)
f3.close()