import pandas as pd
import numpy as np
import json
import ast
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

# read csv
# data = pd.read_csv('final_book_summaries_m2.csv')
data = pd.read_csv('book_summaries_6500.csv')

# turn dataframe into numpy array
np_array = data.to_numpy()


# split data into X and Y
# Y is the genres for each book
data_y = np_array[:, 2]

# X is author, book_title, summaries
data_x = np.delete(np_array, [2], 1)


# Combining title and summaries into one section
title = np_array[:,0].astype(str)

for i in range(len(title)):
    title[i] += ' '

data_x_title_summ = np.core.defchararray.add(title, data_x[:,-1].astype(str))


# Fix Y labels into matrix
# ["Children's literature" 'Fantasy' 'Fiction' 'Mystery' 'Science Fiction']
for i in range(len(data_y)):

    # new for 6500
    data_y[i] = ast.literal_eval(data_y[i])
    # new
    data_y[i] = list(data_y[i].values())

    # old for 800
    # data_y[i] = list(json.loads(data_y[i]).values())

mlb = MultiLabelBinarizer()
data_y = mlb.fit_transform(data_y)


# Fix authors into one hot vector
# First turn authors into numerical categorical data
labelencoder = LabelEncoder()
auth_df = pd.DataFrame(data_x[:,1].astype(str), columns=['Authors'])
auth_df["Author_num"] = labelencoder.fit_transform(auth_df['Authors'])

# Second use numericial categorical data for one hot vectoring
enc = OneHotEncoder(handle_unknown='ignore')
data_x_hot = pd.DataFrame(enc.fit_transform(auth_df[['Author_num']]).toarray())
data_auth = data_x_hot.to_numpy()


# WORD EMBEDDINGS
# get glove
# iterate through each word in book get vector 
# avg vector in the end for each book to get single vector
# then combine with author hot vector 

lines = open('glove.6b\glove.6B.50d.txt', encoding='utf-8').readlines()

# store word embeddings in dictionary
wordvecs = {}

for line in lines:
    word = line.split()[0]
    vec = np.asarray(line.split()[1:], 'float32')
    wordvecs[word] = vec

# vectorize and get BOW
vectorizer = CountVectorizer(stop_words= 'english')
X = vectorizer.fit_transform(data_x_title_summ)

# Bag of words
data_x_summ = X.toarray()

bow = vectorizer.get_feature_names()


# create data X of word embeddings
final_xs = np.empty((0,50), dtype='float')

# iterate through each book in BOW
for book in data_x_summ:

    index = 0
    total = 0
    avg = np.zeros(50, dtype= 'float')

    # iterate through each word in each book
    for word in book:

        # check for words 
        if word != 0:

            # get occurances of each word
            count = word
            # get word
            wd = bow[index]

            # check if GLOVE has the word
            if wd in wordvecs.keys():

                # multiply vector by number of times word occurred
                wdvec = wordvecs[wd] * count
                avg = np.add(avg,wdvec)
                total += count
        
        index += 1

    # average the word vector
    avg = np.array(avg/total)

    # stack on final X data
    final_xs = np.vstack((final_xs, avg))


# final_xs is word embeddings for each book

# Now combine author hot vector and word embeddings
# finish combining author and title/summaries into final x data
# data_x_comb = np.hstack((data_auth, final_xs))

# commented that out since word embedding is dim 50 and and author hot matrix is 2055
# so just try out word embedding by itself first maybe faster without author

# Split into training and test data 80/20 split (random)
x_train, x_test, y_train, y_test = train_test_split(final_xs, data_y, test_size=0.20, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)