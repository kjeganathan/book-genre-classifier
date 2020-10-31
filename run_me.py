import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from operator import itemgetter 
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, f1_score

# read csv
data = pd.read_csv('final_book_summaries_m2.csv')

# turn dataframe into numpy array
# shape (816,4)
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
# ["Children's literature" 'Fantasy' 'Fiction' 'Mystery' 'Other'
#  'Science Fiction']
for i in range(len(data_y)):
    data_y[i] = list(json.loads(data_y[i]).values())

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



# 816 books/documents
# 405 distinct authors
# 26651 sentences total
# 32305 distinct words in in corpus (countvectorizer)
# 32003 without stop-words

# Tokenize books into counts with removing stop words 
vectorizer = CountVectorizer(stop_words= 'english')
X = vectorizer.fit_transform(data_x_title_summ)

# Bag of words
data_x_summ = X.toarray()

# finish combining author and title/summaries into final x data
data_x_comb = np.hstack((data_auth, data_x_summ))


# get stats on highest words counts 50
def stats_plot():
    words = {}
    counter = 0
    for i in data_x_summ.T:
        count_w = sum(i)
        # print(counter)
        words[vectorizer.get_feature_names()[counter]] = count_w
        counter += 1

    res_words = dict(sorted(words.items(), key = itemgetter(1), reverse = True)[:50])

    max_words = list(res_words.keys())
    max_words_counts = list(res_words.values())
    max_words.reverse()
    max_words_counts.reverse()

    # plot max count words 
    fig = plt.figure(figsize=(9,20))
    ax = fig.add_subplot(111)
    ax.barh(max_words, max_words_counts, align='center',height=0.5)
    ax.set_ylabel("Words")
    ax.set_xlabel("Count")
    ax.set_title("50 Most Frequent Words Without Stopwords")
    plt.show()


# Split into training and test data 80/20 split (random)
x_train, x_test, y_train, y_test = train_test_split(data_x_comb, data_y, test_size=0.20, random_state=42)


# logistic regeression 
def log_reg():
    logreg = LogisticRegression(
        random_state=0, solver="lbfgs", multi_class="multinomial", penalty="l2")

    clf = OneVsRestClassifier(logreg)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # class report to get precision and recall for each class
    class_rep = classification_report(y_test, y_pred)
    print(class_rep)


# performing grid search for svm
# Note: THIS TAKES SO LONG AS SVM TAKES FOREVER TO RUN (1 HR - 3 HRS)
def best_params_svm():

    params = {'estimator__C':[0.0001, 0.001, 0.01, 1] ,'estimator__kernel':['linear', 'rbf'], 
                'estimator__gamma':[0.01,0.05,0.1,1]}
    svc = svm.SVC()
    ovrc = OneVsRestClassifier(svc)

    clf = GridSearchCV(estimator=ovrc, param_grid=params, scoring= 'f1_weighted', cv=4)
    clf.fit(data_x_comb,data_y)

    print(clf.best_params_)


# SVM
def supp_vec_mach():

    # {'estimator__C': 0.01, 'estimator__kernel': 'linear'}

    # {'estimator__C': 0.01, 'estimator__gamma': 0.01, 'estimator__kernel': 'linear'}

    svms = svm.SVC(C=0.01,gamma= 0.01, kernel= 'linear')

    clf = OneVsRestClassifier(svms)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # class report to get precision and recall for each class
    class_rep = classification_report(y_test, y_pred)
    print(class_rep)


#################################################################################################################

supp_vec_mach()