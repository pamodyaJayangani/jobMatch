import pandas as pd
import numpy as np
import re

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from xgboost import plot_importance
import nltk

# nltk.download('stopwords')

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

# First XGBoost model for MBTI dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# Tune learning_rate
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pickle

# read data
data = pd.read_csv('D:\\5THYEAR\\FYP\\DataSet\\Personality_mbti_2.csv')
print(data.head(10))
print("***************************************")
[p.split('|||') for p in data.head(2).posts.values]
cnt_types = data['type'].value_counts()

plt.figure(figsize=(12, 4))
sns.barplot(cnt_types.index, cnt_types.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Types', fontsize=12)
plt.show()


def get_types(row):
    t = row['type']

    I = 0;
    N = 0
    T = 0;
    J = 0

    if t[0] == 'I':
        I = 1
    elif t[0] == 'E':
        I = 0
    else:
        print('I-E incorrect')

    if t[1] == 'N':
        N = 1
    elif t[1] == 'S':
        N = 0
    else:
        print('N-S incorrect')

    if t[2] == 'T':
        T = 1
    elif t[2] == 'F':
        T = 0
    else:
        print('T-F incorrect')

    if t[3] == 'J':
        J = 1
    elif t[3] == 'P':
        J = 0
    else:
        print('J-P incorrect')
    return pd.Series({'IE': I, 'NS': N, 'TF': T, 'JP': J})


data = data.join(data.apply(lambda row: get_types(row), axis=1))
data.head(5)

print("Introversion (I) /  Extroversion (E):\t", data['IE'].value_counts()[0], " / ", data['IE'].value_counts()[1])
print("Intuition (N) – Sensing (S):\t\t", data['NS'].value_counts()[0], " / ", data['NS'].value_counts()[1])
print("Thinking (T) – Feeling (F):\t\t", data['TF'].value_counts()[0], " / ", data['TF'].value_counts()[1])
print("Judging (J) – Perceiving (P):\t\t", data['JP'].value_counts()[0], " / ", data['JP'].value_counts()[1])

N = 4
but = (
data['IE'].value_counts()[0], data['NS'].value_counts()[0], data['TF'].value_counts()[0], data['JP'].value_counts()[0])
top = (
data['IE'].value_counts()[1], data['NS'].value_counts()[1], data['TF'].value_counts()[1], data['JP'].value_counts()[1])

ind = np.arange(N)  # the x locations for the groups
width = 0.7  # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, but, width)
p2 = plt.bar(ind, top, width, bottom=but)

plt.ylabel('Count')
plt.title('Distribution accoss types indicators')
plt.xticks(ind, ('I/E', 'N/S', 'T/F', 'J/P',))

plt.show()

data[['IE', 'NS', 'TF', 'JP']].corr()

cmap = plt.cm.RdBu
corr = data[['IE', 'NS', 'TF', 'JP']].corr()
plt.figure(figsize=(12, 10))
plt.title('Pearson Features Correlation', size=15)
sns.heatmap(corr, cmap=cmap, annot=True, linewidths=1)

b_Pers = {'I': 0, 'E': 1, 'N': 0, 'S': 1, 'F': 0, 'T': 1, 'J': 0, 'P': 1}
b_Pers_list = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'}, {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]


def translate_personality(personality):
    # transform mbti to binary vector

    return [b_Pers[l] for l in personality]


def translate_back(personality):
    # transform binary vector to mbti personality

    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s


# Check ...
d = data.head(4)
list_personality_bin = np.array([translate_personality(p) for p in d.type])
print("Binarize MBTI list: \n%s" % list_personality_bin)

#### Compute list of subject with Type | list of comments
# import nltk
#
# # nltk.download('stopwords')
#
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk import word_tokenize

# We want to remove these from the psosts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP',
                    'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']

unique_type_list = [x.lower() for x in unique_type_list]

# Lemmatize
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

# Cache the stop words for speed
cachedStopWords = stopwords.words("english")


def pre_process_data(data, remove_stop_words=True, remove_mbti_profiles=True):
    list_personality = []
    list_posts = []
    len_data = len(data)
    i = 0

    for row in data.iterrows():
        i += 1
        if (i % 500 == 0 or i == 1 or i == len_data):
            print("%s of %s rows" % (i, len_data))

        ##### Remove and clean comments
        posts = row[1].posts
        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)
        temp = re.sub("[^a-zA-Z]", " ", temp)
        temp = re.sub(' +', ' ', temp).lower()
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t, "")

        type_labelized = translate_personality(row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(temp)

    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality


# nltk.download('wordnet')
list_posts, list_personality = pre_process_data(data, remove_stop_words=True)

print("Num posts and personalities: ", list_posts.shape, list_personality.shape)
list_posts[0]
list_personality[0]

# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.manifold import TSNE

# Posts to a matrix of token counts
cntizer = CountVectorizer(analyzer="word",
                          max_features=1500,
                          tokenizer=None,
                          preprocessor=None,
                          stop_words=None,
                          max_df=1.0,
                          min_df=0.0)

# Learn the vocabulary dictionary and return term-document matrix
print("CountVectorizer...")
X_cnt = cntizer.fit_transform(list_posts)

# Transform the count matrix to a normalized tf or tf-idf representation
tfizer = TfidfTransformer()

print("Tf-idf...")
# Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
X_tfidf = tfizer.fit_transform(X_cnt).toarray()

feature_names = list(enumerate(cntizer.get_feature_names()))
feature_names
print("**************^^^^^^^^^^^^^^^^^")
for x in range(len(feature_names)):
    print (feature_names[x])
    print(",")

print("**************^^^^^^^^^^^^^^^^^")
X_tfidf.shape
print("X: Posts in tf-idf representation \n* 1st row:\n%s" % X_tfidf[0])

type_indicators = ["IE: Introversion (I) / Extroversion (E)", "NS: Intuition (N) – Sensing (S)",
                   "FT: Feeling (F) - Thinking (T)", "JP: Judging (J) – Perceiving (P)"]

for l in range(len(type_indicators)):
    print(type_indicators[l])

print("MBTI 1st row: %s" % translate_back(list_personality[0, :]))
print("Y: Binarized MBTI 1st row: %s" % list_personality[0, :])

# # First XGBoost model for MBTI dataset
# from numpy import loadtxt
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
#
# from sklearn.metrics import accuracy_score


# Posts in tf-idf representation
X = X_tfidf

# Let's train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    # Let's train type indicator individually
    Y = list_personality[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))

# Let's train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    Y = list_personality[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    model = XGBClassifier()
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))

# from xgboost import plot_importance

# Only the 1st indicator
y = list_personality[:, 0]
# fit model on training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
ax = plot_importance(model, max_num_features=25)

fig = ax.figure
fig.set_size_inches(15, 20)

plt.show()

features = sorted(list(enumerate(model.feature_importances_)), key=lambda x: x[1], reverse=True)
for f in features[0:25]:
    print("%d\t%f\t%s" % (f[0], f[1], cntizer.get_feature_names()[f[0]]))

# Save xgb_params for late discussuin
default_get_xgb_params = model.get_xgb_params()

# Save xgb_params for later discussuin
default_get_xgb_params = model.get_xgb_params()
print(default_get_xgb_params)

# setup parameters for xgboost
param = {}

param['n_estimators'] = 100
param['max_depth'] = 1
param['nthread'] = 4
param['learning_rate'] = 0.1

# Let's train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    Y = list_personality[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("* %s Accuracy: %.2f%%" % (type_indicators[l], accuracy * 100.0))

# # Tune learning_rate
# from numpy import loadtxt
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold

# Posts in tf-idf representation
X = X_tfidf

# setup parameters for xgboost
param = {}
param['n_estimators'] = 100
param['max_depth'] = 1
param['nthread'] = 4
param['learning_rate'] = 0.1

# Let's train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    Y = list_personality[:, l]
    model = XGBClassifier(**param)
    # learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    # param_grid = dict(learning_rate=learning_rate)

    param_grid = {
        'n_estimators': [100, 150],
        'learning_rate': [0.1, 0.15]
        # 'learning_rate': [ 0.01, 0.1, 0.2, 0.3],
        # 'max_depth': [2,3,4],
    }

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X, Y)

    # summarize results
    print("* Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("* %f (%f) with: %r" % (mean, stdev, param))

description = " Page |  1     Umeshika Ridmi Herath   Kopiyawatta, Alpitiya, Dewanagala ,Mawanella, Sri Lanka.   TEL: +94715412601 | +94355674711   E - MAIL:umaherath2014@gmail.com   Objective       Graduate of Bachelor of Science in Information Technology at Sri Lanka Institute of Information  Technology,   is   keen   to   find   a   Software   Quality   Assurance   job.   Willing   to   learn   under   professionals  to   develop   knowledge   and   improve   skills   in   a   practical   enviro nment   while   making   myself   grow   with  the industry. As a reliable, well - coordinated and open - minded individual, I seek an opportunity to  learn by making positive contribution to enhance the performance of your   company.     Education       Bachelor of Science in Information Technology  (2017)  Sri  Lanka Institute of Information Technology (SLIIT)      G.C.E. (A/L) Examination (201 3 )     Science   G.C.E. (O/L) Examination (2009)       Project and   Research     Baby Learn with Pooh  -   Final   Year   Research   2016     Hybrid mobile application for toddlers, which helps them to learn. Help their parents to figure out  the strength and weaknesses of their kids.   Technologies: PhoneGap (cordova) , Javascript, PHP, HTML, CSS, SQL     Research   Paper   Publication   June   2016     How to Develop Proper Communication in Company Combination with Topology   http://www.ijsrp.org/research - pap er - 0416/ijsrp - p5283.pdf        Management   System   June   2015     Developed a stand - alone java application for Atlas agency, which helps the agent easily manage  distributions through the system, that automated most of the process.   Technologies: J ava, iReport, MySQL               Page |  2       Student Enrollment   Management   System   January   2015      staff students and handle events.   Technologies: PHP, Javascript, HTML, CSS, MySQL       Key Skills and   Competencies          Programming and coding   skills   o   Java, PHP, JavaScript, C#, HTML, CSS,   SQL   o   Tools and IDE: Visual Studio, NetBeans, Dreamweaver, Cisco Packet   Tracer      Excellent ability to take   directions      Good Team   player      Ability to meet the goals in given   time      Ability to work in tough   deadlines       Personal   Information     Name with Initials  : H. M. U. R. Herath   Full   Name   : Herath Mudiyanselage Umeshika Ridmi   Herath   Date   of Birth   :   08.09.1993   NIC   :   937523734V   Gender   :   Female   School   of Attend      Kegalle     Reference       Mrs.K.W.S.P.Athukorala   Mrs. Gayana   Fernando   Director   (projects)   Course Coordinator MSc in   PM(Keele)   Presidential   secretariat   SLIIT Computing   Colombo.   Level 13   0718486907   BOC Merchant   Tower   St Michaels Rd  Colombo 03   Email:  gayana.f@sliit.lk     I hereby certify that the above particular given by me are true and accurate to the best of my  knowledge .    "
# A few few tweets and blog post
# my_posts = " Page |  1     Umeshika Ridmi Herath   Kopiyawatta, Alpitiya, Dewanagala ,Mawanella, Sri Lanka.   TEL: +94715412601 | +94355674711   E - MAIL:umaherath2014@gmail.com   Objective       Graduate of Bachelor of Science in Information Technology at Sri Lanka Institute of Information  Technology,   is   keen   to   find   a   Software   Quality   Assurance   job.   Willing   to   learn   under   professionals  to   develop   knowledge   and   improve   skills   in   a   practical   enviro nment   while   making   myself   grow   with  the industry. As a reliable, well - coordinated and open - minded individual, I seek an opportunity to  learn by making positive contribution to enhance the performance of your   company.     Education       Bachelor of Science in Information Technology  (2017)  Sri  Lanka Institute of Information Technology (SLIIT)      G.C.E. (A/L) Examination (201 3 )     Science   G.C.E. (O/L) Examination (2009)       Project and   Research     Baby Learn with Pooh  -   Final   Year   Research   2016     Hybrid mobile application for toddlers, which helps them to learn. Help their parents to figure out  the strength and weaknesses of their kids.   Technologies: PhoneGap (cordova) , Javascript, PHP, HTML, CSS, SQL     Research   Paper   Publication   June   2016     How to Develop Proper Communication in Company Combination with Topology   http://www.ijsrp.org/research - pap er - 0416/ijsrp - p5283.pdf        Management   System   June   2015     Developed a stand - alone java application for Atlas agency, which helps the agent easily manage  distributions through the system, that automated most of the process.   Technologies: J ava, iReport, MySQL               Page |  2       Student Enrollment   Management   System   January   2015      staff students and handle events.   Technologies: PHP, Javascript, HTML, CSS, MySQL       Key Skills and   Competencies          Programming and coding   skills   o   Java, PHP, JavaScript, C#, HTML, CSS,   SQL   o   Tools and IDE: Visual Studio, NetBeans, Dreamweaver, Cisco Packet   Tracer      Excellent ability to take   directions      Good Team   player      Ability to meet the goals in given   time      Ability to work in tough   deadlines       Personal   Information     Name with Initials  : H. M. U. R. Herath   Full   Name   : Herath Mudiyanselage Umeshika Ridmi   Herath   Date   of Birth   :   08.09.1993   NIC   :   937523734V   Gender   :   Female   School   of Attend      Kegalle     Reference       Mrs.K.W.S.P.Athukorala   Mrs. Gayana   Fernando   Director   (projects)   Course Coordinator MSc in   PM(Keele)   Presidential   secretariat   SLIIT Computing   Colombo.   Level 13   0718486907   BOC Merchant   Tower   St Michaels Rd  Colombo 03   Email:  gayana.f@sliit.lk     I hereby certify that the above particular given by me are true and accurate to the best of my  knowledge .    "
my_posts = " Page |  1     Umeshika Ridmi Herath   Kopiyawatta, Alpitiya, Dewanagala ,Mawanella, Sri Lanka.   TEL: +94715412601 | +94355674711   E - MAIL:umaherath2014@gmail.com   Objective       Graduate of Bachelor of Science in Information Technology at Sri Lanka Institute of Information  Technology,   is   keen   to   find   a   Software   Quality   Assurance   job.   Willing   to   learn   under   professionals  to   develop   knowledge   and   improve   skills   in   a   practical   enviro nment   while   making   myself   grow   with  the industry. As a reliable, well - coordinated and open - minded individual, I seek an opportunity to  learn by making positive contribution to enhance the performance of your   company.     Education       Bachelor of Science in Information Technology  (2017)  Sri  Lanka Institute of Information Technology (SLIIT)      G.C.E. (A/L) Examination (201 3 )     Science   G.C.E. (O/L) Examination (2009)       Project and   Research     Baby Learn with Pooh  -   Final   Year   Research   2016     Hybrid mobile application for toddlers, which helps them to learn. Help their parents to figure out  the strength and weaknesses of their kids.   Technologies: PhoneGap (cordova) , Javascript, PHP, HTML, CSS, SQL     Research   Paper   Publication   June   2016     How to Develop Proper Communication in Company Combination with Topology   http://www.ijsrp.org/research - pap er - 0416/ijsrp - p5283.pdf        Management   System   June   2015     Developed a stand - alone java application for Atlas agency, which helps the agent easily manage  distributions through the system, that automated most of the process.   Technologies: J ava, iReport, MySQL               Page |  2       Student Enrollment   Management   System   January   2015      staff students and handle events.   Technologies: PHP, Javascript, HTML, CSS, MySQL       Key Skills and   Competencies          Programming and coding   skills   o   Java, PHP, JavaScript, C#, HTML, CSS,   SQL   o   Tools and IDE: Visual Studio, NetBeans, Dreamweaver, Cisco Packet   Tracer      Excellent ability to take   directions      Good Team   player      Ability to meet the goals in given   time      Ability to work in tough   deadlines       Personal   Information     Name with Initials  : H. M. U. R. Herath   Full   Name   : Herath Mudiyanselage Umeshika Ridmi   Herath   Date   of Birth   :   08.09.1993   NIC   :   937523734V   Gender   :   Female   School   of Attend      Kegalle     Reference       Mrs.K.W.S.P.Athukorala   Mrs. Gayana   Fernando   Director   (projects)   Course Coordinator MSc in   PM(Keele)   Presidential   secretariat   SLIIT Computing   Colombo.   Level 13   0718486907   BOC Merchant   Tower   St Michaels Rd  Colombo 03   Email:  gayana.f@sliit.lk     I hereby certify that the above particular given by me are true and accurate to the best of my  knowledge .  "

# The type is just a dummy so that the data prep fucntion can be reused
mydata = pd.DataFrame(data={'type': ['INFJ'], 'posts': [my_posts]})

my_posts, dummy = pre_process_data(mydata, remove_stop_words=True)

my_X_cnt = cntizer.transform(my_posts)
my_X_tfidf = tfizer.transform(my_X_cnt).toarray()

# setup parameters for xgboost
param = {}
param['n_estimators'] = 100
param['max_depth'] = 1
param['nthread'] = 4
param['learning_rate'] = 0.1

result = []
# Let's train type indicator individually
for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))

    Y = list_personality[:, l]

    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # fit model on training data
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)

    #Save the model
    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # make predictions for my  data
    y_pred = model.predict(my_X_tfidf)
    result.append(y_pred[0])
    # print("* %s prediction: %s" % (type_indicators[l], y_pred))

print("The result is: ", translate_back(result))
