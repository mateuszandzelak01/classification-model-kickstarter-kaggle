import pandas as pd
import numpy as np
import string
import time

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import  f1_score
from sklearn.metrics import  confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import BallTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV

#### Data_set
pd.set_option('display.max_columns', 12)
df = pd.read_csv('data/ks-projects-201801.csv')

### Research
# plt.subplot(221, title='Main category')
# df['main_category'].value_counts().plot.bar()
# plt.subplot(222, title='Currency')
# df['currency'].value_counts().plot.bar()
# plt.subplot(223, title='Country')
# df['country'].value_counts().plot.bar()
# plt.subplot(224, title='State')
# df['state'].value_counts().plot.bar()
#plt.show()

### Data preparation
categorical_columns = ['main_category']
df = pd.get_dummies(df, columns=categorical_columns)
df = df[df["state"].isin(["failed", "successful"])]
df["state"] = df["state"].apply(lambda x: 1 if x=="successful" else 0)
df = df.drop(columns=['ID','name', 'pledged', 'goal', 'usd pledged', 'usd_pledged_real', 'category', 'currency', 'country'], axis=1)
df['launched'] = pd.to_datetime(df['launched'])
df['deadline'] = pd.to_datetime(df['deadline'])
df['duration_days'] = df['deadline'].subtract(df['launched'])
df['duration_days'] = df['duration_days'].astype('timedelta64[D]')
df = df.drop(columns=['launched', 'deadline'])

####Bag of Words
# df['name'] = df['name'].astype(str)
# df['name'] = df['name'].str.split()
#
# df['name'] = df['name'].apply(lambda x: ' '.join([i for i in x if i not in string.punctuation]))
# df['name'] = df['name'].str.lower()
# from nltk.corpus import stopwords
# stop = stopwords.words('english')
#
# df['name'] = df['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
# df['name'] = df['name'].str.split()
#
# from nltk.stem.snowball import SnowballStemmer
# stemmer = SnowballStemmer("english")
# df['name'] = df['name'].apply(lambda x: [stemmer.stem(y) for y in x])
#
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# vectorizer = TfidfVectorizer()
# bag_of_words = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False, max_features=50).fit_transform(df['name'])
# bow = bag_of_words.toarray()
# bow_df = pd.DataFrame(bow)
# df = pd.merge(df, bow_df, how='left', left_index=True, right_index=True)
# df = df.drop(columns=['name'], axis=1)
# print(bag_of_words.fe)

#####  Delete outlier
q1 = df['usd_goal_real'].quantile(0.25)
q3 = df['usd_goal_real'].quantile(0.75)
iqr = q3 - q1
down = q1 - 1.5 * iqr
up = q3 + 1.5 * iqr
df['usd_goal_real'] = df['usd_goal_real'][(df['usd_goal_real'] > down) & (df['usd_goal_real'] < up)]
df = df.dropna(axis=0, how='any')
# df['usd_goal_real'].plot(kind='box', logy=True)
# plt.show()

#####  Variables X,y
X = df.drop(columns=['state'], axis=1)
y = df['state']

sc = preprocessing.StandardScaler()
X = pd.DataFrame(sc.fit_transform(X.values), index=X.index, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

# scorer to compare results
scorer = make_scorer(accuracy_score)
kfold = KFold(n_splits=5, random_state=11)

#### 0. Logistic regresion
t0=time.time()
logreg = LogisticRegression(solver='lbfgs', multi_class='auto', n_jobs=-1, max_iter=100).fit(X_train, y_train)
coef_print = pd.DataFrame(logreg.coef_)
res_logreg_acc = cross_val_score(logreg, X_train, y_train, cv=kfold, scoring=acc_scorer)
res_logreg_f1 = cross_val_score(logreg, X_train, y_train, cv=kfold, scoring=f1_scorer)
acc_logreg = res_logreg_acc.mean()
f1_logreg = res_logreg_f1.mean()
print('Logistic regresion accuracy:\t', acc_logreg)
print('Logistic regresion F1 score:\t', f1_logreg)
#print(coef_print)
print('Time taken :' , time.time()-t0)

#### 1. KNN,  - Mateusz

t0=time.time()
knn = KNeighborsClassifier().fit(X_train, y_train)
res_knn_acc = cross_val_score(knn, X_train, y_train, cv=kfold, scoring=acc_scorer)
res_knn_f1 = cross_val_score(knn, X_train, y_train, cv=kfold, scoring=f1_scorer)
acc_knn = res_knn_acc.mean()
f1_knn = res_knn_f1.mean()
print('KNN accuracy:\t',acc_knn)
print('F1 score:\t',f1_knn)
print('Time taken :' , time.time()-t0)

#### 2. Random Forest
t0=time.time()
clf_rf = RandomForestClassifier(n_estimators=10, max_depth=10,random_state=101, min_samples_leaf=2, criterion="gini")
clf_rf.fit(X_train,y_train)
res_rf_acc = cross_val_score(clf_rf, X_train, y_train, cv=kfold, scoring=acc_scorer)
res_rf_f1 = cross_val_score(clf_rf, X_train, y_train, cv=kfold, scoring=f1_scorer)
acc_rf = res_rf_acc.mean()
f1_rf = res_rf_f1.mean()
print('Ranres_rf_f1dom Forest accuracy:\t', acc_rf)
print('Random Forest F1 score:\t', f1_rf)
print('Time taken :' , time.time()-t0)

#### 3. SVM
t0=time.time()
clf_svm = LinearSVC(max_iter = 100000, C=10000, dual=False)
clf_svm.fit(X_train, y_train)
res_svm_acc = cross_val_score(clf_svm, X_train, y_train, cv=kfold, scoring=acc_scorer)
res_svm_f1 = cross_val_score(clf_svm, X_train, y_train, cv=kfold, scoring=f1_scorer)
acc_svm = res_svm_acc.mean()
f1_svm = res_svm_f1.mean()
print('SVM accuracy:\t', acc_svm)
print('SVM F1 score:\t', f1_svm)
print('Time taken :' , time.time()-t0)


#### 4. XGBoost with hiperparameters


def run_xgboost_analysis():

    a = [2, 3, 4, 5, 6, 7, 8, 9, 12, 15]
    b = [0.09, 1.0, 1.1]
    c = [50, 100, 150, 200, 250, 300, 320, 350, 400]
    max_scr = 0
    max_dep = 0
    max_len = 0
    max_n_est = 0
    for i in a:
        for j in b:
            for k in c:

                clf_xgbr = XGBClassifier(max_depth=i, learning_rate=j, n_estimators=k)
                #
                results = cross_val_score(clf_xgbr, X_train, y_train, cv=kfold, scoring=scorer)
                #
                res_med = np.median(results)
                if res_med > max_scr:
                    max_dep = i
                    max_len = j
                    max_n_est = k
                    max_scr = res_med

    return max_scr, max_dep, max_len, max_n_est
max_scr_1, max_dep_1, max_len_1, max_n_est_1 = run_xgboost_analysis()
print('Best score is {0}, for parameters depth {1}, learning rate {2}, n_estimators {3}'.format(max_scr_1, max_dep_1, max_len_1, max_n_est_1))


######### Bayes

clf_gnb = GaussianNB()
clf_gnb.fit(X_train,y_train)
clf_gnb.fit(X_train,y_train)
y_pred_gnb = clf_gnb.predict(X_test)
cv_gnb = cross_val_score(clf_gnb, X_train, y_train, cv=kfold, scoring=scorer)
print('Bayes:\t', cv_gnb)









