#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import lightgbm as lgb
from sklearn import metrics
import seaborn as sns
from collections import defaultdict


# In[2]:


def score(Y_true, Y_pred):
    y_true = np.log1p(np.maximum(0, Y_true))
    y_pred = np.log1p(np.maximum(0, Y_pred))
    return 1 - np.mean((y_true-y_pred)**2) / np.mean((y_true-np.mean(y_true))**2)


def evaluate(gold_path, pred_path):
    gold = { x['doi']: x['citations'] for x in json.load(open(gold_path)) }
    pred = { x['doi']: x['citations'] for x in json.load(open(pred_path)) }
    y_true = np.array([ gold[key] for key in gold ])
    y_pred = np.array([ pred[key] for key in gold ])
    return score(y_true, y_pred)


# # 1. Preprocessing

# Load data

# In[3]:


with open('train-1.json', 'r') as f:
    content = f.read()
    train = json.loads(content)
    train = pd.DataFrame(train)
    
    
with open('test.json', 'r') as f:
    content = f.read()
    test = json.loads(content)
    test = pd.DataFrame(test)


# In[4]:


test['citations'] = np.nan


# In[5]:


columns = ['doi', 'title', 'abstract', 'authors', 'venue', 'year', 'references',
           'topics', 'is_open_access', 'fields_of_study', 'citations']
train = train[columns].copy()
test = test[columns].copy()


# There is a duplication in doi, remove it

# In[6]:


train[train.duplicated(subset=['doi'])]


# In[7]:


train[train.doi == '10.18653/v1/W18-6481']


# In[8]:


train.drop_duplicates(subset=['doi'], keep='first', inplace=True)


# Check missing value

# In[9]:


train.shape, test.shape


# In[10]:


train.isna().sum()


# In[11]:


train.year.fillna(train.year.mean(), inplace=True)


# In[12]:


test.isna().sum()


# In[13]:


train.set_index('doi', inplace=True)
test.set_index('doi', inplace=True)


# In[14]:


train.citations.min(), train.citations.max()


# In[15]:


train.citations.sort_values(ascending=False)


# In[16]:


train.head(3)


# # 2. Feature engineering

# In[17]:


sns.scatterplot(x='year', y='citations', data=train)


# There are 41 papers which have a very high citations, which is consider as outlier, we will remove them

# In[18]:


high = train[train.citations > 800].copy()
train.drop(high.index, axis=0, inplace=True)
print(high.shape)


# In[19]:


high.head(3)


# In[20]:


sns.scatterplot(x='year', y='citations', data=train, alpha=0.2)


# In[21]:


sns.scatterplot(x='references', y='citations', data=train, alpha=0.2)


# **A baseline**

# In[22]:


features = ['year', 'references', 'is_open_access']
X = train[features].copy()
y = train.citations


# In[23]:


cv_r2score = cross_val_score(KNeighborsRegressor(n_neighbors=5), X, y, cv=5, scoring='r2')
cv_r2score.mean(), cv_r2score.std()


# In[24]:


cv_r2score = cross_val_score(lgb.LGBMRegressor(), X, y, cv=5, scoring='r2')
cv_r2score.mean(), cv_r2score.std()


# In[25]:


X['num_author'] = train.authors.apply(len)


# In[26]:


cv_r2score = cross_val_score(lgb.LGBMRegressor(), X, y, cv=5, scoring='r2')
cv_r2score.mean(), cv_r2score.std()


# In[27]:


fields = set()
for index in train.index:
    fields_of_study = train.loc[index, 'fields_of_study']
    if not fields_of_study:
        continue
    for field in fields_of_study:
        fields.add(field)


# In[28]:


len(fields)


# In[29]:


field_data = train[['fields_of_study', 'citations']].copy()
field_data.fields_of_study.replace({None: '[]'}, inplace=True)
field_data.fields_of_study = field_data.fields_of_study.apply(lambda x: eval(x) if isinstance(x, str) else x)


# In[30]:


field_counter = Counter()
for index, row in field_data.iterrows():
    field_counter.update(row['fields_of_study'])


# In[31]:


field_counter = pd.Series(field_counter).sort_values()


# In[32]:


field_counter


# **Author feature**

# In[33]:


feature = 'authors'
author_data = train[[feature, 'citations']].copy()


# In[34]:


author_stats = defaultdict(list)
for index, row in author_data.iterrows():
    for author in row['authors']:
        author_stats[author].append(row['citations'])
author_stats = pd.Series(author_stats)


# In[35]:


author_stats.shape


# In[36]:


author_stats = pd.DataFrame(author_stats).rename({0: 'citation_list'}, axis=1)
author_stats['count_citation'] = author_stats.citation_list.apply(len)
author_stats.sort_values(by='count_citation', ascending=False, inplace=True)

threshold = 2
author_stats = author_stats[author_stats.count_citation >= threshold].copy()

author_stats['min_citation'] = author_stats.citation_list.apply(lambda cs: min(cs))
author_stats['max_citation'] = author_stats.citation_list.apply(lambda cs: max(cs))
author_stats['std_citation'] = author_stats.citation_list.apply(lambda cs: np.std(cs))
author_stats['mean_citation'] = author_stats.citation_list.apply(lambda cs: np.mean(cs))
author_stats['count_citation'] = author_stats.citation_list.apply(lambda cs: len(cs))
author_stats['median_citation'] = author_stats.citation_list.apply(lambda cs: np.median(cs))
author_stats = author_stats.reset_index().rename({'index': 'author'}, axis=1)


# In[37]:


author_stats.head(3)


# In[38]:


author_stats.shape


# In[39]:


len(author_data)


# In[ ]:





# In[40]:


author_citation_count = dict()
for index, row in author_stats.iterrows():
    author_citation_count[row['author']] = row['count_citation']


# In[41]:


train.authors = train.authors.apply(lambda x: sorted(x, key=lambda e: author_citation_count[e] if e in author_citation_count else 0))
test.authors = test.authors.apply(lambda x: sorted(x, key=lambda e: author_citation_count[e] if e in author_citation_count else 0))


# In[42]:


def build_author_feature(df):
    authors = df[['authors']].copy()
    num_authors = 3
    feature_columns = ['author', 'count_citation', 'min_citation', 'max_citation', 'std_citation', 'mean_citation', 'median_citation']

    for index in range(num_authors):
        column = 'author_{}'.format(index)
        authors[column] = authors['authors'].apply(lambda a: a[index] if index < len(a) else np.nan)
        authors = pd.merge(left=authors, right=author_stats[feature_columns], left_on=column, right_on='author', how='left')
        authors.rename({col: '{}_{}'.format(col, index) for col in feature_columns[1:]}, axis=1, inplace=True)
        authors.drop([column, 'author'], axis=1, inplace=True)
    authors.drop(['authors'], axis=1, inplace=True)
    authors.index = df.index
    return authors


# In[43]:


train_author_features = build_author_feature(train)


# In[44]:


train_author_features.shape, train.shape


# In[45]:


test_author_features = build_author_feature(test)


# In[46]:


test_author_features.shape, test.shape


# In[47]:


count_authors = train.authors.apply(lambda x: len(x)).sort_values()
count_authors[count_authors >= 3].shape


# In[48]:


X_train = train[features].copy()
X_train = pd.concat((X_train, train_author_features), axis=1)
X_train.fillna(0, inplace=True)


# In[49]:


X_test = test[features].copy()
X_test = pd.concat((X_test, test_author_features), axis=1)
X_test.fillna(0, inplace=True)


# In[50]:


X_train.shape, X_test.shape


# In[51]:


X_train.head(3)


# In[52]:


cv_r2score = cross_val_score(lgb.LGBMRegressor(), X_train, train.citations, cv=5, scoring='r2')
cv_r2score.mean(), cv_r2score.std()


# In[53]:


model = lgb.LGBMRegressor()
model.fit(X_train, train.citations)


# In[54]:


y_test_predict = model.predict(X_test)
ser = pd.Series(index=test.index, data=y_test_predict)
ser = ser.round().astype(int)


# In[55]:


def output_to_file(ser, filename):
    json_data = []
    for index in ser.index:
        json_data.append({
            'doi': index,
            'citations': int(ser[index])
        })


    with open(filename, 'w') as f:
        json.dump(json_data, f)


# In[56]:


output_to_file(ser, 'prediction-1.json')


# In[57]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# In[58]:


X_train.shape


# ~~~Python
# model = lgb.LGBMRegressor()
# nfeatures = X_train.shape[1]
# sfs = SFS(model,
#            k_features=nfeatures,
#            forward=True,
#            verbose=3,
#            scoring = 'neg_mean_squared_error',
#            cv = 3,
#            n_jobs=-1)
# sfs.fit(X_train, train.citations)
# 
# score = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
# plt.figure()
# plt.plot(score.index.tolist(), score['avg_score'],
#         marker = 'o', markevery = [np.argmax(score['avg_score'])],
#          markeredgecolor = 'red', markerfacecolor = 'red')
# plt.xticks(range(1, nfeatures + 1), range(1, nfeatures + 1))
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (accuracy)")
# plt.show()
# 
# selected = list(score.iloc[np.argmax(score['avg_score'])]['feature_names'])
# print('#  features selected (target encoded):', len(selected))
# selected
# 
# cv_r2score = cross_val_score(lgb.LGBMRegressor(), X_train[selected], train.citations, cv=5, scoring='r2')
# cv_r2score.mean(), cv_r2score.std()
# 
# model = lgb.LGBMRegressor()
# model.fit(X_train[selected], train.citations)
# y_test_predict = model.predict(X_test[selected])
# ser = pd.Series(index=test.index, data=y_test_predict)
# ser = ser.round().astype(int)
# output_to_file(ser, 'prediction-2.json')
# ~~~

# In[59]:


from sklearn.linear_model import LinearRegression


# ### Test different models

# In[61]:


for name, model in (
    ('LinearRegression', LinearRegression()),
    ('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=5)),
    ('LGBMRegressor', lgb.LGBMRegressor())
):
    cv_r2score = cross_val_score(model,    X_train, train.citations, cv=5, scoring='r2')
    print(name, 'mean:', cv_r2score.mean(), 'std:', cv_r2score.std())


# # 3. Hyperparameter tuning

# In[60]:


from sklearn.model_selection import GridSearchCV


# In[61]:


param_grid = {
    'boosting_type': ['gbdt'],
    'num_leaves': range(10, 31, 2),
    'max_depth': [-1] + list(range(3, 10, 2)),
    'n_estimators': [50, 100, 150, 200]
}


# In[62]:


grid = GridSearchCV(lgb.LGBMRegressor(), param_grid, cv=3, scoring='r2')
grid.fit(X_train, train.citations)


# In[63]:


grid.best_estimator_


# In[64]:


grid.best_params_


# In[65]:


model = grid.best_estimator_


# In[66]:


cv_r2score = cross_val_score(model, X_train, train.citations, cv=5, scoring='r2')
cv_r2score.mean(), cv_r2score.std()


# In[67]:


model.fit(X_train, train.citations)
y_test_predict = model.predict(X_test)
ser = pd.Series(index=test.index, data=y_test_predict)
ser = ser.round().astype(int)
output_to_file(ser, 'prediction-4.json')


# In[ ]:




