from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k

data = fetch_movielens(min_rating=5.0)

model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

precision_at_k(model, data['test'], k=5).mean()

#%%

import pandas as pd
import os
os.chdir(r'/Users/ganeshbade/Downloads/BRE-master/data')
df = pd.read_csv('candy.csv')

z = df.sample(5)

df['item'].unique().shape # 142 unique items
df['user'].unique().shape # 2531 unique users

# LightFM model require a user-item matrix where each user is a row and each
# item is column. Accordingly, need to transform candy pandas dataframe data coo_matrix 
#https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2Frecommendation-systems%2Fhow-user-based-collaborative-filtering-works-netflix-movie-recommendation-simulation-96f0c40fae21&psig=AOvVaw10Pb61L12GBb1zfQtP9w6i&ust=1624702229577000&source=images&cd=vfe&ved=0CAoQjRxqFwoTCIjfrdHFsvECFQAAAAAdAAAAABAD

# It is possible to create sparse matrix using pivot transform 
# but w'll create a sparse user-item matrix with the coo_matrix
# function from scipy.sparse

# Advantages of the COO format
# facilitates fast conversion among sparse formats

# permits duplicate entries (see example)

# very fast conversion to and from CSR/CSC formats

# Disadvantages of the COO format
# does not directly support:
# arithmetic operations
# slicing

# Intended Usage
# COO is a fast format for constructing sparse matrices

# Once a matrix has been constructed, convert to CSR or CSC format for fast arithmetic and matrix vector operations

# By default when converting to CSR or CSC format, duplicate (i,j) entries will be summed together. This facilitates efficient construction of finite element matrices and the like. (see example)

from scipy.sparse import coo_matrix
import numpy as np

ratings = np.array(df['review']) #data
users = np.array(df['user']) #rows
items = np.array(df['item']) #col

# Label encode users and items string data
from sklearn.preprocessing import LabelEncoder

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

u = user_encoder.fit_transform(users)
i = item_encoder.fit_transform(items)
lenu = len(np.unique(u))
leni = len(np.unique(i))

matrix = coo_matrix((ratings, (u, i)), shape=(lenu, leni), dtype=np.int32)
# <2531x142 sparse matrix of type '<class 'numpy.int32'>'
# 	with 17234 stored elements in COOrdinate format>

matrix.shape # (2531, 142)
matrix.todense()

#%%

from lightfm.cross_validation import random_train_test_split

train, test = random_train_test_split(matrix, test_percentage=0.2)

#%%

model = LightFM(random_state=42)
model.fit(train)

#%%

# Evaluate model performamnce with precision_at_k

precision_at_k(model, test, k=10)
precision_at_k(model, test, k=10).mean()
precision_at_k(model, train, k=10).mean()

from lightfm.evaluation import auc_score
auc_score(model, train).mean()
auc_score(model, test).mean()

#%%

all_candy_ids = list(range(len(item_encoder.classes_)))
pd.Series(all_candy_ids).to_csv('all_candy_ids.csv')
user_id = 2528
pred = model.predict(user_id, all_candy_ids)

import pandas as pd
candies = pd.DataFrame(zip(item_encoder.classes_, pred), columns=['item', 'prediction']).sort_values(by='prediction', axis=0, ascending=False)
pd.Series(item_encoder.classes_).to_csv('item_encoder.classes_.csv', index=False)

#%%
import pickle

filename = 'lightfm_reco.pkl' 
with open('lightfm_reco.pkl', 'wb') as f:
    pickle.dump(model, f)
f.close()

#%%

# Productionalized version

import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
os.chdir(r'/Users/ganeshbade/Downloads/BRE-master/data')

infile = open('lightfm_reco.pkl', 'rb')
model = pickle.load(infile)
infile.close()

user_id = int(input("Enter user_id \n"))
# user_id = 2528
all_candy_ids = pd.read_csv('all_candy_ids.csv').iloc[:, 0].tolist()

pred = model.predict(user_id, all_candy_ids)
recommend = pd.DataFrame(zip(pd.read_csv('item_encoder.classes_.csv').iloc[:, 0].to_numpy(), pred), columns=['item', 'prediction']).sort_values(by='prediction', axis=0, ascending=False)
recommend.head()



