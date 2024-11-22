# %%
import requests as r
from bs4 import BeautifulSoup
import csv
import copy
import misc
import data_processing
from importlib import reload
import pandas as pd
import numpy as np
reload(data_processing)
reload(misc)
pd.set_option('display.max_columns', 50)




# %% [markdown]
# ## Remove Rows with too Many Missing Data

# %%
data = data_processing.read_data("raw_data.csv")

data = list(filter(lambda d: len(d.keys()) == len(list(filter(None, list(d.values())))) , data))
with open('training_data_clean.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=list(data[0].keys()))
    writer.writeheader()
    writer.writerows(data)
  



# %% [markdown]
# ## Load and Clean Data Using Pandas (Remove Bad Characters in Column Titles, Remove Population Column)

# %%
import pandas as pd
import re
with open('training_data_clean.csv', 'r') as file:
    csv_reader = csv.DictReader(file)
    data = list(csv_reader)

df = pd.DataFrame(data)


columns = list(df.columns)
columns = list(map(lambda x: re.sub(r"^\s", "", x), columns))
columns = list(map(lambda x: re.sub(r"^\s", "", x), columns))
columns = list(map(lambda x: re.sub(r"\s$", "", x), columns))
columns = list(map(lambda x: re.sub(r"\s$", "", x), columns))
columns = list(map(lambda x: re.sub(r"\s?.?/$", "", x), columns))
df.columns = columns
population = df[columns[-2]]
df.drop(df[columns[-2]], axis = 1, inplace= True)
#df[columns[-2]] = population.iloc[:,0]
df

# %% [markdown]
# ## Adjust for inflation

# %%
for i in df.index:
    try:
        year = int(df.iloc[i,1])
        df.iloc[i, 4:] = df.iloc[i, 4:].map(lambda x: int(x) * misc.cpi_dict[year])
    except Exception as e:
        print(e)
        pass
df

# %% [markdown]
# ## Add Popultation Column Back

# %%
df[columns[-2]] = population.iloc[:,0]
df

# %% [markdown]
# ## Normalize Data

# %%
for column in df.columns:
    try:
        if column != "Year":
            df[column] = df[column].map(lambda x: int(x))
            x_max = max(list(df[column]))
            x_min = min(list(df[column]))
            df[column] = df[column].map(lambda x: (x - x_min)/(x_max - x_min))
    except Exception as e:
        print(e)
df.to_csv("training_data_clean.csv")

# %% [markdown]
# ## PCA for Exploratory Data Analysis

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

target, data = data_processing.read_data("training_data_clean.csv", mode = "List")
print(data.shape)

pca = PCA()
pca.fit(data)
#print(pca.explained_variance_ratio_)
transformed = pca.transform(data)
xs=transformed[:,0]
ys=transformed[:,1]
zs =transformed[:,2]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
fig.patch.set_facecolor('white')

target = np.array(target)
indices = np.where(target == 'D')
ax.scatter(xs[indices], ys[indices], zs[indices], c="Blue", s=0.05, alpha = 0.1)
indices = np.where(target == 'R')
ax.scatter(xs[indices], ys[indices], zs[indices], c="Red", s=0.05, alpha = 0.1)
# for loop ends
plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

target, data = data_processing.read_data("training_data_clean.csv", mode = "List")
print(data.shape)

pca = PCA()
pca.fit(data)
#print(pca.explained_variance_ratio_)
transformed = pca.transform(data)
xs=transformed[:,0]
ys=transformed[:,1]
zs =transformed[:,2]


fig = plt.figure(figsize=(10, 10)) 
Axes3D(fig) 
ax = fig.add_subplot(projection='3d')
fig.patch.set_facecolor('white')

target = np.array(target)
indices = np.where(target == 'D')
ax.scatter(xs[indices], ys[indices], zs[indices], c="Blue", s=0.05, alpha = 0.3)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(-0.3, 0.3)
indices = np.where(target == 'R')
ax.scatter(xs[indices], ys[indices], zs[indices], c="Blue", s=0.05, alpha = 0.3)

plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()


