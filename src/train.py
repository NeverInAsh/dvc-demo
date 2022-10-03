import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

fruits = pd.read_table('./dvc-demo/dvc-demo/data/raw/fruit_data_with_colors.txt')
fruits.head()

# create a mapping from fruit label value to fruit name to make results easier to interpret
look_up_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))
look_up_fruit_name

# Split the data into training and testing
X = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# Create classifier object
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

# Train the classifier using the training data
knn.fit(X_train, y_train)

# Estimate the accuracy of the classifier on future data, using the test data
with open(os.path.join('./dvc-demo/dvc-demo/results.txt'), 'w') as f:   
    sys.stdout = f
    print(knn.score(X_test, y_test))

pickle.dump(knn,open( "./dvc-demo/dvc-demo/model_artificats/model_knn.pickle",'wb') )



