import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

from sklearn import datasets
from sklearn.decomposition import PCA

#imports
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix
class KNN:

    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.distance_matrix = None
    
    def train(self):
        self.distance_matrix = distance_matrix(self.X_train, self.y_train, k)
    

    def predict(self, example):
        return 

    def get_error(self, predicted, actual):
        return sum(map(lambda x : 1 if (x[0] != x[1]) else 0, zip(predicted, actual))) / len(predicted)

    def test(self, test_input, labels):
        actual = labels
        predicted = (self.predict(test_input))
        print("error = ", self.get_error(predicted, actual))

# Add the dataset here

iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Split the data 70:30 and predict.
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size = 0.3, train_size = 0.7) 

# create a new object of class KNN
k = KNN(3, x_train, y_train)
print(k.X_train)
print(k.k)
print(k.y_train)
k.train()

# plot a boxplot that is grouped by Species. 
# You may have to ignore the ID column

# predict the labels using KNN

# use the test function to compute the error
