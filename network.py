import numpy as np
import pandas as pd


class Network:
    def __init__(self, train_file_path="input/train.csv", batch_size=15, learning_rate=0.1):
        # read and normalize train data
        print("Reading and normalizing training data...")
        train_data = pd.read_csv(train_file_path)

        # remove unnecessary columns
        train_data = train_data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

        # fill in missing data
        train_data["Age"].fillna(int(train_data["Age"].mean()), inplace=True)
        train_data["Embarked"].fillna("S", inplace=True)

        # replace non-numerical values with numbers
        train_data["Sex"].replace({"male": 0, "female": 1}, inplace=True)
        train_data["Embarked"].replace({"S": 0, "C": 1, "Q": 2}, inplace=True)

        # to numpy
        self.train = train_data.to_numpy()

        # normalize data (standard score)
        for i in range(self.train.shape[1]):
            self.train[:, i] = (self.train[:, i] - self.train[:, i].mean()) / self.train[:, i].std()

        print("Initiating other class variables...")
        # variables
        self.batch_size = batch_size
        self.alpha = learning_rate

        # layers
        size1 = 8
        size2 = 5
        size3 = 1

        self.w1 = np.random.rand(size1, self.train.shape[0])
        self.b1 = np.random.randn(size1, 1)

        self.w2 = np.random.rand(size2, size1)
        self.b2 = np.random.randn(size2, 1)

        self.w3 = np.random.rand(size3, size2)
        self.b3 = np.random.randn(size3, 1)

        (
            self.z1, self.a1,
            self.z2, self.a2,
            self.z3, self.a3,
            self.dz1, self.dw1, self.bd1,
            self.dz2, self.dw2, self.bd2,
            self.dz3, self.dw3, self.bd3
        ) = [None for i in range(15)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def for_prop(self):
        pass

    def back_prop(self):
        pass

    def get_batch(self):
        pass

    def gradient_descent(self):
        pass


network = Network()
print(network.train[47])
print(network.train.shape)
print("done")