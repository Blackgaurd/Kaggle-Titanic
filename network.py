import numpy as np
import pandas as pd
import csv

np.random.seed(1)


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
        self.train = train_data.to_numpy().astype(float)

        # normalize data (standard score)
        # don't normalize "Survived" because it is expected values
        for i in range(1, self.train.shape[1]):
            self.train[:, i] = (self.train[:, i] - self.train[:, i].mean()) / self.train[:, i].std()

        print("Initiating other class variables...")
        # variables
        self.batch_size = batch_size
        self.alpha = learning_rate

        # layers
        size1 = 8
        size2 = 5
        size3 = 2

        self.w1 = np.random.rand(size1, self.train.shape[1] - 1)
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

    def softmax(self, x):
        return np.exp(x) / sum(np.exp(x))

    def get_batch(self):
        indices = np.random.choice(self.train.shape[0], self.batch_size, replace=False)
        choices = self.train[indices]

        inputs = choices.T[1:]
        expected = choices.T[0]

        return inputs, expected.astype(int)

    def one_hot(self, x):
        one_hot = np.zeros((x.size, 2))
        one_hot[np.arange(x.size), x] = 1
        return one_hot.T

    def for_prop(self, inputs):
        self.z1 = self.w1.dot(inputs) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = self.w2.dot(self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)

        self.z3 = self.w3.dot(self.a2) + self.b3
        self.a3 = self.softmax(self.z3)

    def back_prop(self, inputs, expected):
        # calculate errors
        y_hat = self.one_hot(expected)

        self.dz3 = self.a3 - y_hat
        self.dw3 = self.dz3.dot(self.a2.T) / self.batch_size
        self.db3 = np.sum(self.dz3) / self.batch_size

        self.dz2 = self.w3.T.dot(self.dz3) * self.sigmoid_d(self.z2)
        self.dw2 = self.dz2.dot(self.a1.T) / self.batch_size
        self.db2 = np.sum(self.dz2) / self.batch_size

        self.dz1 = self.w2.T.dot(self.dz2) * self.sigmoid_d(self.z1)
        self.dw1 = self.dz1.dot(inputs.T) / self.batch_size
        self.db1 = np.sum(self.dz1) / self.batch_size

    def update_params(self):
        self.w1 = self.w1 - self.alpha * self.dw1
        self.b1 = self.b1 - self.alpha * self.db1
        self.w2 = self.w2 - self.alpha * self.dw2
        self.b2 = self.b2 - self.alpha * self.db2
        self.w3 = self.w3 - self.alpha * self.dw3
        self.b3 = self.b3 - self.alpha * self.db3

    def gradient_descent(self, iterations):
        for i in range(1, iterations + 1):
            inputs, expected = self.get_batch()
            self.for_prop(inputs)
            self.back_prop(inputs, expected)
            self.update_params()

            if not i % 10:
                print(f"Iteration:\t{i}")
                predictions = self.get_predictions(self.a3)
                accuracy = self.get_accuracy(predictions, expected)
                print(f"Accuracy:\t{accuracy}\n")

    def get_predictions(self, outputs):
        return np.argmax(outputs, 0)

    def get_accuracy(self, predictions, expected):
        return np.sum(predictions == expected) / expected.size

    def make_submission(self, output_file_path, test_file_path="input/test.csv"):
        print("Reading and normalizing test data...")
        test = pd.read_csv(test_file_path)

        ids = test["PassengerId"].to_numpy()
        test = test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]]

        # fill in missing data
        test["Age"].fillna(int(test["Age"].mean()), inplace=True)
        test["Embarked"].fillna("S", inplace=True)

        # replace non-numerical values with numbers
        test["Sex"].replace({"male": 0, "female": 1}, inplace=True)
        test["Embarked"].replace({"S": 0, "C": 1, "Q": 2}, inplace=True)

        # to numpy
        test = test.to_numpy().astype(float)

        # normalize data (standard score)
        for i in range(test.shape[1]):
            test[:, i] = (test[:, i] - test[:, i].mean()) / test[:, i].std()

        # make predictions
        z1 = self.w1.dot(test.T) + self.b1
        a1 = self.sigmoid(z1)

        z2 = self.w2.dot(a1) + self.b2
        a2 = self.sigmoid(z2)

        z3 = self.w3.dot(a2) + self.b3
        a3 = self.softmax(z3)

        predictions = self.get_predictions(a3)

        rows = np.vstack((ids, predictions)).T.tolist()
        with open(output_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["PassengerId", "Survived"])
            writer.writerows(rows)

network = Network()
network.make_submission("test_sub.csv")