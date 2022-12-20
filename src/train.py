import configparser
import os
import pickle
import pandas as pd
import numpy as np

from utils import mymkdirs
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class Trainer():
    """
    Class dedicated for training sklearn models on Fashion MNIST Dataset.
    
    This class firstly loads a config file and the dataset.

    To launch training call .train method

    """
    def __init__(self) -> None:
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        train_set = np.array(pd.read_csv(self.config["DATA"]["train_image_path"]))
        test_set = np.array(pd.read_csv(self.config["DATA"]["test_image_path"]))
        self.X_train = train_set[:, 1:]
        self.y_train = train_set[:, 0]
        self.X_test = test_set[:, 1:]
        self.y_test = test_set[:, 0]
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        self.out_path = self.config["OUT"]["output_path"]
        mymkdirs(self.out_path)

    def save_model(self, clf, model_name):
        path = os.path.join(self.out_path, model_name + '.sav')
        with open(path, 'wb') as f:
            pickle.dump(clf, f)
        return os.path.isfile(path)

    def svm(self, benchmark=True):
        clf = SVC()
        clf.fit(self.X_train, self.y_train)
        
        if benchmark:
            pred = clf.predict(self.X_test)
            score = accuracy_score(self.y_test, pred)
            print('Accuracy:', score)
            return score
        return

    def logreg(self, benchmark=False):
        clf = LogisticRegression(max_iter = self.config["LOGREG"]['max_iter'])
        clf.fit(self.X_train, self.y_train)
        
        if benchmark:
            pred = clf.predict(self.X_test)
            score = accuracy_score(self.y_test, pred)
            print('Accuracy:', score)
            return score
        return

    def train(self, model_name, benchmark=False):
        if model_name not in self.__dir__():
            print('Wrong model name')
        else:
            model = getattr(self, model_name)
            model(benchmark)
            return self.save_model(model, model_name)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train('svm')
    trainer.train('logreg')

