import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from utils.utils import get_LBP_describe, plot_validation, plot_image, plot_cm

class LBPClassifier:
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.imgs = glob(os.path.join(self.train_dir, '*.jpg'))
        self.imgs_test = glob(os.path.join(self.test_dir, '*.jpg'))
        self.labels = None
        self.data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.parameters = None
        self.grid_search = None
        self.predictions = None

    def extract_features(self):
        self.labels = []
        self.data = []
        for imgnm in self.imgs:
            gray, lbp, desc = get_LBP_describe(imgnm)
            self.labels.append(imgnm.split(os.path.sep)[-1].split('.')[0].split()[0])
            self.data.append(desc)

    def train_test_split(self, test_size=0.3):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data,
                                                                                self.labels,
                                                                                test_size=test_size)

    def train_model(self):
        self.model = SVC()
        self.parameters = {'C': np.arange(0.01, 100, 10),
                           'gamma': np.arange(100, 0.001, -10),
                           'kernel': ['linear', 'rbf']}
        self.grid_search = GridSearchCV(estimator=self.model,
                                        param_grid=self.parameters,
                                        scoring='accuracy',
                                        verbose=2,
                                        n_jobs=-1)
        self.grid_search.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        self.predictions = self.grid_search.predict(self.x_test)
        print('Accuracy Score:', accuracy_score(self.y_test, self.predictions))
        print(classification_report(self.y_test, self.predictions))
        plot_cm(self.y_test, self.predictions)

    def test_model(self):
        for imgnm_teste in self.imgs_test:
            _, _, hist = get_LBP_describe(imgnm_teste)
            prediction = self.grid_search.predict(hist.reshape(1, -1))
            print(prediction[0])

    def plot_validation_curve(self):
        plot_validation(self.model, self.x_train, self.y_train)


lbp_classifier = LBPClassifier('data\\train', 'data\\test')
lbp_classifier.extract_features()
lbp_classifier.train_test_split()
lbp_classifier.train_model()
lbp_classifier.evaluate_model()
lbp_classifier.test_model()
lbp_classifier.plot_validation_curve()
