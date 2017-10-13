'''
A simple text classifier uses SGD logistic regression for multi-class classification.
The classifier is implemented in mxnet.

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''


import numpy as np
import sys
import dill
from sklearn.linear_model import SGDClassifier
from datetime import datetime
from h3mlcore.models.H3BaseActor import H3BaseActor
from h3mlcore.io.Preprocessor import *


class H3TextClassifier(H3BaseActor):

    __name__ = 'H3TextClassifier'

    """
    TextClassifier main class: A Classifier used to classify documents
    """

    def __init__(self,
                 loss='log',
                 penalty='l2',
                 l1_ratio=0,
                 alpha=1e-4,
                 shuffle=True,
                 class_weight=None,
                 *args,
                 **kwargs):
        """
        Constructor
        :return:
        """

        super(H3TextClassifier, self).__init__(*args, **kwargs)
        # training params / optimal training params after model selection
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.shuffle = shuffle
        self.l1_ratio = l1_ratio
        if self.preprocessors is None:
            # the default preprocessor is a Tfidf vectorizer
            workflow_1 = [{'worker': 'TfidfVectorizer',
                           'params': {'encoding': 'utf-8'}}]
            self.preprocessors = [Preprocessor(workflow_1, ['n-grams']), ]

        # default classifier is SGD logistic regressor
        self.classifier = SGDClassifier(loss=self.loss,
                                        penalty=self.penalty,
                                        alpha=self.alpha,
                                        l1_ratio=self.l1_ratio,
                                        shuffle=self.shuffle,
                                        class_weight=class_weight)

    def fit(self, X, y=None):
        """
        Training method for build the recommendation system model
        :param trainset: Nxd training data vectors
        """
        self.classifier.fit(X, y)
        self.epoch_status += 1
        # this only print logs
        super(H3TextClassifier, self).fit(X, y)

    def partial_fit(self, X, y=None, classes=None):
        """
        Partial training method to dynamically factor in new data into the current model
        :param partial_trainset: Data vectors for subset of data to be trained
        :param labels: Output vector for training data
        :param classes: Array of unique outputs
        :return:
        """

        self.classifier.partial_fit(X, y, classes=classes)
        self.epoch_status += 1
        super(H3TextClassifier, self).partial_fit(X, y)

    def predict(self, Xtt):
        """
        Classification of which template to use for replying an email
        :param testset: Mxd test data vectors
        :return: Template labels, e.g., 1,2,..., n (integer)
        """

        if self.epoch_status < 1:
            self.logger.error(__name__ + ' is not trained yet, exit!')
            sys.exit()
        super(H3TextClassifier, self).predict(Xtt)
        return self.classifier.predict(Xtt)

    def decision_function(self, Xtt):
        super(H3TextClassifier, self).decision_function(Xtt)
        return self.classifier.decision_function(Xtt)

    def save(self, path):
        with open(path, 'w') as fd:
            dill.dump(self, fd)   
            self.logger.info("Model %s saved in %s" % (self.__name__, path))

    def visualize(self):
        pass


if __name__ == '__main__':
    """
    User Guide
    """
    import logging
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split, learning_curve, validation_curve, ShuffleSplit
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from h3mlcore.utils.DatasetHelper import load_snp17
    import matplotlib.pylab as plt

    # Load data
    snippets, labels, _ = load_snp17('/Users/hxiao/repos/h3lib/h3db/snp17/train/answer_snippets_coded.csv',
                                     save_path='/Users/hxiao/repos/CognutSecurity/webdemo/datasets/snp17.p')
    clf = H3TextClassifier(penalty='l1', l1_ratio=0.6, log_level=logging.ERROR)
    X, y = clf.prepare_data(data_blocks = [snippets], y_blocks=[np.array(labels)])
    # split the dataset as train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=30)

    # epoch = 10
    # batch_size = 20
    # steps = 100
    # tr_size = X_train.shape[0]

    # accuracies = []
    # precisions = []
    # recalls = []
    # for n in range(epoch):
    #     X_train, y_train = shuffle(X_train, y_train)
    #     for i in xrange(steps):
    #         batch_start = i * batch_size % tr_size
    #         batch_end = min(batch_start + batch_size, tr_size)
    #         X_train_part = X_train[batch_start:batch_end]
    #         y_train_part = y_train[batch_start:batch_end]

    #         # Partial train the segment of data, classify test data and compare to actual values using various data analysis methods
    #         clf.partial_fit(X=X_train_part, y=y_train_part,
    #                         classes=np.unique(y))
    #         y_pred = clf.predict(X_test)
    #         accuracy = accuracy_score(y_test, y_pred)
    #         f1 = f1_score(y_test, y_pred, average=None)
    #         precision = precision_score(y_test, y_pred, average='weighted')
    #         recall = recall_score(y_test, y_pred, average='weighted')

    #         accuracies.append(accuracy)
    #         precisions.append(precision)
    #         recalls.append(recall)
    #         print 'Acc: %f | Prec: %f | Recall: %f' % (accuracy, precision, recall)

    cvfolds = ShuffleSplit(n_splits=5, test_size=0.2)
    tr_sizes, tr_scores, tt_scores = learning_curve(estimator=clf.classifier, n_jobs=1, X=X, y=y, cv=3,
                                                    train_sizes=np.linspace(0.1, 1, 5), scoring='accuracy', verbose=1)
    plt.plot(tr_sizes, tr_scores, 'r-', label='training scores')
    plt.plot(tr_sizes, tt_scores, 'b-', label='testing scores')
    plt.show()