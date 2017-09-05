"""
Base class for machine learning classifier. It is an abstract class defining
methods need to be implemented in subclasses.

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

import numpy as np
import dill, logging, sys
from datetime import datetime
from abc import ABCMeta, abstractmethod
from h3mlcore.utils.H3Logging import setup_logging
from h3mlcore.io.Preprocessor import Preprocessor



class H3BaseActor(object):
    """Abstract class for all H3 machine learnign models
       All models should inherit from this super class.

    Args:
      preprocessor: a list of preprocessors
      max_epoch: maximal epochs to train
      epoch_status: #epochs, if 0 means the classifier is never trained
      log_file: file path to save logging
      log_level: logging level, default logging.INFO

    Returns:
      Instance of H3BaseActor

    """

    __metaclass__ = ABCMeta

    def __init__(self,
                preprocessor=None,
                max_epoch=10,
                epoch_status=0,
                messager=None,
                log_file=None,
                log_config='logging.yaml',
                log_level=logging.INFO):

        self.max_epoch = max_epoch
        self.epoch_status = epoch_status
        self.preprocessor = preprocessor
        self.messager = messager
        self.logger = setup_logging(logging_config=log_config, level=log_level)



    @abstractmethod
    def fit(self, X, y=None):
        """fit model on a dataset X and labels y
        This is a scikit-learn style fit function.

        Args:
        X: training dataset with N rows and M cols
        y:  (Default value = None) training labels 

        """
        # train on a training dataset
        self.logger.info(self.__name__ + ' is trained on {:d} samples with {:d} features.'.format(X.shape[0], X.shape[1]))
        pass


    @abstractmethod
    def partial_fit(self, X, y=None):
        """fit model incrementally on a dataset X and labels y
        This is a scikit-learn style fit function.

        Args:
          X: training dataset with N rows and M cols
          y:  (Default value = None) training labels 

        """
        # update model on a minibatch
        self.logger.info(self.__name__ +
                        ' is updated on dataset with {:d} samples and {:d} features.'. \
                        format(X.shape[0], X.shape[1]))
        pass


    @abstractmethod
    def predict(self, Xtt):
        """Model predicts on test dataset Xtt

        Args:
          Xtt: testing dataset with N rows and M cols

        """
        # predict outputs for test dataset
        self.logger.info(self.__name__ + ' predicts on {:d} samples.'.format(Xtt.shape[0]))
        pass

    @abstractmethod
    def decision_function(self, Xtt):
        """decision scores on test dataset 

        Args:
        Xtt: 

        """
        # predict decision score on test dataset
        self.logger.info(self.__name__ + ' predicts decision scores on {:d} samples.'.format(Xtt.shape[0]))

    @abstractmethod
    def save(self, path):
        """save the actor on disk

        Args:
          path: file path to save the model

        """
        pass

    def add_preprocessor(self, pc):
        """Append additional preprocessor to the list of preprocessor in this classifier.

        Args:
        pc: an instance of preprocessor

        """

        if isinstance(pc, Preprocessor):
            # append a new preprocessor
            self.preprocessor.append(pc)
        else:
            self.logger.error('Invalid preprocessor! exit!')


    def prepare_data(self, data_blocks, restart=False):
        """prepare a trainable dataset from a list data blocks each of which is processable
        by its preprocessor accordingly. Processed data blocks are concatenated as a bigger trainable dataset.

        Args:
        data_blocks: a list of data blocks
        restart:  (Default value = False)

        Returns:
        A nxd trainable ndarray, d = sum(feature sizes of data blocks)

        """

        begin = True
        if self.preprocessor is not None:
            nrows = 0
            if type(self.preprocessor) is not list:
                self.preprocessor = [self.preprocessor]
            if type(data_blocks) is not list:
                data_blocks = [data_blocks]
            if len(self.preprocessor) != len(data_blocks):
                self.logger.error('Num. of data blocks do not align with num. of preprocessors in classifer.')
                sys.exit()
            for pc, block in zip(self.preprocessor, data_blocks):
                if len(block) == 0:
                    # empty data block
                    pc._FEATURE_NAMES = []
                    pc._FEATURE_SIZE = 0
                    continue
                if begin:
                    output = pc.run(block, restart=restart)
                    nrows = output.shape[0]
                    begin = False
                else:
                    cur_output = pc.run(block, restart=restart)
                    if cur_output.shape[0] != nrows:
                        self.logger.error('Preprocessor {:s} does not align with previous data block dimensions'.format(pc.__name__))
                        sys.exit(0)
                    else:
                        output = np.c_[output, cur_output]
            return output
        else:
            self.logger.warn('No preprocessor is found in this classifier, data blocks are directly concatenated.')
            output = data_blocks[0]
            for block in data_blocks[1:]:
                output = np.c_[output, block]
            return output


    def visualize(self, **kwargs):
        """visualize the classifier.

        Args:
        **kwargs: 

        Returns:

        """
        pass
