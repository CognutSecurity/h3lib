"""
Base class for machine learning classifier. It is an abstract class defining
methods need to be implemented in subclasses.

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

import numpy as np
import dill
import logging
import sys
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
                 preprocessors=None,
                 max_epoch=10,
                 epoch_status=0,
                 messager=None,
                 log_file=None,
                 log_config='logging.yaml',
                 log_level=logging.INFO):

        self.max_epoch = max_epoch
        self.epoch_status = epoch_status
        self.preprocessors = preprocessors
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
        self.logger.info(
            self.__name__ + ' is trained on {:d} samples with {:d} features.'.format(X.shape[0], X.shape[1]))
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
                         ' is updated on dataset with {:d} samples and {:d} features.'.
                         format(X.shape[0], X.shape[1]))
        pass

    @abstractmethod
    def predict(self, Xtt):
        """Model predicts on test dataset Xtt

        Args:
          Xtt: testing dataset with N rows and M cols

        """
        # predict outputs for test dataset
        self.logger.info(
            self.__name__ + ' predicts on {:d} samples.'.format(Xtt.shape[0]))
        pass

    @abstractmethod
    def decision_function(self, Xtt):
        """decision scores on test dataset 

        Args:
        Xtt: 

        """
        # predict decision score on test dataset
        self.logger.info(
            self.__name__ + ' predicts decision scores on {:d} samples.'.format(Xtt.shape[0]))

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

    def prepare_data(self, data_blocks, y_blocks=None, restart=False):
        """prepare a trainable dataset from a list data blocks each of which is processable
        by its preprocessor accordingly. Processed data blocks are concatenated as a bigger trainable dataset.

        Args:
        data_blocks: a list of data blocks
        y_blocks: (default: None) a list of labels blocks
        restart:  (Default value = False)

        Returns:
        data_X: (ndarray) A nxd trainable ndarray, d = sum(feature sizes of data blocks)
        labels: (ndarray) An ndarray of n labels

        """

        begin = True
        fake_y = False
        # prepare list of blocks
        if type(data_blocks) is not list:
            data_blocks = [data_blocks]
        if not y_blocks:
            fake_y = True
            y_blocks = [np.zeros(len(block)) for block in data_blocks]
        elif type(y_blocks) is not list:
            y_blocks = [y_blocks]

        if self.preprocessors is not None:
            nrows = 0
            if type(self.preprocessors) is not list:
                self.preprocessors = [self.preprocessors]
            if len(self.preprocessors) != len(data_blocks):
                self.logger.error(
                    'You need same size preprocessors for your datasets.')
                sys.exit()

            for pc, block, y in zip(self.preprocessors, data_blocks, y_blocks):
                if len(block) == 0:
                    # empty data block
                    pc._FEATURE_NAMES = []
                    pc._FEATURE_SIZE = 0
                    pc._SAMPLE_SIZE = 0
                    continue
                if begin:
                    output_x, output_y = pc.run(block, y, restart=restart)
                    nrows = output_x.shape[0]
                    begin = False
                else:
                    cur_output_x, cur_output_y = pc.run(
                        block, y, restart=restart)
                    if cur_output_x.shape[0] != nrows:
                        self.logger.error(
                            'Preprocessor {:s} does not align with previous data block dimensions'.format(pc.__name__))
                        sys.exit(0)
                    else:
                        output_x = np.c_[output_x, cur_output_x]
                        output_y = np.c_[output_y, cur_output_y]
        else:
            self.logger.warn(
                'No preprocessor is found in this classifier, data blocks are directly concatenated.')
            output_x = data_blocks[0]
            output_y = y_blocks[0]

            for block, y in zip(data_blocks[1:], y_blocks[1:]):
                output_x = np.c_[output_x, block]
                output_y = np.c_[output_y, y]

        if fake_y:
            return output_x, None
        else:
            return output_x, output_y

    def visualize(self, **kwargs):
        """visualize the classifier.

        Args:
        **kwargs: 

        Returns:

        """
        pass
