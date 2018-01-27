'''
A lightened CNN model used for image classification. Model details can be found in: 
[1] Wu X, He R, Sun Z. A Lightened CNN for Deep Face Representation[J]. arXiv preprint arXiv:1511.02683, 2015. 

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

import mxnet as mx
import numpy as np
import os
import pickle
import logging
import yaml
import logging.config
from H3BaseActor import H3BaseActor


class H3LightenedCNN(H3BaseActor):
    """A lightened CNN model [1]
    """

    def __init__(self,
                 num_classes=5000,
                 model_config=None,
                 logging_root_dir='logs/',
                 logging_config='../logging.yaml',
                 verbose=False
                 ):

        # setup logging
        try:
            # logging_root_dir = os.sep.join(__file__.split('/')[:-1])
            logging_path = logging_root_dir + self.__class__.__name__ + '/'
            if not os.path.exists(logging_path):
                os.makedirs(logging_path)
            logging_config = yaml.safe_load(open(logging_config, 'r'))
            logging_config['handlers']['info_file_handler']['filename'] = logging_path + 'info.log'
            logging_config['handlers']['error_file_handler']['filename'] = logging_path + 'error.log'
            logging.config.dictConfig(logging_config)
        except IOError:
            logging.basicConfig(level=logging.INFO)
            logging.warning(
                "logging config file: %s does not exist." % logging_config)
        finally:
            self.logger = logging.getLogger('default')

        # setup training parameters
        if not model_config:
            # use default setting
            model_config = {
                'data': {
                    'batch_size': 384,
                    'scale': 1. / 255,
                    'shape': (1, 96, 96),
                },
                'train': {
                    'gpus': [0, 1, 2, 3, 4],
                    'cpus': [0, 1, 2, 3, 4],
                    'num_epoch': 200,
                    'learning_rate': .05,
                    'weight_decay': 5e-4,
                    'momentum': .9,
                    'model_save_prefix': 'lightened_cnn/lightened_cnn',
                    'model_load_prefix': 'lightened_cnn',
                    'model_start_epoch': 0,
                    'kv_store': 'local',
                },
                'eval': {
                    'metric': 'topk'
                },
            }
        else:
            try:
                model_config = yaml.load(open(model_config))
            except ValueError as e:
                self.logger.error(
                    "Model config file is not a valid YAML file. exit!")
                sys.exit()
        self.model_config = model_config
        # setup devices from ctx
        devs = [mx.cpu(i) for i in self.model_config['train']['cpus']] \
            if self.model_config['train']['gpus'] is None \
            else [mx.gpu(i) for i in self.model_config['train']['gpus']]
        self.symbol = self._build_model(num_classes)
        checkpoint = mx.callback.do_checkpoint(
            self.model_config['train']['model_save_prefix'])
        kv = mx.kvstore.create(self.model_config['train']['kv_store'])
        arg_params = None
        aux_params = None
        if self.model_config['train']['model_start_epoch'] > 0:
            _, arg_params, aux_params = mx.model.load_checkpoint(
                self.model_config['train']['model_load_prefix'], self.model_config['train']['model_start_epoch'])
        # this defines a FFNN model
        self.model = mx.module.Module(
            ctx=devs,
            symbol=self.symbol,
            arg_params=arg_params,
            aux_params=aux_params,
            num_epoch=model_config['train']['num_epoch'],
            learning_rate=self.model_config['train']['learning_rate'],
            momentum=model_config['train']['momentum'],
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))
        if self.model_config['eval']['metric'] == 'acc':
            self.metric = mx.metric.Accuracy()
        elif self.model_config['eval']['metric'] == 'cross-entropy':
            self.metric = mx.metric.CrossEntropy()
        elif self.model_config['eval']['metric'] == 'topk':
            self.metric = mx.metric.TopKAccuracy(top_k=3)

    def _build_model(self, num_classes):

        def _group(data, num_r, num, kernel, stride, pad, layer):
            if num_r > 0:
                conv_r = mx.symbol.Convolution(
                    data=data, num_filter=num_r, kernel=(1, 1), name=('conv%s_r' % layer))
                slice_r = mx.symbol.SliceChannel(
                    data=conv_r, num_outputs=2, name=('slice%s_r' % layer))
                mfm_r = mx.symbol.maximum(slice_r[0], slice_r[1])
                conv = mx.symbol.Convolution(
                    data=mfm_r, kernel=kernel, stride=stride, pad=pad, num_filter=num, name=('conv%s' % layer))
            else:
                conv = mx.symbol.Convolution(
                    data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num, name=('conv%s' % layer))
            slice = mx.symbol.SliceChannel(
                data=conv, num_outputs=2, name=('slice%s' % layer))
            mfm = mx.symbol.maximum(slice[0], slice[1])
            pool = mx.symbol.Pooling(data=mfm, pool_type="max", kernel=(
                2, 2), stride=(2, 2), name=('pool%s' % layer))
            return pool

        data = mx.symbol.Variable(name="data")
        pool1 = _group(data, 0, 96, (5, 5), (1, 1), (2, 2), str(1))
        pool2 = _group(pool1, 96, 192, (3, 3), (1, 1), (1, 1), str(2))
        pool3 = _group(pool2, 192, 384, (3, 3), (1, 1), (1, 1), str(3))
        pool4 = _group(pool3, 384, 256, (3, 3), (1, 1), (1, 1), str(4))
        pool5 = _group(pool4, 256, 256, (3, 3), (1, 1), (1, 1), str(5))
        flatten = mx.symbol.Flatten(data=pool5)
        fc1 = mx.symbol.FullyConnected(
            data=flatten, num_hidden=512, name="fc1")
        slice_fc1 = mx.symbol.SliceChannel(
            data=fc1, num_outputs=2, name="slice_fc1")
        mfm_fc1 = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
        drop1 = mx.symbol.Dropout(data=mfm_fc1, p=0.7, name="drop1")
        fc2 = mx.symbol.FullyConnected(
            data=drop1, num_hidden=num_classes, name="fc2")
        softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
        return softmax

    def step(self, data_batch):
        """Feed one data batch from data iterator to train model

        This function is called when we feed one data batch to model to update parameters.
        it can be used in train_epochs.
        See also: train_epochs.

        Args:
          data_batch (mx.io.DataBatch): a data batch matches the model definition
        """
        self.model.forward(data_batch=data_batch)
        metric = mx.metric.CrossEntropy()
        metric.update(data_batch.label, self.model.get_outputs())
        self.logger.debug('train step %s: %f' % (metric.get()))
        self.model.backward()
        self.model.update()

    def train_epochs(self, train_data,
                     eval_data=None,
                     num_epochs=10,
                     ):
        """Train model for many epochs with training data.

        The model will be trained in epochs and possibly evaluated with validation dataset.

        Args:
          train_data: Training data iterator
          eval_data: Validation data iterator
          num_epochs (int): Number of epochs to train
        """

        for e in range(num_epochs):
            train_data.reset()
            for batch in train_data:
                self.step(data_batch=batch)
            if eval_data:
                eval_data.reset()
                self.model.score(
                    eval_data, self.model_config['eval']['metric'])
                self.logger.info("Training epoch %d -- Evaluate %s: %f"
                                 % (e + 1, self.model_config['eval']['metric'], self.metric.get()[1]))

    def fit(self, train, val):
        """Fit the model on the whole training dataset

        Args:
            - train: training data iterator
            - val: validation data iterator     
        """
        self.model.fit(X=train,
                       eval_data=val,
                       kvstore=kv,
                       batch_end_callback=mx.callback.Speedometer(
                           self.model_config['data']['batch_size'], 100),
                       epoch_end_callback=checkpoint)

    def partial_fit(self, X, y=None):
        pass

    def decision_function(self, xtt):
        """Predict the decision value on test sest

        Args:
            - xtt: test data iterator to be predicted     

        Returns:
            - Ndarray of decision values for xtt
        """

        pass

    def predict(self, test_data, batch_size=32):
        """Predict labels on test dataset which is a list of list of encoded tokens (integer).

        Predict labels on a list of list of integers. As for training, test data sample is
        a list of integers mapped from token.

        Args:
          test_data (list): A list of list of integers

        Returns:
          labels (list): a list of integers (labels)
        """

        sample_ids = range(len(test_data))
        labels = np.zeros(shape=(len(test_data, )), dtype=int)
        scores = np.zeros(shape=(len(test_data), self.num_classes))
        tt_iter = BucketSeqLabelIter(
            test_data, sample_ids, batch_size=batch_size)
        for batch in tt_iter:
            self.model.forward(batch, is_train=False)
            out = self.model.get_outputs()[0].asnumpy()
            for logits, idx in zip(out, batch.label[0].asnumpy()):
                labels[idx] = np.argmax(logits)
                scores[idx] = logits
        return labels, scores

    def save(self, path, epoch=None):
        pass


if __name__ == '__main__':

    clf = H3LightenedCNN()