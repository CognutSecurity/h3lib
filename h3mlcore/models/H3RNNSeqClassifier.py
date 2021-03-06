'''
A bidirectional LSTM sequence model used for document classification.
It is basically a sequence classification model developed in mxnet.

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
from h3mlcore.models.H3BaseActor import H3BaseActor
from h3mlcore.utils.MxHelper import BasicArgparser
from h3mlcore.io.BucketSeqLabelIter import BucketSeqLabelIter


class H3RNNSeqClassifier(H3BaseActor):
    """ """

    def __init__(self,
                 num_hidden=256,
                 num_embed=128,
                 input_dim=None,
                 lstm_layer=1,
                 num_classes=2,
                 params_file='',
                 learning_rate=.1,
                 optimizer='sgd',
                 metric='acc',
                 use_gpus=[],
                 use_cpus=[],
                 logging_root_dir='logs/',
                 logging_config='configs/logging.yaml',
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
        self.num_hidden = num_hidden
        self.num_embed = num_embed
        self.input_dim = input_dim
        self.lstm_layer = lstm_layer
        self.num_classes = num_classes
        self.params_file = params_file
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        if metric == 'acc':
            self.metric = mx.metric.Accuracy()
        elif metric == 'cross-entropy':
            self.metric = mx.metric.CrossEntropy()
        elif metric == 'topk':
            self.metric = mx.metric.TopKAccuracy(top_k=3)

        self.ctx = []
        if use_gpus:
            self.ctx = [mx.gpu(i) for i in use_gpus]
        elif use_cpus:
            self.ctx = [mx.cpu(i) for i in use_cpus]
        else:
            self.ctx = mx.cpu(0)
        self.model = None

    def _sym_gen(self, seq_len):
        """Dynamic symbol generator

        For variable length sequence model, we define a dynamic symbol generator
        to generate various length unrolled sequence model based on differnet cells.abs

        Args:
          seq_len(int): The sequence length to unroll

        Returns:
          mx.sym.Symbol: pred-> a symbol for the output of the sequence model

        """

        data = mx.sym.Variable(name='data')
        label = mx.sym.Variable(name='softmax_label')
        embeds = mx.symbol.Embedding(
            data=data, input_dim=self.input_dim, output_dim=self.num_embed, name='embed')
        lstm_1 = mx.rnn.LSTMCell(prefix='lstm_1_', num_hidden=self.num_hidden)
        outputs, _ = lstm_1.unroll(seq_len, inputs=embeds, layout='NTC')
        for i in range(self.lstm_layer - 1):
            new_lstm = mx.rnn.LSTMCell(
                prefix='lstm_' + str(i + 2) + '_', num_hidden=self.num_hidden)
            outputs, _ = new_lstm.unroll(seq_len, inputs=outputs, layout='NTC')
        pred = mx.sym.FullyConnected(
            data=outputs[-1], num_hidden=self.num_classes, name='logits')
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    def initialize(self, data_iter):
        """Initialize the neural network model

        This should be called during model constructor. It tries to load NN parameters from
        a file, if it does exist, otherwise initialize it. It sets up optimizer and optimization
        parameters as well.

        Args:
          data_iter(mx.io.NDArrayIter): initialize the model with data iterator, it should be
            of type BucketSeqLabelIter.
        """

        if not isinstance(data_iter, BucketSeqLabelIter):
            err_msg = "Data iterator for this model should be of type BucketSeqLabelIter."
            raise TypeError(err_msg)
            self.logger.error(err_msg, exc_info=True)
            return

        self.model = mx.module.BucketingModule(
            sym_gen=self._sym_gen, default_bucket_key=data_iter.default_bucket_key, context=self.ctx)
        self.model.bind(data_iter.provide_data, data_iter.provide_label)
        if os.path.isfile(self.params_file):
            try:
                self.model.load_params(self.params_file)
                self.logger.info(
                    "LSTM model parameters loaded from file: %s." % (self.params_file))
            except (IOError, ValueError):
                self.logger.warning(
                    "Parameters file does not exist or not valid! please check the file.")
        else:
            self.model.init_params()
            self.logger.info("LSTM Model initialized.")

        self.model.init_optimizer(optimizer=self.optimizer,
                                  optimizer_params=(('learning_rate', self.learning_rate),))

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

        The model will be trained in epochs and possibly evaluated with validation dataset. The
        model parameters will be saved on disk. Note that for Bucketing model, only network parameters
        will be saved in checkpoint, since model symbols need to be created according to buckets
        which match the training data.

        Args:
          train_data (BucketSeqLabelIter): Training data iterator
          eval_data (BucketSeqLabelIter): Validation data iterator
          num_epochs (int): Number of epochs to train
        """

        for e in range(num_epochs):
            train_data.reset()
            for batch in train_data:
                self.step(data_batch=batch)
            if eval_data:
                eval_data.reset()
                self.model.score(eval_data, self.metric)
                self.logger.info("Training epoch %d -- Evaluate %s: %f"
                                 % (e + 1, self.metric.name, self.metric.get()[1]))

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
        """Save model parameters for BucketingModel

        This function saves model offline, either a checkpoint or parameters for Bucketing model.
        Note that normally it can be saved as checkpoint, but for variable length model such as 
        BucketingModel, we can only save parameters and initialize the model with parameter loading,
        since the unrolled version of models need to be determined by data iterator, which can be
        any length. 

        Args:
          path (str): a valid path to save the checkpoint/parameters
        """

        if epoch:
            path = path + '-' + str(epoch)

        self.model.save_params(path)
        self.logger.info('Network parameters saved in %s' % (path))


if __name__ == '__main__':
    '''
    Run from terminal
    '''
    # arg_parser = BasicArgparser(
    #     prog="LSTM Models with varying length inputs.").get_parser()
    # args = arg_parser.parse_args()
    # # basic parameters
    # epochs = args.epochs
    # batch_size = args.batch_size
    # lr = args.learning_rate
    # ctx = []
    # if args.gpus:
    #     for gid in args.gpus:
    #         ctx.append(mx.gpu(args.gpus[gid]))
    # elif args.cpus:
    #     for cid in args.cpus:
    #         ctx.append(mx.cpu(args.cpus[gid]))
    # else:
    #    # default
    #     ctx = mx.cpu(0)

    from termcolor import colored
#    from nltk.tokenize import word_tokenize
    from sklearn.cross_validation import train_test_split
    from h3mlcore.utils.DatasetHelper import load_snp17, java_tokenize

    # load data
    # datafile = "../datasets/npc_chat_data2.p"
    # data = pickle.load(open(datafile, 'r'))
    # all_sents = data['Xtr']
    # sents = [word_tokenize(sent) for sent in all_sents]
    # labels = np.array(data['ytr'], dtype=int) - 1
    # label_names = data['label_info']
    all_sents, all_labels, _ = load_snp17(csv_file='/Users/hxiao/repos/h3lib/h3db/snp17/train/answer_snippets_coded.csv',
                                          save_path='/Users/hxiao/repos/webdemo/datasets/snp17.p',
                                          force_overwrite=False)

    sents, labels, discard_snippets = java_tokenize(all_sents, all_labels)
    sents_encoded, vocab = mx.rnn.encode_sentences(sents, vocab=None, invalid_key='\n',
                                                   invalid_label=-1, start_label=0)
    word_map = dict([(index, word) for word, index in vocab.iteritems()])
    print 'Total #encoded_snippets: %d, #issue_snippets: %d, total #tokens: %d' \
        % (len(sents_encoded), discard_snippets, len(vocab))
    tr_data, tt_data, tr_labels, tt_labels = train_test_split(
        sents_encoded, labels, train_size=0.8)
    buckets = [50, 100, 200, 1000]
    tr_iter = BucketSeqLabelIter(
        tr_data, tr_labels, buckets=buckets, batch_size=64)
    tt_iter = BucketSeqLabelIter(
        tt_data, tt_labels, buckets=buckets, batch_size=64)

    clf = H3RNNSeqClassifier(input_dim=len(vocab), num_classes=np.unique(labels).size)
    clf.initialize(tr_iter)
    clf.train_epochs(tr_iter, tt_iter, num_epochs=50)

    # test
    # test_sents = [word_tokenize(sent) for sent in all_sents[100:400]]
    # test_labels = labels[100:400]
    # test_sents_encoded, _ = mx.rnn.encode_sentences(test_sents, vocab=vocab)
    # preds, logits = clf.predict(test_sents_encoded, batch_size=50)
    # for s, p, lgt, real in zip(all_sents[100:300], preds, logits, test_labels):
    #     if real == p:
    #         print colored(s, color='blue') + \
    #             colored(' -> ' + label_names[p] +
    #                     ' <- ' + label_names[real], color='green')
    #     else:
    #         print colored(s, color='blue') + \
    #             colored(' -> ' + label_names[p] +
    #                     ' <- ' + label_names[real], color='red')

    # print 'Logits: ', lgt
