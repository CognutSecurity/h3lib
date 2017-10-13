"""
PipelineWorkers is a collection of workers, each of which transform a dataset input to another form.
Typical usage is that any worker accpet a dataset with n rows in a list, and outputs a n rows transformed dataset.

In machine learning, before we can pass the dataset in any learning algorithm, we should preprocess the data in
a right way. For each worker, we define a particular preprocessing step for the learning algorithm, such as tokenizer,
stemmer, tf-idf vectorizer, normalizer, standardizer and so on. You can even append or remove additional features or do
feature engineering as you wish. However, you need to make sure the order of the workers defined in your pipeline, such
that the dataset can be processed in a stream line without any conflicts.

Author: Huang Xiao
Email: xh0217@gmail.com
Copyright@2016, Stanford
"""

import nltk, numpy as np
from collections import OrderedDict
from nltk.tokenize import RegexpTokenizer
from javalang.tokenizer import LexerError, tokenize as javalang_tokenize
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer as TfidfVec
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Worker(object):
    """Super class for all pipeline workers
    Input for every worker should be the same length of the output
    E.g., input a list of n emails returns a list of n lists of tokens

    Args:

    Returns:

    """

    def __init__(self):
        self.fitted = False

    def transform(self, X, y=None):
        pass


class WordTokenizer(Worker):
    """Word tokenizer transforms a list of documents to list of tokens,
    each element in list corresponds to a training sample.
    
    Tokenizer inherits nltk.word_tokenize to transform a list of texts to a list of token rows,
    each of which contains a list of tokens from the text.

    Args:

    Returns:

    """

    def __init__(self, params={'language': 'english'}):
        super(WordTokenizer, self).__init__()
        self.language = params['language']

    def transform(self, X, y=None):
        """

        Args:
          dataset: 

        Returns:

        """

        tokens = list([])
        tokenizer = RegexpTokenizer(r'\w+')
        for row in X:
            tokens.append(tokenizer.tokenize(row.lower()))
        return tokens, y


class JavaTokenizer(Worker):
    """Java tokenizer transforms a list of java code snippets to list of tokens,
    each element in list corresponds to a training sample.
    
    JavaTokenizer use javalang.JavaTokenizer to transform a list of snippets to a list of token rows,
    each of which contains a list of tokens from the code. If it raises value error while parsing, it skips
    the sample.

    Args:

    """

    def __init__(self, params={'language': 'java'}):
        super(JavaTokenizer, self).__init__()
        self.language = params['language']

    def transform(self, X, y=None):
        """
        This function parses a list of java code snippets into a list of lists of tokens.

        Args:
        X: Java code snippets in a list of strings. 
        y:  (Default value = None) a list of labels for code snippets 

        Returns:
        X: a list of lists of tokens
        y: a list of labels 
        discard_snippets: number of issue snippets which are not parsed. 

        """
        list_tokens = []
        labels = []

        if not y:
            fake_y = np.zeros(len(X))
        else:
            fake_y = y

        for sent, label in zip(X, fake_y):
            try:
                token_list = [token.value for token in list(javalang_tokenize(sent))]
                list_tokens.append(token_list)
                labels.append(label)
            except LexerError:
                print "Value error while parsing : \n %s" % (sent)
                continue

        return list_tokens, labels


class Stemmer(Worker):
    """Stemmer transforms a list of token rows to be stemmed tokens, e.g., moved -> move
    
    Stemmer inherits nltk.PorterStemmer to transform a list of token rows to a list of stemmed token rows,
    note that stemming can be problematic sometimes.

    Args:

    Returns:

    """

    def __init__(self, params=None):
        super(Stemmer, self).__init__()

    def transform(self, X, y=None):
        """

        Args:
          X: input list of lists of tokens
          y: (default None)

        Returns:

        """
        # print 'stemming...'
        stemmer = nltk.PorterStemmer()
        stems = list([])
        for row in X:
            stem_row = [stemmer.stem(token) for token in row]
            stems.append(stem_row)
        return stems, y


class TfidfVectorizer(Worker):
    """TfidfVectorizer transforms a list of token rows to a list of real valued feature set.
    
    TfidfVectorizer inherits scikit-learn's tfidf method. We use the defaut tfidf vectorizer from sklearn, to support most important parameters

    Args: 
      More details see:
          http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    Return: 
      instance of the TfidfVectorizer worker

    """

    def __init__(self, params=None):

        if params.has_key('stop_words'):
            _stop_words = params['stop_words']
        else:
            _stop_words = None
        if params.has_key('ngram_range'):
            _ngram_range = params['ngram_range']
        else:
            _ngram_range = (1, 2)
        if params.has_key('max_df'):
            _max_df = params['max_df']
        else:
            _max_df = 1.0
        if params.has_key('min_df'):
            _min_df = params['min_df']
        else:
            _min_df = 1
        if params.has_key('max_features'):
            _max_features = params['max_features']
        else:
            _max_features = None

        super(TfidfVectorizer, self).__init__()
        self._worker = TfidfVec(stop_words=_stop_words,
                                ngram_range=_ngram_range,
                                max_df=_max_df,
                                min_df=_min_df,
                                max_features=_max_features)

    def transform(self, X, y=None):
        """transform is called to initialize the vectorizer, status will be refreshed

        Args:
          X: input dataset
          y: (default: None) input labels

        Returns:
          samples: Tf-idf vectors in ndarray

        """

        # print 'tf-idf vectorizing...'
        samples = self._worker.fit_transform(X)
        self.fitted = True
        return samples.toarray(), y

    def partial_transform(self, X, y=None):
        """partial transform uses current vectorizer to fit more data, typically used for tranform testing data

        Args:
          X: input dataset
          y: (default: None) input labels

        Returns:
          samples: Tf-idf vectors in ndarray

        """
        if self.fitted is False:
            print 'TfidfVectorizer is not yet initialized on any dataset, call transform instread. exit...\n'
            return False
        else:
            samples = self._worker.transform(X)
            return samples.toarray(), y


class Normalizer(Worker):
    """Normalize input samples to unit norm. 

    Args:
      params: dict for parameters 
              params['norm'] can be 'l1' or 'l2'
      See also: sklearn.preprocessing.normalize

    """

    def __init__(self, params={'norm': 'l2'}):

        super(Normalizer, self).__init__()
        if not params.has_key('norm') or params['norm'] not in ['l1', 'l2']:
            print 'Error initialze a normalizer, needs set as l1 or l2'
        self.norm = params['norm']

    def transform(self, X, y=None):
        """transform is called to initialize the vectorizer, status will be refreshed

        Args:
          dataset: input dataset is a ndarray with shape=(n,d), where n is the sample size, and d is the feature size

        Returns:
          samples: (ndarray) normalized dataset

        """

        # print 'normalizing...'
        samples = normalize(X, norm=self.norm, axis=1)
        return samples, y


class FeatureScaler(Worker):
    """Scale an input dataset to zero-mean and unit variance.
    It supports now standard scaling and minmax scaling, see also:
        - sklearn.preprocessing.StandardScaler
        - sklearn.preprocessing.MinMaxScaler

    Args:
      params: (dict) params['online'] Boolean value indicates if fit the data online

    """

    def __init__(self, params=None):
        
        super(FeatureScaler, self).__init__()
        if params.has_key('type'):
            _type = params['type']
        else:
            _type = 'standard'
        if _type == 'standard':
            self._worker = StandardScaler()
        if _type == 'minmax':
            self._worker = MinMaxScaler()

    def transform(self, X, y=None):
        """transform is called to scale the dataset feature to zero-mean and unit variance.

        Args:
          X: Nxd ndarray with shape=(n,d), where n is the sample size, and d is the feature size
          y: (default: None) input labels

        Returns: 
          samples: (ndarray) scaled dataset

        """

        # make sure it is ndarray
        X = np.array(X)
        if X.shape[1] == 0:
            # if the input contains nothing...
            return np.array([])
        samples = self._worker.fit_transform(X)
        self.fitted = True
        return samples, y

    def partial_transform(self, X, y=None):
        """partial tranform use current scaler to scale input samples to zero-mean and unit variance.

        Args:
          X: Nxd ndarray with shape=(n,d), where n is the sample size, and d is the feature size
          y: (default: None) input labels

        Returns: 
          samples: (ndarray) scaled dataset

        """

        if not self.fitted:
            print 'FeatureScaler is not yet initialized on any dataset, call transform instread. exit...\n'
            return False
        else:
            # use existing scaler
            X = np.array(X)
            if X.shape[1] == 0:
                # if the input contains nothing...
                return np.array([])
            samples = self._worker.transform(X)
            return samples, y


class VaderSentiment(Worker):
    """VaderSentiment transforms a list of text corpus to a list of sentiment intensity scores.
    We use the Vader sentiment analysis tools provided by NLTK, this worker is stateless.
    [Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
     Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.]

    Args:
      params: (dict) parameters for VaderSentiment
              params['corpuses'] is the only params

    """

    def __init__(self, params=None):
       
        super(VaderSentiment, self).__init__()

    def transform(self, X, y):
        """append an additional sentiment scores col. in the end of the dataset

        Args:
          X: a list of N text corpuses (document)

        Returns:
          sentiment_scores: (Nx1 ndarray)  sentiment scores

        """

        sentiment_scores = list()
        sid = SentimentIntensityAnalyzer()
        for row in X:
            # tokenzie sentences
            lines_list = nltk.tokenize.sent_tokenize(row)
            # we compute the compouond scores for each sentence in one single text
            scores = [sid.polarity_scores(line)['compound'] for line in lines_list]
            # we compute avg. sentiment score for the text
            sentiment_scores.append(np.mean(scores))
        sentiment_scores = np.array(sentiment_scores).reshape(len(sentiment_scores), 1)
        return sentiment_scores, y


class HashParser(Worker):
    """HashParser parses a list of hashes to get a list of features
    
    Args: 
      params: (dict)

    """

    def __init__(self, params=None):
       
        super(HashParser, self).__init__()
        # keys contain a list of keys found in hash
        self.feature_mapping = OrderedDict()

    def transform(self, X, y=None):
        """convert a list of hashes to a list of features

        Args:
          X: a list of hashes
          y: (default: None) a list of labels  

        Returns:
          X: (ndarray) a list of feature vectors
          y: a list of labels or None

        """

        # find all keys/values first
        for item in X:
            for k in item.keys():
                if k not in self.feature_mapping.keys():
                    value_list = list()
                    value_list.append(item[k])
                    self.feature_mapping[k] = value_list
                else:
                    if item[k] not in self.feature_mapping[k]:
                        self.feature_mapping[k].append(item[k])

        features = []
        for item in X:
            row = []
            for k in self.feature_mapping.keys():
                if item.has_key(k):
                    try:
                        idx = self.feature_mapping[k].index(item[k])
                        row.append(idx + 1.0)
                    except ValueError:
                        row.append(-1.0)
                else:
                    row.append(0.0)
            features.append(row)

        self.fitted = True
        return np.array(features), y

    def partial_transform(self, X, y=None):
        """partial_transform convert a list of dicts to features using current available keys

        Args:
          X: a list of dict

        Returns:
          a list of features

        """

        if not self.fitted:
            print 'HashParser is not yet initialized on any dataset, call transform instead. exit...\n'
            return False
        else:
            features = []
            for item in X:
                row = []
                for k in self.feature_mapping.keys():
                    if item.has_key(k):
                        try:
                            idx = self.feature_mapping[k].index(item[k])
                            row.append(idx + 1)
                        except ValueError:
                            row.append(-1)
                    else:
                        row.append(0)
                features.append(row)
            return np.array(features), y


