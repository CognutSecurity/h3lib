'''
This utility file contains helper functions for dataset preparation. It is intended for preparing
datasets used in other places.

Author: Huang Xiao
Group: Cognitive Security Technologies
Institute: Fraunhofer AISEC
Mail: huang.xiao@aisec.fraunhofer.de
Copyright@2017
'''

import urllib2
import json
import csv
import os.path
import sys
import numpy as np
from termcolor import colored
from javalang.tokenizer import LexerError, tokenize as javalang_tokenize


def read_csv(filename, header_line=True):
    """load dataset from csv file, only numeric values are supported now.

    Args:
    filename: file path to csv
    header_line:  (Default value = True) if there is a header line

    Returns:
    data: a list of data in ndarray
    nodes: a list of node names if header line exists or simply interpreted as indices.

    """
    csv_file = open(filename, 'r')
    reader = csv.reader(csv_file)
    cnt = 0
    input_data = []
    nodes = []
    for row in reader:
        if cnt == 0 and header_line:
            nodes = row
        else:
            input_data.append([float(elem) for elem in row])
        cnt += 1
    csv_file.close()
    return np.array(input_data), nodes


def csv2dict(csv):
    """convert csv data to a list of dicts, each of which is a data sample,
    keys are the feature names, values are the feature values. 

    Args:
      csv: the input file in CSV format 

    Returns:
      a list of dicts

    """

    data, nodes = read_csv(csv)
    res = list([])
    for row_id in range(data.shape[0]):
        instance = dict()
        for i, n in enumerate(nodes):
            instance[n] = data[row_id, i]
        res.append(instance)
    return res


def load_snp17(csv_file,
               sep=',',
               header_line=False,
               force_overwrite=False,
               save_path='../datasets/snp17.p'):
    """Load S&P'17 Stackoverflow dataset from file and save
       the ndarray into pickle.

    Args:
      csv_file: the input data file in CSV format.
      sep:  (Default value = ',') separator for CSV file
      header_line:  (Default value = False) wether file has a header line or not
      force_overwrite:  (Default value = False) if we should force overwrite 
      save_path:  (Default value = '../datasets/snp17.p') where to save the dataset

    Returns:
      snippets: a list of java code snippets
      labels: a list of labels indicating if the codes are secure or not
      feature_names: a list of feature names or empty if this information does not exist

      or False if exception is raised.
    """

    if os.path.isfile(save_path) and not force_overwrite:
        data = json.load(open(save_path, 'r'))
        return data['data'], data['labels'], data['features']

    snippets = []
    labels = []
    feature_names = []
    try:
        with open(csv_file, 'r') as fd:
            reader = csv.reader(fd)
            for row in reader:
                if header_line and len(snippets) == 0:
                    feature_names = row
                    continue
                snippets.append(row[0])
                labels.append(int(row[2]))
            snp_dict = {'data': snippets,
                        'labels': labels,
                        'features': feature_names}
            json.dump(snp_dict, open(save_path, 'w'))
            return snippets, labels, feature_names
    except IOError:
        print 'File does not exist.'
        return False
    finally:
        msg = "=" * 10 + "\n"
        msg += "Total: %d java snippets in %d classes. " % (
            len(snippets), np.unique(labels).size)
        print colored(msg, color='green')


def java_tokenize(snippets, labels=None):
    """
    This function parses a list of java code snippets into a list of lists of tokens.

    Args:
      snippets: Java code snippets in a list of strings. 
      labels:  (Default value = None) a list of labels for code snippets 

    Returns:
      X: a list of lists of tokens
      y: a list of labels 
      discard_snippets: number of issue snippets which are not parsed. 

    """
    X = []
    y = []
    discard_snippets = 0
    for sent, label in zip(snippets, labels):
        try:
            token_list = [token.value for token in list(
                javalang_tokenize(sent))]
            X.append(token_list)
            y.append(label)
        except LexerError:
            discard_snippets += 1
            continue
    return X, y, discard_snippets


def download_GTSRB(dst_dir, unzip=False, remove_cache=False):
    """Download German Traffic Sign Recognition Benchmark dataset

    Args:
      dst_dir: destination folder        
      unzip: if the data should be unzipped
      remove_cache: if the zip file should be removed, only valid when unzip is true

    Returns:
      None  

    Raises:
      HTTPError
    """

    url_tr = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip"
    url_tt = "http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip"
    try:
        resp_tr = urllib2.urlopen(url_tr)
        resp_tt = urllib2.urlopen(url_tt)
        print 30 * "--"
        print "German Traffic Sign Recognition Benchmark"
        print "Training dataset: %d MB | Test dataset: %d MB" % \
            (int(resp_tr.info()['content-length']) / (1024 * 1024),
             int(resp_tt.info()['content-length']) / (1024 * 1024))
        print 30 * "--"
        download_dir = os.path.join(dst_dir, "gtsrb")
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        with open(os.path.join(download_dir, url_tr.split('/')[-1]), 'w') as ftr:
            print 'Dowloading training dataset into %s .. \n' % (download_dir)
            total_chunks = 50
            chunk_size = min(
                int(resp_tr.info()['content-length']) / total_chunks, 1024 * 1024)
            cnt = 1
            while True:
                chunk = resp_tr.read(chunk_size)
                if not chunk:
                    print "done!"
                    break
                else:
                    ftr.write(chunk)
                    cnt += 1
                    progress = min(100, float(cnt) * chunk_size * 100 /
                                   float(resp_tr.info()['content-length']))
                    sys.stdout.write('\r{0:.1f}% {1:s}'.format(
                        progress, int(progress / 5) * "|"))
                    sys.stdout.flush()
        ftr.close()            
        print 30 * '--'
        with open(os.path.join(download_dir, url_tt.split('/')[-1]), 'w') as ftt:
            print 'Dowloading test dataset into %s .. \n' % (download_dir)
            total_chunks = 50
            chunk_size = min(
                int(resp_tt.info()['content-length']) / total_chunks, 1024 * 1024)
            cnt = 1
            while True:
                chunk = resp_tt.read(chunk_size)
                if not chunk:
                    print "done!"
                    break
                else:
                    ftt.write(chunk)
                    cnt += 1
                    progress = min(100, float(cnt) * chunk_size * 100 /
                                   float(resp_tt.info()['content-length']))
                    sys.stdout.write('\r{0:.1f}% {1:s}'.format(
                        progress, int(progress / 5) * "|"))
                    sys.stdout.flush()
        ftt.close()

    except urllib2.URLError as e:
        print e.message
        return False


if __name__ == '__main__':
    download_GTSRB('/Users/hxiao/data')
