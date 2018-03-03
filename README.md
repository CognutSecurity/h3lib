# H3 Machine Learning Core Library

H3 machine learning core library is an application oriented library for ML practitioners who intend to build actors based large scale learning system. The h3mlcore is the main project which is not another general purpose ML frameworks as existings ones, such as tensorflow, mxnet, caffee, or scikit-learn. Nevertheless we found that there're not enough features in those frameworks to help developers to build a full functional system. 

## Introduction 

We briefly introduce the h3mlcore project, mainly descript why and how we reshape some exisiting ML frameworks to help users integrate with systems more easily. 

### Motivation 

While there're many amazing machine learning libraries exist in open source communities, which help researchers and practitioners develop their own learning systems, we found that interfaces between big data systems and machine learning frameworks are generally missing. The main idea of the project is to develop a full functional actor-based machine learning framework which can help practitioners develop their own distributed applications driven by machine learning. 

In fact, most machine learning framework does not provide unified programing model to handle issues such as distribtued feature preprocessing, data ingest, iterator loading and feeding, as well as messaging among processes. We propose a actor based architecture to deal with current issues, which can be unified with a simple prgraming model.

### Actor network architecture 

A learning model is considered as an actor, which impersonates how human processes tasks. Actors connect with each other to form a actor network, where they share the layers, parameters or predictions with each other to help transfer learning. The architecture of actor network is depicted as following, 

![Architecture](docs/ActorML.png)

#### Data ingest
Now we are talking about big data all the time, because the data we can get nowadays explodes exponentially. There're notably a stack of open souced big data technologies which can help users deal with TB/PB level scale of data, such as Hadoop and its ecosystem. In our design, the raw data flows into the system in the nature of stream or minibatches, we believe the nature of data is streaming but not batch. A data batch is a state of information flow, it can change its value over time. 

#### Distributed preprocessing 
In most machine learning frameworks, models are not supposed to be responsible for preprocessing raw data. However, in our concept, actor is also capable of preprocessing raw data with a pipeline of preprocessors, each of which take care of a particular data block, and data blocks are then reduced to a full dataset for training later on. Each preprocessor contains a list of workers, each of which transforms the data block from one format to another. Obviously, preprocessors can be distributed on different machines, such as Spark cluster. But the workers in each preprocessor needs to be executed in a row. The preprocessing is shown as follows, 

![Preprocessing](docs/pipeline.png)

Every actor mounts up with a pipeline of preprocessers, such that actors know exactly how the training data looks like. This can also allocate memory for actor internal neural network, which can further optimize training performance.  

#### Data iterator and feeding
The structured dataset after preprocessing can be stored back to distribtued file systems or databases. And when actors pull data for training, data iterators will be initialized and loaded from file system or databases. Feeding the data to the actor networks will trigger the training process of actor, which updates the internal neural networks parameters.  

#### Update model
The training process is similar as mxnet, minibatches will feed the neural networks and update the parameters. For distributed training, GPUs cluster is neccessary to train deep neural networks, and there's also a key-value store for parameter synchronization among computer cluster nodes. We intend to use mxnet and scikit-learn as our machine learning backend.

#### Actor network
Actors can form a network where actors talk to each other via advanced messaging queue protocol, tools like Apache Kafka and RabbitMQ can easily handle this. Besides, we also use WebSocket to let actors talk to web frontend, so that web based users can interact with the training and testing models. Besides, we can leverage messagers to transfer knowledge between differend models, e.g., transfer learning. Distillation networks is a good example, where softmax output layer is transfered as an input layer for another network. In this way, the second actor is supposed to be more robust against adversarial learning attacks. 

## Dependencies 
We mainly support three main ML frameworks, 

* tensorflow == 1.4.1
* mxnet == 0.12.0 
* scikit-learn == 0.19.0

To visualize, we rely on following, 

* bokeh
* matplotlib

## How to Install 
```
pip install h3mlcore
```

## Hello World Tutorial

## Authors and Copyright

| Huang Xiao, xh0217@gmail.com
| Copyright@2018

## Acknowledge


