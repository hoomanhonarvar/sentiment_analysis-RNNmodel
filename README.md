# Sentiment Analysis with RNN
This repository had became because of NLP class in Isfahan university which had been held by Dr.Baradaran. Sentiment analysis is the process of analyzing digital text to determine if the emotional tone of the message is positive, negative, or neutral. Today, companies have large volumes of text data like emails, customer support chat transcripts, social media comments, and reviews. Sentiment analysis tools can scan this text to automatically determine the author’s attitude towards a topic.


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Introduction](#Sentiment-Analysis-with-RNN)
	* [Requirements](#requirements)
  * [Usage](#usage)
    * [Improve customer service](#imporve-customer-service)
    * [Brand monitoring](#brand-monitoring)
		* [Market research](#market-research)
    * [Track campaign performance](#track-campaign-performance)
	* [Steps](#steps)
		* [Loading dataset](#loading-dataset)
		* [Preprocess](#preprocess) 
		* [Sentence Vectorization](#sentence-vectorization)
    * [Word Embedding](#word-embedding)
		* [RNN Model](#rnn-model)
      * [Embedding layer](#embedding-layer)
      * [Dropout layer](#dropout-layer)
      * [Recurrent Neural Network](#rnn)
      * [Dense layer](#dense-layer)
		* [Loss and metrics](#loss-and-metrics)
	* [Result](#result)


<!-- /code_chunk_output -->

## Requirements
* Python >= 3.9
* NVIDIA® GPU drivers version 450.80.02 or higher
* CUDA® Toolkit 11.8
* tensorflow
* nltk
* sklearn

## Usage
Businesses use sentiment analysis to derive intelligence and form actionable plans in different areas.

### Improve customer service
Customer support teams use sentiment analysis tools to personalize responses based on the mood of the conversation. Matters with urgency are spotted by artificial intelligence (AI)–based chatbots with sentiment analysis capability and escalated to the support personnel.
### Brand monitoring
Organizations constantly monitor mentions and chatter around their brands on social media, forums, blogs, news articles, and in other digital spaces. Sentiment analysis technologies allow the public relations team to be aware of related ongoing stories. The team can evaluate the underlying mood to address complaints or capitalize on positive trends. 
### Market research
A sentiment analysis system helps businesses improve their product offerings by learning what works and what doesn't. Marketers can analyze comments on online review sites, survey responses, and social media posts to gain deeper insights into specific product features. They convey the findings to the product engineers who innovate accordingly. 
### Track campaign performance
Marketers use sentiment analysis tools to ensure that their advertising campaign generates the expected response. They track conversations on social media platforms and ensure that the overall sentiment is encouraging. If the net sentiment falls short of expectation, marketers tweak the campaign based on real-time data analytics. 


## Steps
The aim of this project is to implement a sentiment analysis system. The dataset used in this exercise is the 140sentiment dataset, which consists of 1,600,000 tweets. These tweets are divided into two categories, positive and negative. In this exercise, we work with two columns, text and sentiment. Positive sentiment in this data set with value 4 and negative sentiment with zero
it has been shown.

### Loading dataset
The dataset used in this project is the sentiment140 dataset, which consists of 1,600,000 tweets.
The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .
[you can download it here](https://www.kaggle.com/datasets/kazanova/sentiment140)

### Preprocess

1. Change the label of positive tweets to 1 and the label of negative tweets to zero
2. Replace URL's with URL tokens, mentions with MENTION tokens, and hashtags with HASHTAG tokens.
3. Remove punctuation marks
4. Tokenization of each tweet to its words
5. lemmitization


### Sentence Vectorization

The purpose of this section is to assign each unique word a unique number and then replace that word with the assigned number. Since we need data with dimensions of people for processing, therefore, I also apply the sequences_pad method on the dataset.

### Word Embedding
In this exercise, we implement word embedding using the Vec2Word method. A word embedding is a learned representation for text in which words with the same meaning have a similar representation. Vec2Word is a popular approach that uses neural networks to learn these word embeddings. Use the Vec2Word model to embed words using the gensim library. Run the code snippet below to download this model.
```
# load google news word2vec
import gensim.downloader as api
w2v = api.load('word2vec-google-news-300')
```
### RNN Model
Recurrent Neural Network(RNN) is a type of Neural Network where the output from the previous step is fed as input to the current step. In traditional neural networks, all the inputs and outputs are independent of each other. Still, in cases when it is required to predict the next word of a sentence, the previous words are required and hence there is a need to remember the previous words. Thus RNN came into existence, which solved this issue with the help of a Hidden Layer. The main and most important feature of RNN is its Hidden state, which remembers some information about a sequence. The state is also referred to as Memory State since it remembers the previous input to the network. It uses the same parameters for each input as it performs the same task on all the inputs or hidden layers to produce the output. This reduces the complexity of parameters, unlike other neural networks.

#### Embedding layer
Most deep learning frameworks provide an embedding layer, which can be initialized with pre-trained embeddings or trained from scratch. This layer converts word indices to dense vectors that are fed into the RNN.
#### Dropout layer
The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1 / (1 - rate) such that the sum over all inputs is unchanged.
#### Recurrent Neural Network
In this layer used 300 nodes for RNN hidden layer
#### Dense layer
This layer is representing the output of model.

### Loss
The used loss function is : [Cross Entropy](https://en.wikipedia.org/wiki/Cross-entropy)
#### Metrics
  ```json
  "metrics": ["f1-score", "precision","recall"],
  ```


## Result
|                 |    f1-score      |  precision  |  recall  |
| --------------- | ---------------- |  ---------- |  ------  | 
| negetive class  |       0.82       |     0.79    |   0.86   |
| posetive class  |       0.81       |     0.85    |   0.77   |

accuracy score:  0.81533125

