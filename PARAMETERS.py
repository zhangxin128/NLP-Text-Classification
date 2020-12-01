# -*- coding: utf-8 -*-
class Parameter(object):

    embedding_size = 100    #dimension of word embedding
    vocab_size = 10000      #number of vocabulary
    pre_trianing = None      #use vector_char trained by word2vec
    max_features=100

    seq_length = 100        #max length of sentence
    num_classes = 10         #number of labels

    num_filters = 128       #number of convolution kernel
    kernel_size = 5  #size of convolution kernel

    keep_prob = 0.5          #droppout
    lr= 0.001                #learning rate
    lr_decay= 0.9            #learning rate decay
    clip= 5.0                #gradient clipping threshold

    epochs =20         #epochs
    batch_size = 64          #batch_size



    train_filename='D:\THUCNEWS\word2Vectrain.bin'  #train data
    test_filename='D:\THUCNEWS/cnews.test.txt'    #test data
    val_filename='D:\THUCNEWS/cnews.val.txt'      #validation data
    vocab_filename='D:\THUCNEWS/cnews.vocab.txt'        #vocabulary
    vector_word_filename='D:\THUCNEWS/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='D:\THUCNEWS/vector_word.npz'   # save vector_word to numpy file
