# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from PARAMETERS import Parameter as pm
from DATA_PRO import read_category,get_wordid,get_word2vec,process,batch_iter,seq_length
from LSTM_CNN import Cnn
import matplotlib.pyplot as plt
import os
from sklearn import metrics


filenames = [pm.train_filename, pm.test_filename, pm.val_filename]
categories, cat_to_id = read_category()
wordid = get_wordid(pm.vocab_filename)
pm.vocab_size = len(wordid)
pm.pre_trianing = get_word2vec(pm.vector_word_npz)

# 加载cnews数据
x_train, y_train =process(pm.train_filename,wordid, cat_to_id, pm.seq_length)
x_val, y_val = process(pm.val_filename, wordid, cat_to_id, pm.seq_length)
x_test, y_test = process(pm.test_filename, wordid, cat_to_id, pm.seq_length)

#model = Lstm_Cnn(vocab_size=pm.vocab_size, embed_size=pm.embedding_size, class_num=pm.num_classes)
model =Cnn(pm.vocab_size, pm.embedding_size, pm.num_classes)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),

              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

checkpoint_save_path ='D:\THUCNEWS\data/checkpoint/Lstm_Cnn'
'''if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 verbose=1)'''

history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    batch_size=pm.batch_size, epochs=pm.epochs,validation_freq=1, )

#

model.evaluate(x_test, y_test, batch_size=pm.batch_size)

model.summary()
'''y_train_cls=np.argmax(y_train,1)
y_pred_cls=np.zeros(shape=(x_train),dtype=np.int32)
metrics.accuracy_score(y_train_cls,y_pred_cls,normalize=False)
metrics.recall_score(y_train_cls,y_pred_cls,average="macro")
metrics.f1_score(y_train_cls,y_pred_cls,average='weighted')
print("Acc，Recall，F1")
print(metrics.classification_report(y_train_cls,y_pred_cls,target_names=categories))'''


# 显示训练集和验证集的acc和loss曲线
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


plt.subplot()
plt.plot(acc, label='Training Accuracy',linewidth = '2',  linestyle=':', marker='.')
plt.plot(val_acc,label='Validation Accuracy',linewidth = '2',  linestyle=':', marker='.')
plt.title('Accuracy and Loss')
plt.legend()
plt.ylim(0,1)
plt.grid()
plt.axhline(y=0.90,color='black')
plt.tick_params(axis='both',color='yellow')
plt.show()


