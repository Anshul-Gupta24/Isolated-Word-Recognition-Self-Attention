from model_attention import WordNet
import keras	
import numpy as np
from keras.utils import plot_model
import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import functions
from keras.callbacks import ModelCheckpoint
import sklearn.model_selection
import timeit
import json


if __name__ == '__main__':

    start = timeit.default_timer()
    BATCH_SIZE = 32
    FEATURE_SIZE = 80   
    data_path = '/home/data.pkl'
    save_path = '/home'


    model = WordNet()("train")
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5), 
                loss='categorical_crossentropy',
                metrics=['accuracy'])


    # Get Data
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp) 

    lst = list(data.keys())
    labels = [l[17:functions.get_sec_und(l)-1] for l in lst]
    classes = set(labels)
    class_enc = {}
    for i,c in enumerate(classes):
        class_enc[c] = i


    label_vecs = np.zeros((len(lst), len(classes)))
    for i,lv in enumerate(label_vecs):
       lv[class_enc[labels[i]]] = 1

    lst_train, lst_val, label_vecs_train, label_vecs_val = sklearn.model_selection.train_test_split(lst, label_vecs, test_size=0.2)
    print(len(lst))
    print(len(lst_train))
    print(len(lst_val))
    

    def generator(lst, label_vecs):
        batch_size = BATCH_SIZE
        features = FEATURE_SIZE
        i = 0
        while(True):

            j=i+batch_size
            if j>=len(lst):
                i = 0
                j=i+batch_size


            seq_lens = [len(data[lst[x]]) for x in range(i,j)]
            max_len = max(seq_lens)			# Get max sequence length

            batch_data = np.zeros((batch_size, max_len, features))
            batch_labels = np.array(label_vecs[i:j])

            c=0
            for x in range(i,j):
                batch_data[c, :seq_lens[c], :]=data[lst[x]]
                c+=1

            i=j
                        
            yield batch_data, batch_labels

        

    filepath = save_path+"/model.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='min', period=1)
    callbacks_list = [checkpoint]


    train_size = len(lst_train)
    val_size = len(lst_val)


    history = model.fit_generator(generator(lst_train, label_vecs_train), steps_per_epoch=int(train_size/BATCH_SIZE), epochs=80, callbacks=callbacks_list, validation_data=generator(lst_val, label_vecs_val), validation_steps=int(val_size/BATCH_SIZE))


    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(save_path+"/model_loss.png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['train', 'val'], loc='upper left')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.savefig(save_path+"/model_acc.png", bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    with open(save_path+'/history.json', 'w') as fp:
        json.dump(history.history, fp)

    stop = timeit.default_timer()
    time = open('time.txt','w')
    time.write(str(stop-start))
