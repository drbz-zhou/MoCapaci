# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:40:23 2021

@author: bzhou
"""

import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
from tensorflow import keras
from datetime import datetime
import csv

def tf_mem_patch():
    # to cure tensorflow memory allocation problem
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    # end curing memory allocation problem
    return session

def label2categorical(m_labels, numClass):
    m_labels_exp = np.zeros( ( len(m_labels), numClass ) )
    for i in range (0, len(m_labels)):
        m_labels_exp[i, int(m_labels[i])] = 1
    return m_labels_exp


def train_step(model, epoch, m_data_train, m_y_train, m_data_valid, m_y_valid, modelsavefile, Patience = 50, batch_size=50, weights_only=True):
    cb_checkpoint = keras.callbacks.ModelCheckpoint(modelsavefile, monitor='val_accuracy', mode='max', 
                                                    verbose=1, save_weights_only=weights_only, save_best_only=True)
    cb_earlystop = keras.callbacks.EarlyStopping(patience=Patience, monitor='val_accuracy', verbose = 1, restore_best_weights=True )
    history = model.fit( x = m_data_train, y = m_y_train, epochs = epoch, batch_size=batch_size,
              #use_multiprocessing = True,
              validation_data = (m_data_valid, m_y_valid),
              callbacks=[cb_checkpoint, cb_earlystop],
              #callbacks=[cb_earlystop],  #sometimes can't save model because of h5 bug, early stop restore best weights
              verbose = 2
        )
    return model, history

def train_gen(model, epoch, m_datagen_train, m_datagen_valid, modelsavefile, Patience = 50, Batch_size = 32, weights_only=True):
    cb_checkpoint = keras.callbacks.ModelCheckpoint(modelsavefile, monitor='val_accuracy', mode='max', 
                                                    verbose=1, save_weights_only=weights_only,save_best_only=True)
    cb_earlystop = keras.callbacks.EarlyStopping(patience=Patience, monitor='val_accuracy', verbose = 1, restore_best_weights=True )
    history = model.fit( x = m_datagen_train, epochs = epoch, batch_size=Batch_size,
              #use_multiprocessing = True,
              validation_data = m_datagen_valid,
              callbacks=[cb_checkpoint, cb_earlystop],
              verbose = 1
        )
    return model, history

def append_history(history, acc, val_acc, loss, val_loss):
    acc = np.concatenate( ( acc, np.array(history.history['accuracy'])))
    val_acc = np.concatenate( ( val_acc, np.array(history.history['val_accuracy'])))
    loss = np.concatenate( ( loss, np.array(history.history['loss'])))
    val_loss = np.concatenate( ( val_loss, np.array(history.history['val_loss'])))
    return acc, val_acc, loss, val_loss

def plot_confusion_matrix(cm, class_names, if_save = True, file_path = '', acc=0):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(len(class_names)/2, len(class_names)/2))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("CM "+file_path+", acc:"+str(round(acc,4)))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    if if_save:
        now = datetime.now()
        date_time = now.strftime("%m%d%H%M%S")
        plt.savefig(file_path+'CM-'+date_time+'.png')
    else:
        plt.show()
    return figure


def plot_acc_loss(acc, val_acc, loss, val_loss, if_save = True, file_path = ''):
    epochs_range = range(len(acc))
    
    figure = plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    #plt.show()
    if if_save:
        now = datetime.now()
        date_time = now.strftime("%m%d%H%M%S")
        plt.savefig(file_path+'AccLoss-'+date_time+'.png')
    else:
        plt.show()
    return figure

def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


def train_valid_split_jump(source_list, m_ratio):
    randperm_subind = np.random.permutation(len(source_list))
    valid_subind = np.arange(0, len(source_list),m_ratio, dtype = int)
    train_subind = np.arange(1, len(source_list),m_ratio, dtype = int)
    for i in range(m_ratio-1):
        train_subind = np.concatenate( (train_subind, np.arange(i+2, len(source_list),5)) )
    return train_subind, valid_subind

def save_history(file_path, acc, val_acc, loss, val_loss, cm):
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H-%M-%S")
    with open(file_path+"history-"+date_time+'.txt', "wb") as fp: 
        pickle.dump(acc, fp)
        pickle.dump(val_acc, fp)
        pickle.dump(loss, fp)
        pickle.dump(val_loss, fp)
        pickle.dump(cm, fp)
        fp.close()
        
def load_history(file_path):
    with open(file_path, "rb") as fp:
        acc = pickle.load(fp)
        val_acc = pickle.load(fp)
        loss = pickle.load(fp)
        val_loss = pickle.load(fp)
        cm = pickle.load(fp)
        fp.close()
    return acc, val_acc, loss, val_loss, cm

def cm2acc(cm):
    tp=0
    for i in range(cm.shape[0]):
        tp+=cm[i,i]
    return tp/(sum(sum(cm)))

def create_log(file_path, header = ['condition', 'best_valid_loss', 'best_valid_acc', 'test_acc']):
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H-%M-%S")
    f = open(file_path+"log-"+date_time+'.csv', 'w',
                        encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(header)
    f.close()
    return f
def write_log_line(f, line =  ['condition', 'best_valid_loss', 'best_valid_acc', 'test_acc']):
    f = open(f.name, 'a', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(line)
    f.close()
    return f
