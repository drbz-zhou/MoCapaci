import numpy as np
import argparse
import os
import sys

import torch.nn as nn
import torch
import time
import csv
from utils import *

from data_loader import Dataload_fabric
from model import DeepConvLSTM, DeepConv1D

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import math

# from cvtorchvision import cvtransforms


from sklearn.metrics import confusion_matrix


import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class TrainOps(object):
    def __init__(self, device, data_path, model, train_idx, valid_idx, test_idx, epochs, batch_size, lr):
        self.device = device
        self.data_path = data_path
        self.model = model
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_classes = 20
        
        self.model_save_path = './save_model/' + self.model + '_' + str(valid_idx[0]) + '_'+ str(test_idx[0]) + '.pth'
                
        self.window_size= 400
        self.channel_size= 4
        
    def train_model(self):
        print("Training " + self.model)

        train_dataset = Dataload_fabric(self.data_path, self.train_idx, self.window_size, self.channel_size)
        train_loader = DataLoader(train_dataset, batch_size = self.batch_size, shuffle=True)

        valid_dataset = Dataload_fabric(self.data_path, self.valid_idx, self.window_size, self.channel_size)
        valid_loader = DataLoader(valid_dataset, batch_size = self.batch_size, shuffle=True)

        test_dataset = Dataload_fabric(self.data_path, self.test_idx, self.window_size, self.channel_size)
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

        if self.model == 'DeepConvLSTM':
            net = DeepConvLSTM(n_classes=self.n_classes, n_conv=4, n_layers=2, window_size=self.window_size, n_channels=self.channel_size, drop_prob = 0.5, filter_size=41)
            net.apply(init_weights)
        elif self.model == '1DConv':
            net = DeepConv1D(n_classes=self.n_classes, window_size=self.window_size, n_channels=self.channel_size, drop_prob = 0.2, filter_size=41)
            net.apply(init_weights)

        net.to(self.device,dtype=torch.float64)
        
        if self.model == 'DeepConvLSTM':
            optimizer = torch.optim.AdamW(net.parameters(),lr=self.lr,weight_decay=3e-8,amsgrad=True)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,100) # Learning rate scheduler to reduce LR every 100 epochs
            # criterion = nn.CrossEntropyLoss()
            
        elif self.model == '1DConv':
            optimizer = torch.optim.Adam(net.parameters(), lr = self.lr, betas=(0.9, 0.99))
            criterion = nn.CrossEntropyLoss()
            # criterion = nn.BCELoss()
        
        train_label = train_dataset.label.reshape(-1,1)
        valid_label = valid_dataset.label.reshape(-1,1)
        test_label = test_dataset.label.reshape(-1,1)
        # train_label = np.eye(self.n_classes)[train_dataset.label]
        # valid_label = np.eye(self.n_classes)[valid_dataset.label]
        

        print(train_label.shape)

        train_stats = np.unique([a for y in train_label for a in y],return_counts=True)[1]
        val_stats = np.unique([a for y in valid_label for a in y],return_counts=True)[1]
        test_stats = np.unique([a for y in test_label for a in y],return_counts=True)[1]

        print('Training set statistics:')
        print(len(train_stats),'classes with distribution',train_stats)
        print('Validation set statistics:')
        print(len(val_stats),'classes with distribution',val_stats)
        print('Test set statistics:')
        print(len(test_stats),'classes with distribution',test_stats)

        weights = torch.tensor([max(train_stats)/i for i in train_stats],dtype=torch.double)

        cuda = True if torch.cuda.is_available() else False

        if cuda:
            weights = weights.cuda()
        
        criterion = nn.CrossEntropyLoss(weight=weights) # Prepare weighted cross entropy for training and validation.
        val_criterion = nn.CrossEntropyLoss()
                
        if cuda:
            criterion.cuda()
            val_criterion.cuda()
        
        best_valid_loss = 999
        best_valid_acc = 0
        start_time = time.time()

        print('length of train_loader', len(train_loader))
        print('length of valid_loader', len(valid_loader))
        # print('length of test_loader', len(test_loader))

        for epoch in range(self.epochs):
            net.train()
            net.requires_grad_(True)

            step = 0
            epoch_loss = 0
            train_acc = 0
            total_inputs_len = 0

            for x,y in train_loader:
                x = x.to(device=self.device, dtype=torch.float64)
                # if self.model == '1DConv':
                #     target = one_hot_embedding(y.long(), self.n_classes)
                #     target = target.to(device=self.device)

                y = y.to(device=self.device)
                    
                
                # if (step == 0 or step == (len(train_loader)-1)) and self.model == 'DeepConvLSTM':
                #     h = net.init_hidden(x.size(0))
                
                if self.model == 'DeepConvLSTM':
                    h = net.init_hidden(x.size(0))
                    h = tuple([each.data for each in h])
                    output, h = net(x,h,x.size(0)) # Run inputs through network
                    # y = torch.reshape(y, (-1,1))

                    loss = criterion(output, y.long())
                elif self.model == '1DConv':
                    output = net(x)
                    loss = criterion(output, y.long())
                    # loss = criterion(output, target.double())

                epoch_loss = epoch_loss + loss.item()
                _, preds = torch.max(output, 1)
                corr_sum = torch.sum(preds == y)
                train_acc = train_acc + corr_sum.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step+=1
                total_inputs_len += x.size(0)

            epoch_loss = epoch_loss / step
            train_acc = train_acc / total_inputs_len
            
            if self.model == 'DeepConvLSTM':
                scheduler.step()

            # net.eval()
            # net.requires_grad_(False)
            step_val = 0
            valid_loss = 0
            valid_acc = 0
            total_inputs_len = 0
            with torch.no_grad():
                for x,y in valid_loader:
                    x = x.to(device=self.device, dtype=torch.float64)
                    # if self.model == '1DConv':
                    #     target = one_hot_embedding(y.long(), self.n_classes)
                    #     target = target.to(device=self.device)
                        
                    y = y.to(device=self.device)


                    # if step_val == 0 and self.model == 'DeepConvLSTM':
                    # if (step_val == 0 or step_val == (len(valid_loader)-1)) and self.model == 'DeepConvLSTM':
                    #     valid_h = net.init_hidden(x.size(0))
                    
                    if self.model == 'DeepConvLSTM':
                        valid_h = net.init_hidden(x.size(0))
                        output, valid_h = net(x,valid_h,x.size(0))

                        loss = val_criterion(output, y.long())    

                    elif self.model == '1DConv':
                        output = net(x)
                        loss = val_criterion(output, y.long())
                        # loss = criterion(output, target.double())
                    
                    _, preds = torch.max(output, 1)
                    corr_sum = torch.sum(preds == y)
                    
                    valid_loss = valid_loss + loss.item()
                    valid_acc = valid_acc + corr_sum.item()

                    step_val+=1
                    total_inputs_len += x.size(0)
            
            valid_loss = valid_loss / step_val
            valid_acc = valid_acc / total_inputs_len

            if valid_acc > best_valid_acc:
                best_epoch = epoch
                best_valid_loss = valid_loss
                best_valid_acc = valid_acc

                print('Best Valid Loss={:.4f} Best validation accuracy={:.4f}'.format(
                    best_valid_loss, best_valid_acc))

                torch.save(net.state_dict(), self.model_save_path)
            elif valid_acc == best_valid_acc and valid_loss < best_valid_loss:
                best_epoch = epoch
                best_valid_loss = valid_loss
                best_valid_acc = valid_acc

                print('Best Valid Loss={:.4f} Best validation accuracy={:.4f}'.format(
                    best_valid_loss, best_valid_acc))

                torch.save(net.state_dict(), self.model_save_path)

            elapsed_time = time.time() - start_time
            print('Epoch {}/{} time={:.2f}s train_loss={:.4f} train_acc={:.4f} valid_loss={:.4f} valid_acc={:.4f}'.format(
                                    epoch +1, self.epochs,
                                    elapsed_time,
                                    epoch_loss, train_acc, valid_loss, valid_acc))
            
        return best_epoch, best_valid_loss, best_valid_acc

    def test_model(self):
        print("Testing " + self.model)

        test_dataset = Dataload_fabric(self.data_path, self.test_idx, self.window_size, self.channel_size)
        test_loader = DataLoader(test_dataset, batch_size = self.batch_size, shuffle=True)

        if self.model == 'DeepConvLSTM':
            net = DeepConvLSTM(n_classes=self.n_classes, n_conv=4, n_layers=2, window_size=self.window_size, n_channels=self.channel_size, drop_prob = 0.2, filter_size=41)
            
        elif self.model == '1DConv':
            net = DeepConv1D(n_classes=self.n_classes, window_size=self.window_size, n_channels=self.channel_size, drop_prob = 0.2, filter_size=41)

        net.load_state_dict(torch.load(self.model_save_path))
        net.requires_grad = False
        net.to(self.device,dtype=torch.float64)
        
        cuda = True if torch.cuda.is_available() else False
        
        if self.model == 'DeepConvLSTM':
            criterion = nn.CrossEntropyLoss()
        elif self.model == '1DConv':
            criterion = nn.CrossEntropyLoss()
            # criterion = nn.BCELoss()
        
        if cuda:
            criterion.cuda()

        net.train()

        step_test = 0
        test_loss = 0
        test_acc = 0
        total_inputs_len = 0
        with torch.no_grad():
            for x,y in test_loader:
                x = x.to(device=self.device, dtype=torch.float64)
                y = y.to(device=self.device)

                # if self.model == '1DConv':
                #     target = one_hot_embedding(y.long(), self.n_classes)
                #     target = target.to(device=self.device)

                if self.model == 'DeepConvLSTM':
                    test_h = net.init_hidden(x.size(0))
                    output, test_h = net(x,test_h,x.size(0))
                    loss = criterion(output, y.long())    

                elif self.model == '1DConv':
                    output = net(x)
                    loss = criterion(output, y.long())
                    # loss = criterion(output, target.double())
                
                _, preds = torch.max(output, 1)
                corr_sum = torch.sum(preds == y)

                test_loss = test_loss + loss.item()
                test_acc = test_acc + corr_sum.item()
                
                step_test+=1
                total_inputs_len += x.size(0)
        
        test_loss = test_loss / step_test
        test_acc = test_acc / total_inputs_len
        
        print('Test Loss={:.4f} Test accuracy={:.4f}'.format(
                    test_loss, test_acc))
        
        # cm = confusion_matrix(y_test, np.argmax(y_predict, axis=1))
        # tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path=outfolder+'LPO_'+str(m_test)+'_'+str(m_valid)+'_'+model_type,acc=acc_test)
        # cm_all = np.concatenate( (cm_all, np.expand_dims(cm,2)), 2)
        
        # tools.save_history(outfolder+'LPO_'+str(m_test)+'_'+str(m_valid)+'_'+model_type, acc, val_acc, loss, val_loss, cm)
        
        # print(acc_test)
        # print(cm)
        return test_loss, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0', help="GPU number")
    parser.add_argument("--data_path", type=str, default='/mnt/nas2/data/activity_recognition/fabric/', help="data path")
    parser.add_argument("--epoch", type=int, default=300, help="epoch")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning_rate")
    parser.add_argument("--model", type=str, default='DeepConvLSTM', help=" 'DeepConvLSTM', '1DConv', 'UNetLSTM'")
    
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device : ")
    print(device)
    # device = torch.device("cuda:%s" % args.gpu)

    train_idx = [0,2,4,5,6,7,8]
    valid_idx = [3]
    test_idx = [1]
    n_population = 10
    n_iteration = 0
    for m_test in range(0, n_population):
        train_idx = list(range(n_population))
        train_idx.remove(m_test)
        test_idx = [m_test]

        for m_valid in train_idx:
            n_iteration+=1
            train_idx = list(range(n_population))
            train_idx.remove(m_test)
            train_idx.remove(m_valid)

            valid_idx = [m_valid]
            print('train_idx:',train_idx)
            print('valid_idx:',valid_idx)
            print('test_idx:',test_idx)
            
            train = TrainOps(device=device, data_path=args.data_path, model=args.model, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx, epochs=args.epoch, batch_size=args.batch_size, lr=args.lr)
            best_epoch, best_valid_loss, best_valid_acc = train.train_model()
            test_loss, test_acc = train.test_model()

            result_save_path = './testresult/' + args.model + '_' + str(n_population) + '_2maxpool.csv'
            if n_iteration == 1:
                f = open(result_save_path, 'w',
                        encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow(['n_iteration', 'm_valid', 'm_test', 'best_epoch', 'best_valid_loss', 'best_valid_acc', 'test_loss', 'test_acc'])
                wr.writerow([n_iteration, m_valid, m_test, best_epoch, best_valid_loss, best_valid_acc, test_loss, test_acc])
                f.close()
            else:
                f = open(result_save_path, 'a',
                        encoding='utf-8',
                        newline='')
                wr = csv.writer(f)
                wr.writerow([n_iteration, m_valid, m_test, best_epoch, best_valid_loss, best_valid_acc, test_loss, test_acc])
                f.close()