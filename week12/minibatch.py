# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 01:23:54 2022

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

training_data_file=open('C:/Users/electronic/Desktop/week11/XOR_truthtable.csv','r')
training_data_list = training_data_file.readlines()
training_data_file.close()

    
epoch = 10000
learning_rate = 0.03
input_layer = 2
hidden_layer1 = 5
hidden_layer2 = 5
output_layer = 2

w1=np.random.normal(0,np.sqrt(2/(input_layer+hidden_layer1)),size=(input_layer, hidden_layer1))
w2=np.random.normal(0,np.sqrt(2/(hidden_layer1+hidden_layer2)),size=(hidden_layer1, hidden_layer2))
w3=np.random.normal(0,np.sqrt(2/(hidden_layer2+output_layer)),size=(hidden_layer2, output_layer))

classification_accuracy_per_epoch = np.array([])
total_cost_per_epoch = np.array([])
n_epoch = np.array([])

def activation (z):
    g_z = 1/(1+np.exp(-z))
    return g_z

def feedforward (feature, w1, w2, w3):
    x = np.array(feature, ndmin = 2)
    z2 = np.dot (x,w1)
    x2 = activation (z2)
    z3 = np.dot (x2, w2)
    x3 = activation(z3)
    z4 = np.dot (x3, w3)
    h = activation (z4)
    return x2, x3, h

def backpropagation (h,d,x1,x2,x3):
    x = np.array(x1,ndmin =2)
    dd = np.array(d,ndmin = 2)
    e4 = h-dd
    delta4 = e4
    e3 = np.dot (delta4, w3.T)
    delta3 = e3*x3*(1-x3)
    e2 = np.dot (delta3, w2.T)
    delta2 = e2*x2*(1-x2)
    
    w3_update = -learning_rate*np.dot(x3.T, delta4)
    w2_update = -learning_rate*np.dot(x2.T, delta3)
    w1_update = -learning_rate*np.dot(x.T,delta2)
    
    return w1_update, w2_update, w3_update

def inference (h):
    inferred_label = np.argmax(h)
    return inferred_label

scorecard = np.array([])
cost = np.array([])
for record in training_data_list:
    all_values = record.split (',')
    feature = np.asfarray(all_values[1:])
    correct_label = int(all_values[0])
    d = np.zeros(2)
    d [correct_label] = 1
    
    x2, x3, h = feedforward (feature, w1,w2,w3)
    print (h)
    
    if (inference(h)==correct_label):
        scorecard = np.append(scorecard,1)
    else: 
        scorecard = np.append(scorecard,0)
    cost_example = np.sum(-d*np.log(h)-(1-d)*np.log(1-h))
    cost = np.append(cost, cost_example)
classification_accuracy = np.sum(scorecard)/4
classification_accuracy_per_epoch = np.append(classification_accuracy_per_epoch, classification_accuracy)
total_cost = np.sum(cost)/4
total_cost_per_epoch = np.append(total_cost_per_epoch, total_cost)
n_epoch = np.append(n_epoch,0)

print ("initial cost: ", total_cost)
print ("initial accuracy: ", classification_accuracy)

for i in range(epoch):
    scorecard = np.array([])
    cost = np.array([])
    w1_update_temp=np.zeros([input_layer, hidden_layer1])
    w2_update_temp=np.zeros([hidden_layer1, hidden_layer2])
    w3_update_temp=np.zeros([hidden_layer2, output_layer])
    batch = 2 
    batch_count = 0
    
    for record in training_data_list:
        all_values = record.split (',')
        feature = np.asfarray(all_values[1:])
        correct_label = int(all_values[0])
        d = np.zeros(2)
        d [correct_label] = 1
        
        x2, x3, h = feedforward (feature, w1,w2,w3)
        w1_update, w2_update, w3_update = backpropagation (h,d,feature,x2,x3)
        
        w1_update_temp = w1_update_temp+w1_update
        w2_update_temp = w2_update_temp+w2_update
        w3_update_temp = w3_update_temp+w3_update
        batch_count = batch_count+1
        
        if (batch_count == batch):
            w1 = w1+w1_update_temp/batch
            w2 = w2+w2_update_temp/batch
            w3 = w3+w3_update_temp/batch
            batch_count = 0
            w1_update_temp=np.zeros([input_layer, hidden_layer1])
            w2_update_temp=np.zeros([hidden_layer1, hidden_layer2])
            w3_update_temp=np.zeros([hidden_layer2, output_layer])
        
    for record in training_data_list:
        all_values = record.split (',')
        feature = np.asfarray(all_values[1:])
        correct_label = int(all_values[0])
        d = np.zeros(2)
        d [correct_label] = 1
        x2, x3, h = feedforward (feature, w1,w2,w3)
        
        if (inference(h)==correct_label):
            scorecard = np.append(scorecard,1)
        else: 
            scorecard = np.append(scorecard,0)
        cost_example = np.sum(-d*np.log(h)-(1-d)*np.log(1-h))
        cost = np.append(cost, cost_example)
    classification_accuracy = np.sum(scorecard)/4
    classification_accuracy_per_epoch = np.append(classification_accuracy_per_epoch, classification_accuracy)
    total_cost = np.sum(cost)/4
    total_cost_per_epoch = np.append(total_cost_per_epoch, total_cost)
    n_epoch = np.append(n_epoch,i+1)

scorecard = np.array([])
cost = np.array([])
for record in training_data_list:
    all_values = record.split (',')
    feature = np.asfarray(all_values[1:])
    correct_label = int(all_values[0])
    d = np.zeros(2)
    d [correct_label] = 1
    x2, x3, h = feedforward (feature, w1,w2,w3)
    print (h)
    if (inference(h)==correct_label):
        scorecard = np.append(scorecard,1)
    else: 
        scorecard = np.append(scorecard,0)
    cost_example = np.sum(-d*np.log(h)-(1-d)*np.log(1-h))
    cost = np.append(cost, cost_example)
    
classification_accuracy = np.sum(scorecard)/4
total_cost = np.sum(cost)/4
print ("Final cost: ", total_cost)
print ("Final accuracy: ", classification_accuracy)
plt.plot(n_epoch,classification_accuracy_per_epoch)
plt.plot(n_epoch,total_cost_per_epoch)
