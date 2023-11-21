# -*- coding: utf-8 -*-

import numpy as np 
import scipy.special
import pandas as pd
import matplotlib.pyplot as plt

training_data_file = open('C:/Users/electronic/Desktop/week11/mnist_train.csv', 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

test_data_file = open('C:/Users/electronic/Desktop/week11/mnist_test.csv', 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

test_number = 1000 #test set 중에 1000개만 사용할거라 개수 비교할 test_number 선언
training_number = 10000 #training set 중에 5000개만 사용할거라 개수 비교할 training_number 선언

epoch = 10 # epoch
learning_rate = 0.01 # learning rate
input_layer = 784
hidden_layer1 = 400
hidden_layer2 = 200
output_layer = 10

w1 = np.random.normal(0,np.sqrt(2/(input_layer+hidden_layer1)),size=(input_layer, hidden_layer1))
w2 = np.random.normal(0,np.sqrt(2/(hidden_layer1+hidden_layer2)),size=(hidden_layer1, hidden_layer2))
w3 = np.random.normal(0,np.sqrt(2/(hidden_layer2+output_layer)),size=(hidden_layer2, output_layer))
#w1,w2,w3 Xavier 초기화

classification_accuracy_per_epoch = np.array([]) #epoch별 분류정확도 저장할 깡통 array
total_cost_per_epoch = np.array([]) # epoch별 cost 저장할 깡통 array
n_epoch = np.array([]) #epoch번호 저장할 깡통 array

def activation (z):
    g_z = np.tanh(z) # 활성화함수가 tanh 일 때 사용
#    g_z = scipy.special.expit(z) # 활성화함수가 sigmoid일 때 사용
    return g_z

def softmax(z):
    g_z = np.exp(z)/np.sum(np.exp(z)) # 출력층에서 사용하는 softmax 함수 
    return g_z


def feedforward (feature,w1,w2,w3): # Feedforward
    x = np.array(feature, ndmin = 2)
    z2 = np.dot (x,w1)
    x2 = activation (z2)
    z3 = np.dot (x2, w2)
    x3 = activation(z3)
    z4 = np.dot (x3, w3)
    h = softmax (z4) # softmax
    return x2,x3, h

def backpropagation (h,d,x1,x2,x3): # Backpropagation을 통한 학습
    # 작성 필요
    x = np.array(x1,ndmin =2)
    dd = np.array(d,ndmin = 2)
    e4 = h - dd
    delta4 = e4
    e3 = np.dot(delta4, w3.T)
    delta3 = e3*(1-x3*x3)
    e2 = np.dot(delta3, w2.T)
    delta2 = e2*(1-x2*x2)
    
    w3_update = -learning_rate*np.dot(x3.T, delta4)
    w2_update = -learning_rate*np.dot(x2.T, delta3)
    w1_update = -learning_rate*np.dot(x.T,delta2)
    
    return w1_update, w2_update, w3_update

def inference (h): # 정답 추론
    inferred_label = np.argmax(h)
    return inferred_label

scorecard = np.array([]) #data 하나씩 feedforward하면서 정답 여부 기록할 깡통
cost = np.array([]) #data 하나씩 feedforward해서 cost 기록할 깡통

number=0 # 학습전 test에 사용할 number (개수)
for record in test_data_list: # test data를 한 줄씩 열어서 

        all_values = record.split(',') # 로 나누고
        feature = np.asfarray(all_values[1:])/255*0.99+0.01 # pixel값 0.01~1로 normalize
        correct_label = int(all_values[0]) # 정답 추출
        d = np.zeros(10)
        d [correct_label] = 1 # 정답 벡터 one-hot encoding

        x2, x3, h = feedforward (feature,w1,w2,w3) # data feedforward
#        print (h)
 
        if (inference (h)==correct_label):
            scorecard = np.append(scorecard,1) #추론한 정답이 correct_label과 같으면 scorecard에 1추가
        else:
          scorecard = np.append(scorecard,0) # 틀리면 0 추가
        cost_example = np.sum(-d*np.log(h)-(1-d)*np.log(1-h))  # 해당 data example의 cost 계산
        cost = np.append(cost, cost_example) # cost 라는 깡통 array에 example 의 cost 값 추가
        
        number = number+1 # 1 data test가 끝났으니 number에 1추가
        if (number==test_number):# number가 처음설정한 test number와 같게되면 for문 탈출
            break

classification_accuracy = np.sum(scorecard)/test_number # scorecard를 합산한 후 test_number로 나눠 종합적인 분류정확도 계산
classification_accuracy_per_epoch = np.append(classification_accuracy_per_epoch, classification_accuracy) #epoch별 분류정확도 array에 결과 추가
total_cost = np.sum(cost)/test_number # example별 cost 계산한 cost array 성분 합산 후 test number로 나누어 total cost 계산
total_cost_per_epoch = np.append(total_cost_per_epoch, total_cost) #epoch별 cost 저장할 total_cost_per_epoch에 결과 저장
n_epoch = np.append(n_epoch, 0) #epoch number 저장하는 n_epoch에 0 저장

print ('initial cost =  ', total_cost) # 초기 cost 출력
print ('initial_accuracy =  ', classification_accuracy) #초기 분류정확도 출력

for i in range(epoch):
    number_training = 0 #학습 data 개수 셀 변수
    number_test = 0 #test data 개수 셀 변수
    scorecard = np.array([]) #scorecard 초기화
    cost = np.array([]) #cost 초기화
    w1_update_temp = np.zeros([input_layer, hidden_layer1]) #batch update 위한 임시 행렬
    w2_update_temp = np.zeros([hidden_layer1, hidden_layer2])
    w3_update_temp = np.zeros([hidden_layer2, output_layer])
    
    batch = 5 #batch size
    batch_count= 0 #data 개수 세면서 batch와 같은지 즉 update 할 때가 되었는지 판단하는 데 사용할 변수
    for record in training_data_list: #training data를 한줄씩 읽어서 
        all_values = record.split(',') #, 로 숫자분리
    
        feature = np.asfarray(all_values[1:])/255*0.99+0.01 #pixel값 normalize
        correct_label = int(all_values[0]) #정답 추출
        d = np.zeros(10)+0.01 #정답 벡터 0.01로 초기화 
        d [correct_label]=0.99 # 정답인 성분도 1이 아니라 0.99로 초기화

        number_training = number_training + 1
        
        # 작성필요
        x2, x3, h = feedforward (feature, w1,w2,w3)
        w1_update, w2_update, w3_update = backpropagation (h,d,feature,x2,x3)
        
        w1_update_temp = w1_update_temp+w1_update
        w2_update_temp = w2_update_temp+w2_update
        w3_update_temp = w3_update_temp+w3_update
        batch_count = batch_count+1
        
        if (batch_count==batch): #batch_count 가 batch size와 같아지면
            w1 = w1+w1_update_temp/batch
            w2 = w2+w2_update_temp/batch
            w3 = w3+w3_update_temp/batch
            batch_count = 0
            w1_update_temp=np.zeros([input_layer, hidden_layer1])
            w2_update_temp=np.zeros([hidden_layer1, hidden_layer2])
            w3_update_temp=np.zeros([hidden_layer2, output_layer])
            
        if (number_training == training_number):
            break

    for record in test_data_list: # 1epoch마다 test data를 한줄씩 열어서 
        all_values = record.split(',')
    
        feature = np.asfarray(all_values[1:])/255*0.99+0.01
        correct_label = int(all_values[0])
            
        d = np.zeros(10)
        d [correct_label]=1

        x2, x3, h = feedforward(feature,w1,w2,w3)
        cost_example = np.sum(-d*np.log(h)-(1-d)*np.log(1-h))
        cost = np.append(cost, cost_example)
        number_test = number_test+1 #data 하나 평가마다 number_test 1증가
        if (inference (h)==correct_label):
            scorecard = np.append(scorecard,1)
        else:
            scorecard = np.append(scorecard,0)

        if (number_test==test_number): #number_test가 처음설정한 test_number와 같아지면 for문 탈출
            break

    classification_accuracy = np.sum(scorecard)/np.size(scorecard) # epoch별 분류 정확도 계산
    classification_accuracy_per_epoch = np.append(classification_accuracy_per_epoch, classification_accuracy) #깡통 array에 저장
    total_cost = np.sum(cost)/test_number #epoch별 total cost 계산
    total_cost_per_epoch = np.append(total_cost_per_epoch, total_cost) #깡통 array에 저장
    n_epoch = np.append(n_epoch, i+1) #epoch number 깡통 array에 저장
    print ('epoch:', i+1, 'classification accuracy', classification_accuracy) #epoch number와 분류정확도 출력

print ('Final cost =   ', total_cost) #최종 cost 출력
print ('Final_accuacy =   ', classification_accuracy) #최종 분류정확도 출력
plt.plot(n_epoch,classification_accuracy_per_epoch) # 그래프 출력
plt.plot(n_epoch,total_cost_per_epoch)
df = pd.DataFrame({'epoch': n_epoch, 'accuracy': classification_accuracy_per_epoch, 'cost:': total_cost_per_epoch})
df.to_csv('C:\\MNIST.csv', index=None)#epoch number, 정확도, cost 파일로 저장
plt.plot(n_epoch,classification_accuracy_per_epoch) # 그래프 출력
plt.plot(n_epoch,total_cost_per_epoch)

