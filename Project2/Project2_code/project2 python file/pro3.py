# 1) Use python library to extract images, and separate them into test and training set.
import sklearn.datasets as dataset
import numpy as np
import matplotlib.pyplot as plt

iris = dataset.load_iris()

iris_data = iris.data
iris_target = iris.target
print('iris data shape : ', iris_data.shape)
print('iris target shape : ', iris_target.shape)
print()

def separate(data,target,testset_rate=0.2):
    rate = testset_rate

    testset_idx = np.sort( np.random.choice(len(data), round( len(data)* rate),replace=False) )
    trainingset_idx = np.sort( np.setdiff1d(np.arange(len(data)), testset_idx) )

    print('testset_idx \n')
    print(testset_idx)
    print('\n trainingset_idx \n')
    print(trainingset_idx)

    test_data = data[testset_idx]
    test_target = target[testset_idx]

    training_data = data[trainingset_idx]
    training_target = target[trainingset_idx]

    print('\n test set shape:  ', test_data.shape, test_target.shape )
    print('\n training set shape:  ', training_data.shape, training_target.shape )
    return test_data, test_target, training_data,training_target

test_data, test_target, training_data, training_target = separate(iris_data, iris_target)

# 2) Select any two datasets for training and test.
random_index1 = np.random.choice(len(training_data),1)
training_sample1 = training_data[random_index1]
training_target1 = training_target[random_index1]
random_index2 = np.random.choice(len(test_data),1)
test_sample1 = test_data[random_index2]
test_target1 = test_target[random_index2]
print(training_sample1, training_target1)
print(test_sample1, test_target1)

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))     # sigmoid

# sigmoid 미분
def differ_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x)) 

def differ_MSE(label, target):
    return label - target

class NN():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # weights
        self.weight = np.random.randn(input_size, output_size)
        print('input size:    ', self.input_size)
        print('weight1 shape: ', self.weight.shape)
        print('output size:   ', self.output_size)

    # Forward propagation
    def forward(self, x):
        self.output_in  = np.dot(x, self.weight)
        self.output_out = sigmoid(self.output_in)
        return self.output_out
    
    def backward(self, input_, label, predict): 
        error_diff = differ_MSE(predict, label)                   # loss function differectial         
        sig_diff = error_diff * differ_sigmoid(predict)           # sigmoid differectial
        gradient = np.expand_dims(input_, axis=0).T @ np.expand_dims(sig_diff, axis=0)             # sum differential            
        self.weight -= gradient * 0.1                             # learning rate = 0.1   
        
input_size, output_size= 5, 3
NN = NN(input_size, output_size)

# 학습 이전
def one_hot_encoding(input_, num_of_class):
    encoding = np.zeros(num_of_class)
    encoding[input_]=1
    return encoding

input_ = np.append(training_sample1, 1)
print('input:   ', input_)
target= one_hot_encoding(training_target1, 3)
print('target:   ', target)
predict = NN.forward(input_)
print('predict: ', np.round_(predict, 4))
print()

input_ = np.append(test_sample1, 1)
print('input:   ', input_)
target= one_hot_encoding(test_target1, 3)
print('target:   ', target)
predict = NN.forward(input_)
print('predict: ', np.round_(predict, 4))

# MSE(loss function)
def MSEloss(label, predict):
    return np.mean(np.square(label - predict))

epoch = 500
for j in range(epoch):
    print('='*80)
    print('epoch: ', j+1)
    number = 0
    average_loss = 0
    for i in range(len(training_data)):                   # training set
        input_ = training_data[i]
        input_ = np.append(input_, 1)                      # input에 bias 추가
        label1_ = training_target[i]
        label1_ = one_hot_encoding(label1_,3)              # target one-hot encoding
        predict = NN.forward(input_)                       # predict
        correct = np.argmax(predict)==np.argmax(label1_)   # 정답 여부 확인
        if correct:
            number+=1
        if i%20 == 0:                                      # for visualization
            print(i,'번째 predict:', np.round_(predict, 3),'    label:', label1_, correct) 
        average_loss += MSEloss(label1_, predict)           # loss 누적
        NN.backward(input_, label1_, predict)               # back-propagation, 학습
    print()
    print('='*80) 
    print('< training set >')                               # for visualization
    print('average loss: ', 100*(average_loss/len(training_data)))
    print('accuracy: ', 100*(number/len(training_data)), '%')
    
    number = 0
    for i in range(len(test_data)):                         # test set
        input_ = test_data[i]
        input_ = np.append(input_, 1) 
        label1_ = test_target[i]
        label1_ = one_hot_encoding(label1_,3)
        predict = NN.forward(input_)
        correct = np.argmax(predict)==np.argmax(label1_)
        if correct:
            number+=1
        average_loss += MSEloss(label1_, predict)           # back-propagation 삭제됨
    print()
    print('='*80) 
    print('< test set >')
    print('average loss: ', 100*(average_loss/len(test_data)))
    print('accuracy: ', 100*(number/len(test_data)), '%')
    