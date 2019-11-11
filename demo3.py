import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import pickle

# 线性函数测试
def sin_test():
    x = np.arange(0,6,0.1)
    y = np.sin(x)
    plt.plot(x,y)
    plt.show()


# 非线性函数测试
def sin_cos_test():
    x = np.arange(0,6,0.1)
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x,y1,label='sin')
    plt.plot(x,y2,linestyle='--',label='cos')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('x&y')
    plt.legend()
    plt.show()


# 展示图片
def show_img():
    img = imread('d:\show.png')
    plt.imshow(img)
    plt.show()


# 使用偏差值代替权重
def sum_b_test(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1


# 与门
def and_test(x1,x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


# 与非门
def no_and_test(x1,x2):
    w1, w2, theta = -0.5, -0.5, -0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


# 或门
def or_test(x1,x2):
    w1,w2,b = 0.5,0.5,-0.2
    tmp = x1*w1+x2*w2+b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

# 异或门制作
def no_or_test(x1,x2):
    a = or_test(x1,x2)
    b = no_and_test(x1,x2)
    return and_test(a,b)

# sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU 激活函数
def relu(x):
    return np.maximum(0, x)


# 阶跃函数
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


# 阶跃函数升级版
def step_function_plus(x):
    y = x > 0
    return y.astype(np.int)


# 阶跃函数终极版
def step_function_plus_edit(x):
    return np.array(x > 0, dtype=np.int)

# 展示阶跃函数与sigmoid区别
def show_step_sigmoid():
    test = np.arange(-6,6,0.1)
    test1 = sigmoid(test)
    test2 = step_function_plus_edit(test)
    plt.plot(test,test1,label='sigmoid')
    plt.plot(test,test2,linestyle='--',label='step')
    plt.legend()
    plt.ylim(-0.1,1.1)
    plt.show()


# 多维数组测试
def array_test():
    a = np.array([1,2,3,4])
    b = np.array([[1,2],[3,4],[5,6]])
    print(a)
    print(np.ndim(a))
    print(a.shape)
    
    print(b)
    print(np.ndim(b))
    print(b.shape)

# 矩阵之间相乘测试
def matrix_test():
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    c = np.dot(a,b)
    print(c)


# 2*3与3*2矩阵之间相乘测试
def matrix_two_test():
    a = np.array([[1,2,3],[4,5,6]])
    b = np.array([[1,2],[3,4],[5,6]])
    c = np.dot(b,a)
    print(c)


# 神经网络的矩阵表现形式
def matrix_to_tens_test():
    x = np.array([0.5,1.0]) # 输入值
    w = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]]) # 权重
    b = np.array([0.1,0.2,0.3]) # 偏差值
    a = np.dot(x,w) + b 
    z = sigmoid(a) ## 第一层隐藏层通过激活函数处理后的结果
    # plt.plot(x,a,label='old')
    # plt.plot(x,a1,label='new',linestyle='--')
    # plt.legend()
    # print(a)
    # print(z)
    ## 开始第二层的处理
    w2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    b2 = np.array([0.1,0.2])
    a2 = np.dot(z,w2) + b2
    z2 = sigmoid(a2) ## 第二层隐藏层通过激活函数处理后的结果
    print(a2)
    print(z2)

# softmax函数 问题：当a里有数值特别大的参数时，可能会导致函数溢出
def softmax_function(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# softmax函数 解决溢出问题
def softmax_function_plus(a):
    c = np.max(a)
    exp_a = np.exp(a-c) # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# 对mnist数据的测试
def mnist_demo():
    (x_train, t_train),(x_test, t_test) = load_mnist(flatten=True,normalize=False) # flatten为True时读入的图片是一个一维数组
    plt.figure(figsize=(10,10))
    for i in range(25):
        img = x_train[i].reshape(28,28) # 将图片变为原来的尺寸
        plt.subplot(5,5,i+1)
        plt.xticks([])
        # plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap=plt.cm.binary)
        plt.xlabel(t_train[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

## ----  下面是对已有权重参数进行展示 ---- ##

def get_data():
    (x_train, t_train),(x_test, t_test) = load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network


def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = sigmoid(a3)
    return y


def predict_demo():
    x,t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network,x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1
    print('Accuracy:'+str(float(accuracy_cnt) / len(x)))

## ----------------------------------- ##

predict_demo()

# print(matrix_to_tens_test())