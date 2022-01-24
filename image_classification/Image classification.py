#!/usr/bin/env python
# coding: utf-8

# In[14]:


from matplotlib import image
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os


# In[46]:


# give labels some values
rootdir = "./Dataset/five_faces/"
i=0;
name_to_number={}
number_to_name={}
for folder in os.listdir(rootdir):
    name_to_number[str(folder)]=i
    number_to_name[i]=str(folder)
    i+=1
print(name_to_number)
print(number_to_name)
no_of_classes=len(number_to_name)


# # Store images as numpy arrays

# In[63]:


height=60
width=60
columns=3
no_of_features=height*width*columns
# no of features for one image=height*width*columns
"""
store all pixels of an image in list and also append the label in this list 
so the last element of this list is label
these lists of all images will be stored in data
"""
data=[] 
rootdir = "./Dataset/five_faces/"
for folder in os.listdir(rootdir):
    for file in os.listdir(rootdir+folder+"/"):
        name=rootdir+folder+"/"+file
        try:
            img=Image.open(name)
            img=img.convert('RGB')# from rgba to rgb conversion
            img=img.resize((width,height))
            label=name_to_number[str(folder)]
            a=np.array(img)
            l=a.flatten().tolist()
            l.append(label)
            data.append(l)
        except:
            pass


# In[64]:


data=np.array(data)
print(data.shape)


# In[65]:


# shuffle the dataset\
np.random.seed(0)# to obtain same shuffling each time
np.random.shuffle(data)

#distribute in training data and testing data
split=int(data.shape[0]*0.8)
x_train,y_train=data[0:split,:-1],data[0:split,-1]
x_test,y_test=data[split:,:-1],data[split:,-1]
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[66]:


x_test=x_test/255.0
x_train=x_train/255.0




def sigmoid(x):
    #sigmoid
    return 1/(1+np.exp(-x));
    #relu function
    y=x
    y[x<0]=0
    return y

def relu(x):
    #relu function
    y=x
    y[x<0]=0
    return y


def tan_inv(x):
    #tanh
    return np.tanh(x)

activations={'sigmoid':sigmoid,'relu':relu,'tanh':tan_inv}

def sigmoid_diff(x):
    # first differential of activation function
    #sigmoid:
    """
    sigmoid'(x)=sigmoid(x)*(1-sigmoid(x))
    """
    sig=sigmoid(x)
    return sig*(1-sig) # element wise multiplication
def relu_diff(x):
    # first differential of activation function
    #relu:
    y=x
    y[x<0]=0
    y[x>0]=1
    return y
def tanh_diff(x):
    # first differential of activation function
    #tanh
    y=tan_inv(x)
    return (1-np.square(y))
activations_differential={'sigmoid':sigmoid_diff,'relu':relu_diff,'tanh':tanh_diff}

def softmax(a):
    exp_a=np.exp(a)
    exp_sum=np.sum(exp_a,axis=1,keepdims=True)
    return exp_a/exp_sum


# In[113]:


class NeuralNetwork:
    def __init__(self,input_size,output_size,hidden,activation_function):
        #np.random.seed(0)
        model={}# dictionary
        """
        input layer--(W1,B1)-->layer1--(W2,B2)-->layer2--(W3,B3)-->output layer
        """
        
        # from input layer to layer 1
        model['W1']=np.random.randn(input_size,hidden[0])
        model['B1']=np.zeros((1,hidden[0]))
        
        # from layer 1 to layer 2
        model['W2']=np.random.randn(hidden[0],hidden[1])
        model['B2']=np.zeros((1,hidden[1]))
        
        # from  layer 2 to output layer
        model['W3']=np.random.randn(hidden[1],output_size)
        model['B3']=np.zeros((1,output_size))
        
        """
        for x in model:
            print(x,model[x].shape)
        """
        self.model=model
        self.activation=activations[activation_function]
        self.activation_diff=activations_differential[activation_function]
    def forward_propagation(self,X):
        """
        X--> Matrix of size m X n where m is the no of examples and n=input_size
        """
        model=self.model
        W1,W2,W3=model['W1'],model['W2'],model['W3']
        b1,b2,b3=model['B1'],model['B2'],model['B3']
        activation=self.activation
        
        z1=np.dot(X,W1)+b1
        a1=activation(z1)
        
        z2=np.dot(a1,W2)+b2
        a2=activation(z2)
        
        z3=np.dot(a2,W3)+b3
        a3=softmax(z3)
        #print(a3)
        #print(a3.sum(axis=1))
        self.activations=(a1,a2,a3,z1,z2,z3)
        return a3
        
    def backward_propagation(self,x,y,learning_rate=0.001,lambd=0.5):
        # lambd is the regularization hyper-parameter
        model=self.model
        W1,W2,W3=model['W1'],model['W2'],model['W3']
        b1,b2,b3=model['B1'],model['B2'],model['B3']
        (a1,a2,a3,z1,z2,z3)=self.activations
        activation_diff=self.activation_diff
        
        m=x.shape[0]
        delta3=a3-y
        dw3=np.dot(a2.T,delta3)/m+(lambd/m)*(W3)
        db3=np.sum(delta3,axis=0)/m
        
        
        delta2=np.dot(delta3,W3.T)*(activation_diff(z2))
        dw2=np.dot(a1.T,delta2)/m+(lambd/m)*(W2)
        db2=np.sum(delta2,axis=0)/m
        
        
        delta1=np.dot(delta2,W2.T)*(activation_diff(z1))
        dw1=np.dot(x.T,delta1)/m+(lambd/m)*(W1)
        db1=np.sum(delta1,axis=0)/m
        
        # update
        self.model["W3"]-=learning_rate*dw3
        self.model["B3"]-=learning_rate*db3
        
        self.model["W2"]-=learning_rate*dw2
        self.model["B2"]-=learning_rate*db2
        
        self.model["W1"]-=learning_rate*dw1
        self.model["B1"]-=learning_rate*db1
        


    def predict(self,x):
        y_=self.forward_propagation(x)
        return np.argmax(y_,axis=1);
    def summary(self):
        model=self.model
        W1,W2,W3=model['W1'],model['W2'],model['W3']
        print(W1,W2,W3)
        
    def loss(self,y_oht,y_,lambd=0.5):
        m=y_.shape[0]
        W1,W2,W3=self.model['W1'],self.model['W2'],self.model['W3']
        regularization_loss=lambd*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))/(2*m)
        return -np.mean(y_oht*np.log(y_))+regularization_loss
        
        

        


# In[114]:


# convert the simple labels into one hot vector
def convert_to_oht(y,no_of_classes):
    m=y.shape[0]
    y_oht=np.zeros((m,no_of_classes))
    y_oht[range(m),y]=1
    return y_oht

def train(model,X,y,no_of_classes,epochs=200,learning_rate=0.0001,lambd=0.5,logs=True,*args):
    div=100
    if(len(args)>0):
        div=int(args[0])
    
    y_oht=convert_to_oht(y,no_of_classes)
    losses=[]
    for i in range(epochs):
        y_=model.forward_propagation(X)
        l=model.loss(y_oht,y_,lambd)
        losses.append(l)
        if(logs and i%div==0):
            print("Iteration {}, loss= {}".format(i,l))
        model.backward_propagation(X,y_oht,learning_rate,lambd)
    return losses

def visualize_decision_boundry(X,Y,model):
    # X is only two featured
    [x0_min,x1_min]=np.min(X,axis=0)
    [x0_max,x1_max]=np.max(X,axis=0)
    x0=np.linspace(x0_min,x0_max,100)
    x1=np.linspace(x1_min,x1_max,100)
    x0,x1=np.meshgrid(x0,x1)
    x=np.zeros((x0.shape[0]*x0.shape[1],2))
    x[:,0]=x0.reshape(-1)
    x[:,1]=x1.reshape(-1)
    y_=model.predict(x)
    print(x.shape)
    print(y_.shape)
    
    plt.scatter(x[:,0],x[:,1],c=y_,cmap=plt.get_cmap('Accent'))
    plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.get_cmap("rainbow"))
    

        
        


# In[ ]:



model=NeuralNetwork(input_size=no_of_features,output_size=no_of_classes,hidden=[60,30],activation_function='sigmoid')
losses=train(model,x_train,y_train,no_of_classes,5000,0.01,0,True,100)
Y_=model.predict(x_train)
print(np.mean(Y_==y_train))
Y_=model.predict(x_test)
print(np.mean(Y_==y_test))




