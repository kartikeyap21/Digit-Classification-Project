#!/usr/bin/env python
# coding: utf-8

# In[1]:


a="hello world"
print(a)
r=[1,2]
print(r[-1])


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.use()
ar.shape()
np.zeros(2)
np.zeros_like(arr)
plt.scatter(x_train,y_train,label="data", marker="x")
plt.plot()
plt.legend
lab_utils_uni; %matplotlib widget
plt.close('all') 
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)
plt_gradient; 
plt_contour_wgrad
fig,ax= plt.subplots
fig.suptitle
plt_divergence,plt_contour_wgrad,plt_gradients,%matplotlib widget
np.arange
np.concatenate((a,b),axis=0)
np.dot
time.time
ar.reshape(-1,#col)
_, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)
w_norm, b_norm, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)
mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
np.ptp(X_train,axis=0)
norm_plot
           np.random.seed(12)#fixes 12
arr=np.random.randint(1,10)
x_trn=np.random.randint(2,8,size=(3,3))
print(arr,x_trn)
ar=np.random.randint(1,50,5)# returns 5 integers between 1 and 50

r=np.random.randn(3,3,3)# 3, 3/3 array  
print(r)
           


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'widget')
from plt_logistic_loss import plt_simple_examples soup_bowl
from plt_logistic_loss import plt_logistic_squared_error


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'widget')
from plt_logistic_loss import plt_simple_example plt_two_logistic_loss_curves soup_bowl
from plt_logistic_loss import plt_logistic_squared_error plt_
x_train=np.array([1,10,1])
y_train=([0,9,8])
plt.plot(x_train,y_train)
plt.title('practice')
plt.scatter(x_train,y_train)
plt.hist()



# In[ ]:


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
x=[1,2,3];y=[2,4,5]
plt.plot([1,2,3],[2,4,5],label='first' ,linewidth='3')
fig,ax=plt.subplot(1,2)

x1=[1,3,4]; y1=[1,2,4]
plt.title('kakis data')
plt.xlabel(' x label')
plt.ylabel(' y label')
plt.legend()
plt.scatter(x,y,marker='x', c='r')
plt.show()


# In[3]:


x_train=[1,3,4,5]
y_train=[2,4,5,0]
fig,ax=plt.subplots(2,2)
print(len(ax))
for i in range(len(ax)):
    ax[i].scatter(x_train,y_train)
    ax[i].set_xlabel('x-axis')
    ax[i].set_ylabel('y-axis')

# ax[0].scatter(x_train,y_train)
# ax[0].set_xlabel('X axis')

# plt.plot(x_train,y_train)
# plt.subplot(2,2,1)
# x=[1,2,3,4]
# plt.plot(x,y_train)
# plt.subplot(2,2,2)


# In[19]:


import numpy as np 
np.random.seed(12)#fixes 12
arr=np.random.randint(1,10)
x_trn=np.random.randint(2,8,size=(3,3))
print(arr,x_trn)
ar=np.random.randint(1,50,5)# returns 5 integers between 1 and 50
print(ar)
r=np.random.randn(3,3,3)# 3, 3/3 array  
print(r)


# In[20]:


r=np.random.random((3,3))# 3/3 array  
print(r)
r=np.random.


# In[21]:


r=np.array([1,2,4,5])
_r=np.arange(0,10)# 0 to 9 , 10 elements
rc=np.reshape(_r,(2,5))#reshaped array
r=np.arange(7)
np.resize(r,(3,4))
print(rc)
print(_r)


# In[18]:


from lab_utils_common import sigmoid
r=np.arange(7)
s=sigmoid(r,w)


np.resize(r,(3,4))#repitition in large sized arry



# In[ ]:





# In[ ]:




