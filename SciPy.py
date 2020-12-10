#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.integrate import nquad


# In[3]:


def f(x):
    return x


# In[4]:


area = quad(f, 0, 2)
print(area)


# In[5]:


def f(x,y):
    return x*y

area = dblquad(f,0,2,0,5)


# In[ ]:


area


# In[ ]:


f = lambda x,y: x*y
area = dblquad(f,0,2,0,5)


# In[ ]:


area


# In[ ]:


y_low_limit = lambda y: y


# In[ ]:


y_upp_limit = lambda y: y*5


# In[ ]:


area = dblquad(f, 0, 2, y_low_limit, y_upp_limit)


# In[ ]:


area


# In[ ]:


import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.integrate import nquad
print('Setup')


# In[ ]:


def f(x):
    return x


# In[ ]:


area = quad(f, 0,2)


# In[ ]:


print(area)


# In[ ]:


def f(x,y):
    return x*y
area = dblquad(f,0,2,0,5)
print(area)


# In[ ]:


area = dblquad(f,0,2,0,5)

print(area)
# In[ ]:


def f(x,y):
    return x*y

area = dblquad(f, 0, 2, 5, 7)
print(area)


# In[ ]:


y_low_limit = lambda y:y
y_upp_limit = lambda y: y*5


# In[ ]:


area = dblquad(f, 0, 2, y_low_limit, y_upp_limit)


# In[ ]:


area


# In[ ]:


def f(x,y,z):
    return x*y*z
area = nquad(f,[[0,2], [2,4], [4,6]])


# In[ ]:


area


# In[ ]:


from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.integrate import nquad


# In[ ]:


f = lambda x,y,z: x*y*z
area = nquad(f, [[0,2], [2,4], [4,6]])


# In[ ]:


area


# In[ ]:


from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.integrate import nquad
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy


# In[ ]:


x = np.linspace(-1, 3, 1000)
def f(x):
    return x**2
plt.plot(x,f(x))
plt.show()


# In[ ]:


# To See the Area Between the curve
plt.plot(x, f(x))
plt.axhline(color = 'k')
print(plt.show())


# In[ ]:


plt.plot(x,f(x))
plt.axhline(color = 'k')
plt.fill_between(x, f(x), where =[(x > 0) & (x < 2.5) for x in x], color = 'g')


# In[ ]:


help()


# In[ ]:


help()


# In[ ]:


x = sy.Symbol('x')
def f(x): return x**2
sy.integrate(f(x), (x,0,2))


# In[2]:


from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.integrate import nquad
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy


# In[13]:


def g(x):
    return x**(1/2)
x = np.linspace(-1, 1.5,100)


# In[25]:


plt.plot(x, g(x), color = 'k', linewidth = 2.5)
plt.axhline(color = 'k')
plt.fill_between(x,g(x), where = [(x > 0 ) and (x < 1.2) for x in x], color = 'g')
plt.show()


# In[28]:


from scipy import integrate
from scipy import cluster
from scipy import optimize
from scipy import linalg
from scipy.integrate import quad


# In[30]:


def IntegrateFunction(x):
    return x
quad(IntegrateFunction,0,1)


# In[31]:


def F(x,a,b):
    return x*a+b

a = 3
b = 2


# In[35]:


quad(F, 0, 1, args= (5,3))


# In[9]:


import numpy as np
from scipy.optimize import minimize


# In[10]:


def Cal(x):
    lenght = x[0]
    width = x[1]
    height = x[2]
    volume = lenght*width*height
    return volume
def Calc(x):
    lenght = x[0]
    width = x[1]
    height = x[2]
    SurfaceArea = 2*lenght*width + 2*lenght*height + 2*height*width
    return SurfaceArea


# In[ ]:





# In[14]:


def objective(x):
    return - Cal(x)


# In[15]:


def constraints(x):
    return 100 - Calc(x)


# In[16]:


cons = ({'type': 'ineq', 'fun': constraints})


# In[17]:


LG = 10
WG = 10
HG = 10
x0 = np.array([LG, WG, HG])


# In[20]:


sol = minimize(objective, x0, method='SLSQP', constraints=cons, options={'disp': True})


# In[6]:


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
def f(x,a,b):
    return a*np.exp(b*x)

xData = np.array([1,2,3,4,5])
yData = np.array([1,9,50,300,1500])

plt.plot(xData,yData,  label = 'Experimental Data')
plt.show()


# In[7]:


popt,pcov = curve_fit(f, xData, yData)


# In[8]:


print(popt)


# In[9]:


from scipy import linalg


# In[10]:


mat = np.array([[10,6],[2,7]])
mat


# In[11]:


type(mat)


# In[12]:


linalg.inv(mat)


# In[15]:


linalg.det(mat)


# In[16]:


# Solving Linear Equation


# In[17]:


numArray = np.array([[2,3,1], [-1, 5, 4], [3, 2, 9]])
num = np.array([21,9,6])


# In[18]:


numArray, num


# In[19]:


ab = np.linalg.solve(numArray,num)


# In[20]:


ab


# In[21]:


np.allclose(np.dot(numArray,ab), num)


# In[22]:


nums = np.array([[3,5,1], [9,5,7]])


# In[23]:


nums.shape


# In[24]:


linalg.svd(nums)


# In[25]:


test = np.array([[5,5], [6,7]])


# In[26]:


linalg.svd(test)


# In[27]:


linalg.eig(test)


# In[33]:


x = np.array([[1,2], [4,5]])


# In[31]:


y = np.array([5,6])


# In[34]:


x


# In[35]:


y


# In[36]:


linalg.solve(x,y)


# In[37]:


linalg.inv(x)


# In[39]:


linalg.svd(x)


# In[40]:


linalg.det(x)


# In[41]:


from scipy.stats import norm


# In[43]:


norm.rvs(loc =0, scale =1, size = 10)


# In[44]:


norm.cdf(5, loc = 1, scale = 2)


# In[45]:


norm.pdf(9, loc = 0, scale = 1)


# In[ ]:




