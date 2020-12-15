#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Numpy 

"""it is used for Linear Algerbra purpose
it has binding with c libaray
to create array, vector(1D), matrices(2D), higer dimesion and do some of the operations
"""


# In[3]:


import numpy as np


# In[4]:


list1 = [1,2,3]


# In[5]:


list1


# In[6]:


New = np.array(list1)


# In[7]:


New


# In[8]:


list2 = [[1, 2, 3], [4, 5, 6]]


# In[9]:


New2 = np.array(list2)


# In[10]:


New2


# In[11]:


#arange Function 


# In[13]:


#It is used to find out the first element, last element and the steps taken


# In[15]:


New3 = np.arange(1,10,2)


# In[16]:


New3


# In[17]:


N1 = range(1, 10, 1)


# In[18]:


N1


# In[19]:


tuple(N1)


# In[20]:


np.zeros(6)


# In[21]:


np.zeros([5,5])


# In[22]:


#linspace divides the whole thing into equal parts


# In[27]:


np.linspace(0,10,10)


# In[28]:


np.eye(5)


# In[30]:


np.random.rand(5,5) # Produces the Random numbers on the Rows and Columns


# In[36]:


np.random.randint(1,10) # random.randint helps in explaining the random numner interger from the range that we have given


# In[38]:


np.random.randint(1, 1000,(10, 10)) # It Means we need to divide the elements in 10 X 10 rows and columns between the numbers 1 to 1000


# In[45]:


X = np.random.randint(1,100,50)


# In[46]:


X


# In[49]:


X.reshape(10,5) # The Reshape must be performned in the manner such that the required shape is a multiple of the total number of elements


# In[51]:


X.max()


# In[53]:


X.min()


# In[55]:


X.argmax()


# In[56]:


X.mean()


# In[57]:


X


# In[58]:


X[2]


# In[60]:


X.argmax()


# In[61]:


X[20]


# In[62]:


X.max()


# In[63]:


# Slicing


# In[64]:


X


# In[70]:


Y = X[1:6] # We only consider the first element, but not last element's index i.e. numbers at index 6 must be excluded


# In[69]:


X[12:14] #Suppose we want to check the index of the number 87 and 55, we calculated the index of 87 i.e 12 and add last + 1 for ;ast element


# In[71]:


Y


# In[75]:


Y[-1]


# In[76]:


Y[:] = -1


# In[77]:


Y


# In[78]:


# For Accesing the both Rows and Columns i.e


# In[87]:


Yz = np.random.rand(1,10,5)


# In[88]:


Yz


# In[97]:


X = np.array([[1,2,3], [4,5,6]])


# In[98]:


X


# In[100]:


X[1][1]


# In[116]:


My = np.random.randn(1,5)


# In[118]:


import pandas as pd


# In[119]:


list1 = [1,2,3]


# In[120]:


ser1 = pd.Series(list1)


# In[121]:


ser1


# In[122]:


ser1[1]


# In[128]:


ser3 = pd.Series([1,2,3], index = list('abc'))


# In[124]:


ser3


# In[132]:


ser4 = pd.Series([4,5,6], index = list('aef'))


# In[133]:


ser4


# In[135]:


ser3 + ser4 # Only Add Common Elements in both the lists


# In[136]:


X = np.random.rand(5,5)


# In[137]:


X


# In[146]:


X1 = pd.DataFrame(X, index=list('abcde'), columns=list('mnbvc'))


# In[151]:


X1


# In[152]:


X1[2:3]


# In[154]:


X1['m']


# In[155]:


X1.m


# In[156]:


X1


# In[157]:


X1[['m','n']]


# In[159]:


X1['Sum of m and n'] = X1['m'] + X1['n']


# In[160]:


X1


# In[163]:


X2 = X1.drop('Sum of m and n', axis=1)


# In[164]:


X2


# In[166]:


X1.drop('Sum of m and n', axis=1, inplace= True)


# In[167]:


X1


# In[168]:


X1.drop('c', axis=0)


# In[171]:


X1.drop('c', axis=0, inplace= True)


# In[172]:


X1


# In[173]:


X1


# In[186]:


X1.loc['a': 'b'] # loc is only available for the Rows i.e. a,b,c,d


# In[185]:


X1.iloc[1]


# In[188]:


X1.loc['a', 'b']


# In[190]:


X1.reset_index(inplace=True)


# In[191]:


X1


# In[192]:


Dict1 = { 'A': [1, 2, 3.5], 
         'B': [5, 6, 7], 
         'c': [9, 5, 'nan']}


# In[193]:


Dict1


# In[197]:


Data = pd.DataFrame(Dict1)


# In[198]:


Data


# In[240]:


Data.dropna()


# In[ ]:





# In[201]:


Data.dropna(axis=1)


# In[ ]:





# In[207]:


Data['c'].fillna(value=5, inplace=True)


# In[208]:


Data


# In[216]:


Data.fillna(5)


# In[217]:


Data


# In[221]:


Data.dropna()


# In[222]:


Data = Data.dropna()


# In[224]:


Data.reset_index


# In[225]:


Data


# In[233]:


Data.dropna()


# In[234]:




dict1 = {'A':[1,2,np.nan], 'B':[12, np.nan, np.nan], 'c':[3,5,10]}


# In[235]:


dict1


# In[236]:


df2 = pd.DataFrame(dict1)


# In[237]:


df2


# In[238]:


df2.dropna()


# In[241]:


df2


# In[242]:


df2.dropna(axis=1)


# In[243]:


df2


# In[244]:


df2.dropna(thresh=2)


# In[245]:


df2.fillna(value='as')


# In[246]:


df2['A'].fillna(df2['A'].mean(), inplace= True)


# In[247]:


df2


# In[250]:


df2.fillna(value=df2['B'].mean(), inplace=True)


# In[251]:


df2


# In[261]:


A = {'A': [1, 2, 5, 7],
     'B': [8, np.nan, 10, 11], 
     'C': [ 25, 10, np.nan, np.nan]}


# In[262]:


df = pd.DataFrame(A)


# In[263]:


df


# In[265]:


Dict = {'Name': ['Sam', 'Smith', np.nan, 'Justin'], 
        'Sub': ['Maths', np.nan, 'Bio', 'Chem'], 
        'Score': [78, np.nan, 45, np.nan]}


# In[266]:


Dict


# In[267]:


DD = pd.DataFrame(Dict)


# In[268]:


DD


# In[269]:


DD.dropna()


# In[270]:


DD


# In[277]:


DD['Score'].fillna(value=DD['Score'].mean(), inplace= True)


# In[278]:


DD


# In[280]:


'How are you doing'.split()


# In[281]:


Worldcup = {'Country': 'India Australia India Australia Pakistan Pakistan WI SA WI SA'.split(), 
            'Year': [2008, 2008, 2006, 2006, 2008, 2006, 2008, 2008, 2006, 2006], 
            'Rank': [1, 2, 2, 1, 3,3, 4,5,5,4]}


# In[282]:


Worldcup


# In[283]:


df = pd.DataFrame(Worldcup)


# In[284]:


df


# In[286]:


df[(df.Country == 'India')]


# In[290]:


df[df['Country'] == 'India']


# In[291]:


bycountry = df.groupby('Country')


# In[294]:


bycountry.groups


# In[295]:


bycountry.mean()


# In[ ]:





# In[298]:


byyear.groups


# In[301]:


df = pd.DataFrame({'C1': [1, 2, 3, 4], 'C2': [21, 43, 21, 43], 'C3': 'abc def ghi jkl'.split()})


# In[302]:


df


# In[305]:


df['C1'].unique()


# In[ ]:




