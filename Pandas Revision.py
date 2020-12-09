#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import datetime


# In[10]:


Base = datetime.datetime.today()


# In[13]:


Base


# In[19]:


Date_List = [Base - datetime.timedelta(days = x) for x in range(0, 365)]


# In[20]:


Date_List[0:5]


# In[21]:


Score_List = list(np.random.randint(low = 1, high = 1000,size =365))


# In[24]:


Score_List


# In[25]:


df= pd.DataFrame()


# In[26]:


df


# In[32]:


import pandas as pd


# In[34]:


import numpy as np


# In[35]:


df1 = pd.DataFrame({'Key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})


# In[40]:


df2 = pd.DataFrame({'Key': ['a', 'b', 'd'], 'data2': range(3)})


# In[41]:


print(df1)
print(df2)


# In[42]:


pd.merge(df1, df2)


# In[44]:


pd.merge(df1, df2, on = 'Key')


# In[45]:


df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'Data1': range(7)})
df4 = pd.DataFrame({'Rkey': ['a', 'b', 'd'], 'Data2': range(3)})


# In[46]:


df4


# In[47]:


df3


# In[51]:


pd.merge(df3, df4, left_on= 'lkey', right_on='Rkey')


# In[53]:


pd.merge(df1, df2, how = 'inner')


# In[54]:


pd.merge(df1, df2, how = 'outer')


# In[55]:


left = pd.DataFrame({'Key1': ['Foo', 'Foo', 'Bar'], 'Key2': ['One', 'Two', 'One'], 'lval': [1,2,3]})


# In[56]:


left


# In[58]:


right = pd.DataFrame({'Key1': ['Foo', 'Foo', 'Bar', 'Bar'], 'Key2': ['one', 'one', 'one', 'Two'], 'Rval': [4, 5, 6, 7]})


# In[59]:


right


# In[63]:


pd.merge(left, right, on = ['Key1', 'Key2'], how = 'outer')


# In[64]:


pd.merge(left, right, on = 'Key1')


# In[65]:


#83


# In[68]:


Data = pd.DataFrame({'Key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'Data1': range(7)})
Data2 = pd.DataFrame({'Key2': ['a', 'b', 'd'], 'Data2': range(3)})


# In[70]:


print(Data), print(Data2)


# In[71]:


#performing the Merge


# In[72]:


pd.merge(Data, Data2)


# In[80]:


My = pd.DataFrame({'lKey': ['b', 'b', 'a', 'c', 'a','a', 'b'], 'Data1': range(7)})
My1 = pd.DataFrame({'Rkey': ['a', 'b', 'd'], 'Data2': range(3)})


# In[81]:


print(My), print(My1)


# In[83]:


pd.merge(My, My1, left_on='lKey', right_on='Rkey')


# In[85]:


Data = pd.DataFrame(np.arange(6).reshape((2,3)), 
index = pd.Index(['India', 'Europe'], Name = 'Ctry'), 
Columns = pd.Index(['One', 'Two', 'Three'], Name = 'Number'))


# In[88]:


Data = pd.DataFrame(np.arange(6).reshape((2,3)), index = pd.Index(['India', 'Europe'], name = 'Ctry'), 
                    columns = pd.Index(['one', 'two', 'three'], name = 'number'))


# In[89]:


Data


# In[92]:


Data = pd.DataFrame(np.arange(6).reshape((2,3)), 
       index = pd.Index(['India', 'Europe'], name = 'Ctry'),
       columns = pd.Index(['one', 'Two', 'Three'], name = 'number'))


# In[93]:


Data


# In[94]:


Results = Data.stack()


# In[97]:


Results.unstack()


# In[96]:


Data.unstack()


# In[98]:


import matplotlib.pyplot as plt
import matplotlib as mpl


# In[99]:


randvals = np.random.rand(1000)


# In[102]:


pd.Series(randvals).plot(title = 'Random White Noise', color = 'r')


# In[101]:


plt.show()


# In[103]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# In[ ]:




