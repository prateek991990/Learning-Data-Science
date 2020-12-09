#!/usr/bin/env python
# coding: utf-8

# In[3]:


UserSalary = int(input('Enter Your Monthly Salary'))
AnnualSalary = (UserSalary * 12)
if AnnualSalary < 250000:
    print('You Do Not Need to Pay Text')
   
if AnnualSalary < 500000:
    Tax = ((AnnualSalary - 250000) * 10)/100 
    print('Your Tax Calculated for this year is {}'.format(Tax))
if AnnualSalary > 500000 and AnnualSalary < 800000:
    Tax = ((AnnualSalary - 250000) * 10)/100 + ((AnnualSalary - 800000) * 20)/100
    print('Your Tax Calculated for this year is {}'.format(Tax))
if AnnualSalary > 800000:
    Tax = ((AnnualSalary - 250000) * 10)/100 + ((AnnualSalary - 800000) * 30)/100
    print('Your Tax Calculated for this year is {}'.format(Tax))
if Tax < 100000:
    print('You Are Doing Great')
if Tax > 100000 and Tax < 200000:
    print('You Should Try Some of The Investments')
if Tax > 200000:
    print('Please be in Touch with a Finacial Advisor')


# In[ ]:





# In[ ]:




