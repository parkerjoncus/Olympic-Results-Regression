#!/usr/bin/env python
# coding: utf-8

# # MATH 484 Project
# We are trying to predict the future times and results of sprints and field events at the Olympics.

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np
import statsmodels.api as sm
import statsmodels.nonparametric as snp
import statsmodels.stats as sms
import scipy.linalg as lg
import plotly

track_results = pd.read_csv('results.csv')


# In[2]:


# Creating separate dictionaries for each event.
tf_results = dict(tuple(track_results.groupby("Event")))


# I am going to create dataframes and plots of the sprint events from the Summer Olympics that we have results from.

# In[3]:


print(track_results.Event.unique())


# We will create results scatter plots for field events as well overtime at the Olympics.

# # Simple Linear Regression Models for 100 Meter, 200 Meter, Long Jump, Shot Put, 1500 M Run
# 
# We start by running a simple linear regression for the 100 M Men and Women's Events.

# In[4]:


Men_100M = tf_results['100M Men']
Men_100M.head()


# In[5]:


X = tf_results['100M Men'].Year.astype(float)
y = tf_results['100M Men'].Result.astype(float)
X = sm.add_constant(X)
model_100M_Men = sm.OLS(y, X).fit()
print(model_100M_Men.params)
model_100M_Men.summary()


# In[6]:


X = tf_results['100M Women'].Year.astype(float)
y = tf_results['100M Women'].Result.astype(float)
X = sm.add_constant(X)
model_100M_Women = sm.OLS(y, X).fit()
print(model_100M_Women.params)
model_100M_Women.summary()


# 200 M Men's and Women's Events now:

# In[7]:


X = tf_results['200M Men'].Year.astype(float)
y = tf_results['200M Men'].Result.astype(float)
X = sm.add_constant(X)
model_200M_Men = sm.OLS(y, X).fit()
print(model_200M_Men.params)
model_200M_Men.summary()


# In[8]:


X = tf_results['200M Women'].Year.astype(float)
y = tf_results['200M Women'].Result.astype(float)
X = sm.add_constant(X)
model_200M_Women = sm.OLS(y, X).fit()
print(model_200M_Women.params)
model_200M_Women.summary()


# In[9]:


X = tf_results['Long Jump Men'].Year.astype(float)
y = tf_results['Long Jump Men'].Result.astype(float)
X = sm.add_constant(X)
LJ_Men = sm.OLS(y, X).fit()
print(LJ_Men.params)
LJ_Men.summary()


# In[10]:


X = tf_results['Long Jump Women'].Year.astype(float)
y = tf_results['Long Jump Women'].Result.astype(float)
X = sm.add_constant(X)
LJ_Women = sm.OLS(y, X).fit()
print(LJ_Women.params)
LJ_Women.summary()


# In[11]:


X = tf_results['Shot Put Men'].Year.astype(float)
y = tf_results['Shot Put Men'].Result.astype(float)
X = sm.add_constant(X)
SP_Men = sm.OLS(y, X).fit()
print(SP_Men.params)
SP_Men.summary()


# In[12]:


X = tf_results['Shot Put Women'].Year.astype(float)
y = tf_results['Shot Put Women'].Result.astype(float)
X = sm.add_constant(X)
SP_Women = sm.OLS(y, X).fit()
print(SP_Women.params)
SP_Women.summary()


# In[13]:


#Men's 1500M
men_1500m = pd.read_csv('Men 1500M.csv')
X = men_1500m['Year'].astype(float)
y = men_1500m['seconds'].astype(float)
X = sm.add_constant(X)
Men_1500M = sm.OLS(y, X).fit()
print(Men_1500M.params)
Men_1500M.summary()


# In[14]:


#Women's 1500M
women_1500m = pd.read_csv('Women 1500M.csv')
women_1500m.head(3)
X = women_1500m['Year'].astype(float)
y = women_1500m['seconds'].astype(float)
X = sm.add_constant(X)
Women_1500M = sm.OLS(y, X).fit()
print(Women_1500M.params)
Women_1500M.summary()


# In[15]:


#Regression Subplots
plt.figure(figsize=(16, 12))
plt.subplot(321)
time_m = np.linspace(1894, 2018, num = 2000)
time_w = np.linspace(1928, 2018, num = 2000)
Men_pred = 37.670494 - 0.013918 * time_m
Women_pred =  39.259190 - 0.014167 * time_w
plt.scatter(tf_results['100M Men'].Year, tf_results['100M Men'].Result, color = 'k')
plt.plot(time_m, Men_pred)
plt.scatter(tf_results['100M Women'].Year, tf_results['100M Women'].Result, color = 'm')
plt.plot(time_w, Women_pred)
plt.xlabel('Year')
plt.ylabel('Result in Seconds')
plt.title('Podium Finishes at the Olympics-100 M')
plt.legend(['100M Men Prediction', '100M Women Prediction', '100M Men Results', '100M Women Results'])


plt.subplot(322)
time_m = np.linspace(1898, 2018, num = 2000)
time_w = np.linspace(1946, 2018, num = 2000)
Men_pred = 67.599404 -0.023895 * time_m
Women_pred =  94.100684 -0.036020  * time_w
plt.scatter(tf_results['200M Men'].Year, tf_results['200M Men'].Result, color = 'k')
plt.plot(time_m, Men_pred)
plt.scatter(tf_results['200M Women'].Year, tf_results['200M Women'].Result, color = 'm')
plt.plot(time_w, Women_pred)
plt.xlabel('Year')
plt.ylabel('Result in Seconds')
plt.title('Podium Finishes at the Olympics-200 M')
plt.legend(['200M Men Prediction', '200M Women Prediction', '200M Men Results', '200M Women Results'])

plt.subplot(323)
time_m = np.linspace(1894, 2018, num = 2000)
time_w = np.linspace(1950, 2018, num = 2000)
Men_pred = -18.587442 +0.013485 * time_m
Women_pred = -21.278026 +0.014137  * time_w
plt.scatter(tf_results['Long Jump Men'].Year, tf_results['Long Jump Men'].Result, color = 'b')
plt.plot(time_m, Men_pred)
plt.scatter(tf_results['Long Jump Women'].Year, tf_results['Long Jump Women'].Result, color = 'g')
plt.plot(time_w, Women_pred)
plt.xlabel('Year')
plt.ylabel('Result in Meters')
plt.title('Podium Finishes at the Olympics-Long Jump')
plt.legend(['Long Jump Men Prediction', 'Long Jump Women Prediction', 'Long Jump Men Results', 'Long Jump Women Results'])


plt.subplot(324)
time_m = np.linspace(1894, 2018, num = 2000)
time_w = np.linspace(1950, 2018, num = 2000)
Men_pred = -142.645216 +0.082039 * time_m
Women_pred = -103.642325 +0.062040  * time_w
plt.scatter(tf_results['Shot Put Men'].Year, tf_results['Shot Put Men'].Result, color = 'b')
plt.plot(time_m, Men_pred)
plt.scatter(tf_results['Shot Put Women'].Year, tf_results['Shot Put Women'].Result, color = 'g')
plt.plot(time_w, Women_pred)
plt.xlabel('Year')
plt.ylabel('Result in Meters')
plt.title('Podium Finishes at the Olympics-Shot Put')
plt.legend(['Shot Put Men Prediction', 'Shot Put Women Prediction', 'Shot Put Men Results', 'Shot Put Women Results'])

plt.subplot(325)
time_m = np.linspace(1898, 2018, num = 2000)
time_w = np.linspace(1970, 2018, num = 2000)
Men_pred = 846.904980 -0.316465 * time_m
Women_pred = 40.274876 +0.101555  * time_w
plt.scatter(men_1500m['Year'], men_1500m['seconds'], color = 'c')
plt.plot(time_m, Men_pred)
plt.scatter(women_1500m['Year'], women_1500m['seconds'], color = 'y')
plt.plot(time_w, Women_pred)
plt.xlabel('Year')
plt.ylabel('Result in Seconds')
plt.title('Podium Finishes at the Olympics-1500M')
plt.legend(['1500M Men Prediction', '1500M Women Prediction', '1500M Men Results', '1500M Women Results'])


plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
plt.savefig('slr_results.png')
plt.show()


# # Diagonostics of the Simple Linear Models
# 
# We now will look at the residuals to try to determine if we can make the assumption that the error is normally distributed with mean 0 and variance $\sigma^2$, and run some diagnostic tests to determine whether the error is constant.

# In[16]:


#Residuals for 100M
res_100_m = tf_results['100M Men'].Result.astype(float) - (37.670494 - 0.013918 * tf_results['100M Men'].Year.astype(float))
res_100_w =  tf_results['100M Women'].Result.astype(float) - (39.259190 - 0.014167 * tf_results['100M Women'].Year.astype(float))

#Residuals for 200M
res_200_m = tf_results['200M Men'].Result.astype(float) - (67.599404 -0.023895 * tf_results['200M Men'].Year.astype(float))
res_200_w = tf_results['200M Women'].Result.astype(float) - (94.100684 -0.036020  * tf_results['200M Women'].Year.astype(float))

#Residuals for Long Jump
res_lj_m = tf_results['Long Jump Men'].Result.astype(float) - (-18.587442 +0.013485 * tf_results['Long Jump Men'].Year.astype(float))
res_lj_w = tf_results['Long Jump Women'].Result.astype(float) - (-21.278026 +0.014137  * tf_results['Long Jump Women'].Year.astype(float))

#Residuals for Shot Put
res_sp_m = tf_results['Shot Put Men'].Result.astype(float) - (-142.645216 +0.082039 * tf_results['Shot Put Men'].Year.astype(float))
res_sp_w = tf_results['Shot Put Women'].Result.astype(float) - (-103.642325 +0.062040  * tf_results['Shot Put Women'].Year.astype(float))

#Residuals for 1500M
res_1500_m = men_1500m['seconds'] - (846.904980 -0.316465 * men_1500m['Year'])
res_1500_w = women_1500m['seconds'] - (40.274876 +0.101555 * women_1500m['Year'])


# In[17]:


#Residual Plots for Simple Linear Models
plt.figure(figsize = (16, 12))
plt.subplot(321)
plt.scatter(tf_results['100M Men'].Year, res_100_m, color = 'k')
plt.scatter(tf_results['100M Women'].Year, res_100_w, color = 'm')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for 100M')
plt.legend(['100M Men Residuals', '100M Women Residuals'])

plt.subplot(322)
plt.scatter(tf_results['200M Men'].Year, res_200_m, color = 'k')
plt.scatter(tf_results['200M Women'].Year, res_200_w, color = 'm')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for 200M')
plt.legend(['200M Men Residuals', '200M Women Residuals'])

plt.subplot(323)
plt.scatter(men_1500m['Year'], res_1500_m, color = 'c')
plt.scatter(women_1500m['Year'], res_1500_w, color = 'y')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for 1500M')
plt.legend(['1500M Men Residuals', '1500M Women Residuals'])

plt.subplot(324)
plt.scatter(tf_results['Shot Put Men'].Year, res_sp_m, color = 'b')
plt.scatter(tf_results['Shot Put Women'].Year, res_sp_w, color = 'g')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for Shot Put')
plt.legend(['Shot Put Men Residuals', 'Shot Put Women Residuals'])

plt.subplot(325)
plt.scatter(tf_results['Long Jump Men'].Year, res_lj_m, color = 'b')
plt.scatter(tf_results['Long Jump Women'].Year, res_lj_w, color = 'g')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for Long Jump')
plt.legend(['Long Jump Men Residuals', 'Long Jump Women Residuals'])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
plt.savefig('slr_residual_results.png')
plt.show()


# We are somewhat suspicious of whether or not the error variance is constant or normally distributed. We also believe that we may need higher order terms or a Box-Cox transformation to fit our regression model better than it currently fits the data.
# 
# # Normal Probability Plots

# In[18]:


#Normal Probability Plots
plt.figure(figsize=(20, 10))
plt.subplot(251)
#100M Men
res_prob_100m_m = stats.probplot(res_100_m, plot= plt)
plt.title('100M Men Probability Plot')

plt.subplot(252)
#100M Women
res_prob_100m_w = stats.probplot(res_100_w, plot= plt)
plt.title('100M Women Probability Plot')

plt.subplot(253)
#200M Men
res_prob_200m_m = stats.probplot(res_200_m, plot= plt)
plt.title('200M Men Probability Plot')

plt.subplot(254)
#200M Women
res_prob_200m_w = stats.probplot(res_200_w, plot= plt)
plt.title('200M Women Probability Plot')

plt.subplot(255)
#1500M Men
res_prob_1500m_m = stats.probplot(res_1500_m, plot= plt)
plt.title('1500M Men Probability Plot')

plt.subplot(256)
#1500M Women
res_prob_1500m_w = stats.probplot(res_1500_w, plot= plt)
plt.title('1500M Women Probability Plot')

plt.subplot(257)
#Men's Shot Put
res_prob_sp_m = stats.probplot(res_sp_m, plot= plt)
plt.title('Shot Put Men Probability Plot')

plt.subplot(258)
#Women's Shot Put
res_prob_sp_w = stats.probplot(res_sp_w, plot= plt)
plt.title('Shot Put Women Probability Plot')

plt.subplot(259)
#Men's Long Jump
res_prob_lj_m = stats.probplot(res_lj_m, plot= plt)
plt.title('Long Jump Men Probability Plot')

plt.subplot(2,5,10)
#Women's Long Jump
res_prob_lj_w = stats.probplot(res_lj_w, plot= plt)
plt.title('Long Jump Women Probability Plot')
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
plt.savefig('slr_normprob_results.png')
plt.show()


# It appears that there are some departures in normality in every single normal probability plot in varying degrees of seriousness except for Shot Put, Women's Long Jump, and Men's 200M dash. This again strongly implies that the error variance is not constant for this model and that we may need to utilize some higher order terms to better fit the data.
# 
# # Bruesch-Pagan Tests for Constant Error Variance
# 
# We will also run a Bruesch-Pagan test to test form departures in constancy of the error terms. Remember, this test is carried out as follows:
# 
# We regress the function: $log(\sigma^2_i) = \gamma_0 + \gamma_1 x_{i1}$
# 
# $H_0: \gamma_1 = 0$
# 
# $H_1: \gamma_1 \neq 0$
# 
# Test statistic: $\frac{SSR^* / 2}{(SSE / n)^2}$
# 
# Where $SSR^*$ is the regression sum of squares for the log regression of the residuals.
# 
# Critical value for the rejection region would be $\chi^2_{\alpha/2; 1}$, and we will test at an $\alpha$ = 0.05 level.

# In[19]:


from statsmodels.stats import diagnostic as dn
bp_100_m = dn.het_breuschpagan(model_100M_Men.resid, model_100M_Men.model.exog)
print("The Breusch-Pagan test yields a p-value of: ", bp_100_m[3],".")


# In[20]:


bp_100_w = dn.het_breuschpagan(model_100M_Women.resid, model_100M_Women.model.exog)
print("The Breusch-Pagan test yields a p-value of: ", bp_100_w[3],".")


# In[21]:


bp_200_m = dn.het_breuschpagan(model_200M_Men.resid, model_200M_Men.model.exog)
print("The Breusch-Pagan test yields a p-value of: ", bp_200_m[3],".")


# In[22]:


bp_200_w = dn.het_breuschpagan(model_200M_Women.resid, model_200M_Women.model.exog)
print("The Breusch-Pagan test yields a p-value of: ", bp_200_w[3],".")


# In[23]:


bp_1500_m = dn.het_breuschpagan(Men_1500M.resid, Men_1500M.model.exog)
print("The Breusch-Pagan test yields a p-value of: ", bp_1500_m[3],".")


# In[24]:


bp_1500_w = dn.het_breuschpagan(Women_1500M.resid, Women_1500M.model.exog)
print("The Breusch-Pagan test yields a p-value of: ", bp_1500_w[3],".")


# In[25]:


bp_lj_m = dn.het_breuschpagan(LJ_Men.resid, LJ_Men.model.exog)
print("The Breusch-Pagan test yields a p-value of: ", bp_lj_m[3],".")


# In[26]:


bp_lj_w = dn.het_breuschpagan(LJ_Women.resid, LJ_Women.model.exog)
print("The Breusch-Pagan test yields a p-value of: ", bp_lj_w[3],".")


# In[27]:


bp_sp_m = dn.het_breuschpagan(SP_Men.resid, SP_Men.model.exog)
print("The Breusch-Pagan test yields a p-value of: ", bp_sp_m[3],".")


# In[28]:


bp_sp_w = dn.het_breuschpagan(SP_Women.resid, SP_Women.model.exog)
print("The Breusch-Pagan test yields a p-value of: ", bp_sp_w[3],".")


# Based on the Bruesch-Pagan tests we have run, it appears that the only track results that we have regressed that do not have constant error variance are the 100M Results, the Women's 200M, the Men's Long Jump, and the Women's Shot Put at an significance level of $\alpha = 0.05$.
# 
# # Goodness of Fit Tests on First Order Model:
# 
# Since we have repeat observations at each independent predictor variable, we should be able to run an F test for lack of fit. Remember, the F-test for lack of fit is carried out as follows:
# 
# $H_0: E(Y) = \beta_0 + \beta_1 X$
# 
# $H_1: E(Y) \neq \beta_0 + \beta_1 X$
# 
# The test statistic here would be $F^* = \frac{SSLF}{c-2} / \frac{SSPE}{n-c} = \frac{MSLF}{MSPE}$, where c is the number of years with replicates and $n = \sum{j=1}^{c} n_j$ and n is the number of observations.
# 
# Our decision rule is:
# 
# $F^* \leq F(1 - \alpha; c - 2, n - c)$ then we conclude that our simple linear model is appropriate.
# 
# $F^* > F(1 - \alpha; c - 2, n -c)$ then we conclude that our simple linear model is not appropriate for our data.
# 
# We will control the error at $\alpha$ = 0.05. 
# 
# # ALL THE CODE WILL BE IN R FOR THESE TESTS.
# 
# All of the results are on Github. In all cases we conclude that the simple linear models are not appropriate. Clearly, we should use some Box-Cox transformations on our results in order to better predict the results.
# 
# # Multiple Linear Regression - Standardized

# In[29]:


from scipy.stats.mstats import zscore
Mens_100M = pd.read_csv('X100M_Men.csv')
Mens_100M.head()


# In[30]:


X = Mens_100M[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Mens_100M['Result'].astype(float)
#X = sm.add_constant(X)
big_100M_Men = sm.OLS(zscore(y), zscore(X)).fit()
print(big_100M_Men.summary())


# In[31]:


Womens_100M = pd.read_csv('100M_Women.csv')
Womens_100M.head()


# In[32]:


X = Womens_100M[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Womens_100M['Result'].astype(float)
#X = sm.add_constant(X)
big_100M_Women = sm.OLS(zscore(y), zscore(X)).fit()
print(big_100M_Women.params)
print(big_100M_Women.summary())


# In[33]:


Mens_200M = pd.read_csv('200M_Men.csv')
X = Mens_200M[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Mens_200M['Result'].astype(float)
#X = sm.add_constant(X)
big_200M_Men = sm.OLS(zscore(y), zscore(X)).fit()
print(big_200M_Men.params)
print(big_200M_Men.summary())


# In[34]:


Womens_200M = pd.read_csv('200M_Women.csv')
X = Womens_200M[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Womens_200M['Result'].astype(float)
#X = sm.add_constant(X)
big_200M_Women = sm.OLS(zscore(y), zscore(X)).fit()
print(big_200M_Women.params)
print(big_200M_Women.summary())


# In[35]:


Mens_1500M = pd.read_csv('1500M_Men.csv')
X = Mens_1500M[['Year', 'age', 'weight', 'height']].astype(float)
y = Mens_1500M['seconds'].astype(float)
#X = sm.add_constant(X)
big_1500M_Men = sm.OLS(zscore(y), zscore(X)).fit()
print(big_1500M_Men.summary())


# In[36]:


Womens_1500M = pd.read_csv('1500M_Women.csv')
X = Womens_1500M[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Womens_1500M['seconds'].astype(float)
#X = sm.add_constant(X)
big_1500M_Women = sm.OLS(zscore(y), zscore(X)).fit()
print(big_1500M_Women.params)
print(big_1500M_Women.summary())


# In[37]:


Mens_LJ = pd.read_csv('LJ_Men.csv')
X = Mens_LJ[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Mens_LJ['Result'].astype(float)
#X = sm.add_constant(X)
big_Mens_LJ= sm.OLS(zscore(y), zscore(X)).fit()
print(big_Mens_LJ.params)
print(big_Mens_LJ.summary())


# In[38]:


Mens_SP = pd.read_csv('SP_Men.csv')
X = Mens_SP[['Age', 'Weight', 'Height']].astype(float)
y = Mens_SP['Result'].astype(float)
#X = sm.add_constant(X)
big_Mens_SP= sm.OLS(zscore(y), zscore(X)).fit()
print(big_Mens_SP.params)
print(big_Mens_SP.summary())


# In[39]:


res_prob_sp_w = stats.probplot(big_Mens_SP.resid, plot= plt)
plt.title('Mens Shot Put Probability Plot')
plt.show()


# In[40]:


Mens_SP = pd.read_csv('SP_Men.csv')
X = Mens_SP[['Year','Age', 'Weight', 'Height']].astype(float)
y = Mens_SP['Result'].astype(float)
#X = sm.add_constant(X)
big_Mens_SP= sm.OLS(zscore(y), zscore(X)).fit()
print(big_Mens_SP.params)
print(big_Mens_SP.summary())


# In[41]:


Womens_LJ = pd.read_csv('LJ_Women.csv')
X = Womens_LJ[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Womens_LJ['Result'].astype(float)
#X = sm.add_constant(X)
big_Womens_LJ= sm.OLS(zscore(y), zscore(X)).fit()
print(big_Womens_LJ.summary())


# In[42]:


Womens_SP = pd.read_csv('SP_Women.csv')
X = Womens_SP[['Year','Age', 'Weight', 'Height']].astype(float)
y = Womens_SP['Result'].astype(float)
#X = sm.add_constant(X)
big_Womens_SP= sm.OLS(zscore(y), zscore(X)).fit()
print(big_Womens_SP.params)
print(big_Womens_SP.summary())


# In[43]:


#Standardized Normal Probability Plots for Multiple Linear Regression Models
plt.figure(figsize=(20, 10))
plt.subplot(251)
#100M Men
res_prob_100m_m = stats.probplot(big_100M_Men.resid, plot= plt)
plt.title('100M Men Probability Plot')

plt.subplot(252)
#100M Women
res_prob_100m_w = stats.probplot(big_100M_Women.resid, plot= plt)
plt.title('100M Women Probability Plot')

plt.subplot(253)
#200M Men
res_prob_200m_m = stats.probplot(big_200M_Men.resid, plot= plt)
plt.title('200M Men Probability Plot')

plt.subplot(254)
#200M Women
res_prob_200m_w = stats.probplot(big_200M_Women.resid, plot= plt)
plt.title('200M Women Probability Plot')

plt.subplot(255)
#1500M Men
res_prob_1500m_m = stats.probplot(big_1500M_Men.resid, plot= plt)
plt.title('1500M Men Probability Plot')

plt.subplot(256)
#1500M Women
res_prob_1500m_w = stats.probplot(big_1500M_Women.resid, plot= plt)
plt.title('1500M Women Probability Plot')

plt.subplot(257)
#Men's Shot Put
res_prob_sp_m = stats.probplot(big_Mens_SP.resid, plot= plt)
plt.title('Shot Put Men Probability Plot')

plt.subplot(258)
#Women's Shot Put
res_prob_sp_w = stats.probplot(big_Womens_SP.resid, plot= plt)
plt.title('Shot Put Women Probability Plot')

plt.subplot(259)
#Men's Long Jump
res_prob_lj_m = stats.probplot(big_Mens_LJ.resid, plot= plt)
plt.title('Long Jump Men Probability Plot')

plt.subplot(2,5,10)
#Women's Long Jump
res_prob_lj_w = stats.probplot(big_Womens_LJ.resid, plot= plt)
plt.title('Long Jump Women Probability Plot')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
plt.savefig('std_mlr_normprob_results.png')
plt.show()


# In[44]:


#Standardized Residual Plots for Simple Linear Models
plt.figure(figsize = (16, 12))
plt.subplot(321)
plt.scatter(Mens_100M['Year'], big_100M_Men.resid, color = 'k')
plt.scatter(Womens_100M['Year'], big_100M_Women.resid, color = 'm')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for 100M')
plt.legend(['100M Men Residuals', '100M Women Residuals'])

plt.subplot(322)
plt.scatter(Mens_200M.Year, big_200M_Men.resid, color = 'k')
plt.scatter(Womens_200M.Year, big_200M_Women.resid, color = 'm')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for 200M')
plt.legend(['200M Men Residuals', '200M Women Residuals'])

plt.subplot(323)
plt.scatter(Mens_1500M.Year, big_1500M_Men.resid, color = 'c')
plt.scatter(Womens_1500M.Year, big_1500M_Women.resid, color = 'y')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for 1500M')
plt.legend(['1500M Men Residuals', '1500M Women Residuals'])

plt.subplot(324)
plt.scatter(Mens_SP.Year, big_Mens_SP.resid, color = 'b')
plt.scatter(Womens_SP.Year, big_Womens_SP.resid, color = 'g')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for Shot Put')
plt.legend(['Shot Put Men Residuals', 'Shot Put Women Residuals'])

plt.subplot(325)
plt.scatter(Mens_LJ.Year, big_Mens_LJ.resid, color = 'b')
plt.scatter(Womens_LJ.Year, big_Womens_LJ.resid, color = 'g')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for Long Jump')
plt.legend(['Long Jump Men Residuals', 'Long Jump Women Residuals'])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
plt.savefig('std_mlr_residual_results.png')
plt.show()


# # Multiple Linear Regression - Unstandardized

# In[45]:


#Unstandardized Full First Order Models
X = Mens_100M[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Mens_100M['Result'].astype(float)
X = sm.add_constant(X)
big_100M_Men = sm.OLS(y, X).fit()
print(big_100M_Men.summary())

X = Womens_100M[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Womens_100M['Result'].astype(float)
X = sm.add_constant(X)
big_100M_Women = sm.OLS(y, X).fit()
print(big_100M_Women.summary())

Mens_200M = pd.read_csv('200M_Men.csv')
X = Mens_200M[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Mens_200M['Result'].astype(float)
X = sm.add_constant(X)
big_200M_Men = sm.OLS(y, X).fit()
print(big_200M_Men.summary())

Womens_200M = pd.read_csv('200M_Women.csv')
X = Womens_200M[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Womens_200M['Result'].astype(float)
X = sm.add_constant(X)
big_200M_Women = sm.OLS(y, X).fit()
print(big_200M_Women.summary())

Mens_1500M = pd.read_csv('1500M_Men.csv')
X = Mens_1500M[['Year', 'age', 'weight', 'height']].astype(float)
y = Mens_1500M['seconds'].astype(float)
X = sm.add_constant(X)
big_1500M_Men = sm.OLS(y, X).fit()
print(big_1500M_Men.summary())

Womens_1500M = pd.read_csv('1500M_Women.csv')
X = Womens_1500M[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Womens_1500M['seconds'].astype(float)
X = sm.add_constant(X)
big_1500M_Women = sm.OLS(y, X).fit()
print(big_1500M_Women.summary())

Mens_LJ = pd.read_csv('LJ_Men.csv')
X = Mens_LJ[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Mens_LJ['Result'].astype(float)
X = sm.add_constant(X)
big_Mens_LJ= sm.OLS(y, X).fit()
print(big_Mens_LJ.summary())

Mens_SP = pd.read_csv('SP_Men.csv')
X = Mens_SP[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Mens_SP['Result'].astype(float)
X = sm.add_constant(X)
big_Mens_SP= sm.OLS(y, X).fit()
print(big_Mens_SP.summary())

Womens_LJ = pd.read_csv('LJ_Women.csv')
X = Womens_LJ[['Year', 'Age', 'Weight', 'Height']].astype(float)
y = Womens_LJ['Result'].astype(float)
X = sm.add_constant(X)
big_Womens_LJ= sm.OLS(y, X).fit()
print(big_Womens_LJ.summary())

Womens_SP = pd.read_csv('SP_Women.csv')
X = Womens_SP[['Year','Age', 'Weight', 'Height']].astype(float)
y = Womens_SP['Result'].astype(float)
#X = sm.add_constant(X)
big_Womens_SP= sm.OLS(y, X).fit()
print(big_Womens_SP.summary())


# In[46]:


#Unstandardized Residual Plots for Simple Linear Models
plt.figure(figsize = (16, 12))
plt.subplot(321)
plt.scatter(Mens_100M['Year'], big_100M_Men.resid, color = 'k')
plt.scatter(Womens_100M['Year'], big_100M_Women.resid, color = 'm')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for 100M')
plt.legend(['100M Men Residuals', '100M Women Residuals'])

plt.subplot(322)
plt.scatter(Mens_200M.Year, big_200M_Men.resid, color = 'k')
plt.scatter(Womens_200M.Year, big_200M_Women.resid, color = 'm')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for 200M')
plt.legend(['200M Men Residuals', '200M Women Residuals'])

plt.subplot(323)
plt.scatter(Mens_1500M.Year, big_1500M_Men.resid, color = 'c')
plt.scatter(Womens_1500M.Year, big_1500M_Women.resid, color = 'y')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for 1500M')
plt.legend(['1500M Men Residuals', '1500M Women Residuals'])

plt.subplot(324)
plt.scatter(Mens_SP.Year, big_Mens_SP.resid, color = 'b')
plt.scatter(Womens_SP.Year, big_Womens_SP.resid, color = 'g')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for Shot Put')
plt.legend(['Shot Put Men Residuals', 'Shot Put Women Residuals'])

plt.subplot(325)
plt.scatter(Mens_LJ.Year, big_Mens_LJ.resid, color = 'b')
plt.scatter(Womens_LJ.Year, big_Womens_LJ.resid, color = 'g')
plt.xlabel('Year')
plt.ylabel('Residual')
plt.title('Residuals for Long Jump')
plt.legend(['Long Jump Men Residuals', 'Long Jump Women Residuals'])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.25)
plt.savefig('unstd_mlr_residual_results.png')
plt.show()


# In[47]:


#Unstandardized Normal Probability Plots for Multiple Linear Regression Models
plt.figure(figsize=(20, 10))
plt.subplot(251)
#100M Men
res_prob_100m_m = stats.probplot(big_100M_Men.resid, plot= plt)
plt.title('100M Men Probability Plot')

plt.subplot(252)
#100M Women
res_prob_100m_w = stats.probplot(big_100M_Women.resid, plot= plt)
plt.title('100M Women Probability Plot')

plt.subplot(253)
#200M Men
res_prob_200m_m = stats.probplot(big_200M_Men.resid, plot= plt)
plt.title('200M Men Probability Plot')

plt.subplot(254)
#200M Women
res_prob_200m_w = stats.probplot(big_200M_Women.resid, plot= plt)
plt.title('200M Women Probability Plot')

plt.subplot(255)
#1500M Men
res_prob_1500m_m = stats.probplot(big_1500M_Men.resid, plot= plt)
plt.title('1500M Men Probability Plot')

plt.subplot(256)
#1500M Women
res_prob_1500m_w = stats.probplot(big_1500M_Women.resid, plot= plt)
plt.title('1500M Women Probability Plot')

plt.subplot(257)
#Men's Shot Put
res_prob_sp_m = stats.probplot(big_Mens_SP.resid, plot= plt)
plt.title('Shot Put Men Probability Plot')

plt.subplot(258)
#Women's Shot Put
res_prob_sp_w = stats.probplot(big_Womens_SP.resid, plot= plt)
plt.title('Shot Put Women Probability Plot')

plt.subplot(259)
#Men's Long Jump
res_prob_lj_m = stats.probplot(big_Mens_LJ.resid, plot= plt)
plt.title('Long Jump Men Probability Plot')

plt.subplot(2,5,10)
#Women's Long Jump
res_prob_lj_w = stats.probplot(big_Womens_LJ.resid, plot= plt)
plt.title('Long Jump Women Probability Plot')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
plt.savefig('unstd_mlr_normprob_results.png')
plt.show()


# In[ ]:




