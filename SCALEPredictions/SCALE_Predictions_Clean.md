
This notebook reads in output files from SCALE simulations to generate a large dataset of uranium fission product gamma-ray measurements with variable enrichment and decay time. The SCALE irradiations were performed in ORNL's HFIR neutron field - a highly thermal spectrum. That dataset is then fed into machine learning predictions algorithms via scikit-learn to develop prediction models of uranium enrichment. The ability of models to generalize to predictions outside of trained decay times and enrichment levels is examined. The accuracy of the models is characterized as a function of multi-variate parameters. The suitability of various learning algorithms for this task is compared. This process may be iterated with a fast spectrum neutron field.

# Setting up the notebook


```python
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
cd /Users/adam/Box Sync/Dissertation/SCALE
```

    /Users/adam/Box Sync/Dissertation/SCALE


# Importing and preprocessing data

We will start with the simplest scenario: binary classification of U-235 versus U-238. We import gamma-ray data of each isotope measured at decay times ranging from 0 days to 1 year post-irradiation. There are 180 measurements of each isotope broken down into 100 gamma-ray energy groups from 0 to 2.5 MeV.


```python
u235 = pd.read_csv('u235_100Egroups.csv').as_matrix()
u238 = pd.read_csv('u238_100Egroups.csv').as_matrix()
```

The required format for analysis is one "observation", or measurement, per ROW, not column. Therefore we must transpose the data.


```python
u235 = u235.T
u238 = u238.T
```

# Functions

Here I write a few functions which will be used later to: 

1. Create data of intermediate enrichment levels on-the-fly by manipulating the imported SCALE data of each isotope, 

2. Build, grade, and visualize the results of models constructed with varying training/testing splits and measurement replicates.

I also import a useful function for creating training/testing splits of data.

I also create a wrapper function to automate a few of these steps.

centrifuge(w, mode) generates a matrix X which has user-specified n levels of uranium enrichment. There are 180 observations, or rows, of gamma-ray data per enrichment level. It also generates the accompanying y matrix which labels each row with the correct enrichment value. 

Note: Specifying an n value of 1 will return the previously used X matrix, which consists of pure u238 and pure u235. The mode parameter tells the function whether to generate classification ('clf') data or regression ('reg') data, depending on the type of model for which the data is constructed.


```python
def centrifuge(w, mode):
    X = u238
    if mode=='clf':
        y = np.zeros([180], dtype=str)
    elif mode=='reg':
        y = np.zeros([180], dtype=float)
    else:
        print('Please specify classification (clf) or regression (reg).')
    for i in range(1,w+1):
        enrichment = (float(i)/float(w))
        enriched_u = enrichment * u235 + (1 - enrichment) * u238
        X = np.append(X, enriched_u, axis=0)
        if mode=='clf':
            category = str(enrichment)
        elif mode=='reg':
            category = float(enrichment)
        else:
            print('Please specify classification (clf) or regression (reg).')
        for j in range(0, 180):
            y = np.append(y, category)
    return X, y
        
```

grader(results,y_test) checks the predictions of a model and compares it to the known answers, and provides a score of the model accuracy as a percentage.


```python
def grader(results,y_test):
    if len(results)==len(y_test):
        pass
    else:
        print "Mismatch in predicted and testing data!"
    score=0
    for i in range(0,len(y_test)-1):
        if results[i]==y_test[i]:
            score+=1
        else:
            pass
    grade = 100 * score/len(results)
    # print('The score of the algorithm is')
    # print(grade)
    return grade
```

The scikit-learn function model_selection.train_test_split can create training/testing splits of data and labels with user specified splitting factors.


```python
from sklearn import model_selection
```

multi_grader_LinearSVC(n,m) serves as a wrapper for grader and model_selection.train_test_split, while also adding the capability to replicate data splitting, model creation, model grading, and result visualization across a range of user-specified training/testing splits. The user enters a number n of training/testing splits, and a number m of replicate measurements to make for each training/testing split.

Note: Currently this wrapper function is specific to LinearSVC models. It will be useful to create a more generalized wrapper function that can call any learning algorithm later.


```python
def multi_grader_LinearSVC(n,m):
    grades = np.array([])
    train_fractions = np.array([])
    for i in range (1,n):
        test_fraction = (float(i)/float(n))
        for j in range(1,m):
            X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
            X, y, test_size = test_fraction)
            clf = svm.LinearSVC()
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
            new_grade = grader(pred, y_test)
            grades = np.append(grades,new_grade)
            train_fractions = np.append(train_fractions,1-test_fraction)
    train_fractions = np.around(train_fractions, 2)
    out = list(zip(train_fractions, grades))
    sns.set_style('dark')
    sns.set_context("talk")
    figure = sns.boxplot(x=train_fractions, y=grades, color='orange').set(xlabel='Training Fraction', ylabel = 'Grade (%)')
    # return out, train_fractions, grades, figure
    return figure
```

multi_grader_RidgeRegression(a,b,n) tests the RidgeRegression model at a user-specified 'n' training split (between 0 and 1), for 'a' different alpha values ranging from 0 to 'b'. The results are then placed into  boxplots which show the absolute values of the errors in prediction as a function of alpha values.

Let's adjust multi_grader_RidgeRegression to make logarithmically-spaced alpha regularization values. Now the regularization values are powers of ten ranging from -a to +a.


```python
def multi_grader_RidgeRegression(a,b,n):
    errors = np.array([])
    alphas = np.array([])
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
    X, y, test_size=n)
    for i in range(-a,a+1):
        #alpha_fraction = np.ln((float(i)*b/float(a)))
        alpha_fraction = 10 ** (-i)
        reg = linear_model.Ridge(alpha=alpha_fraction)
        reg.fit(X_train,y_train)
        pred = reg.predict(X_test)
        error=abs(pred-y_test)
        errors = np.append(errors,error)
        for k in range(0,len(error)):
            alphas = np.append(alphas, alpha_fraction)
        alphas = np.around(alphas, 2)
        out = list(zip(alphas, errors))
        sns.set_style('dark')
        sns.set_context('talk')
    figure = sns.boxplot(x=alphas, y=errors, color='orange').set(xlabel='Regularization Parameter', ylabel = 'Error Absolute Values')
    return figure
```

# Importing and preparing learning algorithms

We will study a few learning algorithms:

1. The classification algorithm 'LinearSVC'
2. The regression algorithm 'Lasso'


```python
from sklearn import svm #for LinearSVC
```


```python
from sklearn import linear_model #for Lasso
```

# Model Testing

# Classification Model

We begin by testing the 'LinearSVC' classification algorithm. We build models based on varying numbers of enrichment 'categories', and grade their prediction accuracies as a function of training fraction. The results are then summarized with one boxplot demonstrating distribution of grades as a function of training fraction for each number of enrichment 'categories'.


```python
X, y = centrifuge(1,'clf')
multi_grader_LinearSVC(20,100)
```




    [Text(0,0.5,u'Grade (%)'), Text(0.5,0,u'Training Fraction')]




![png](output_29_1.png)



```python
X, y = centrifuge(2,'clf')
multi_grader_LinearSVC(20,100)
```




    [Text(0,0.5,u'Grade (%)'), Text(0.5,0,u'Training Fraction')]




![png](output_30_1.png)



```python
X, y = centrifuge(3,'clf')
multi_grader_LinearSVC(20,50)
```




    [Text(0,0.5,u'Grade (%)'), Text(0.5,0,u'Training Fraction')]




![png](output_31_1.png)



```python
X, y = centrifuge(4,'clf')
multi_grader_LinearSVC(20,100)
```




    [Text(0,0.5,u'Grade (%)'), Text(0.5,0,u'Training Fraction')]




![png](output_32_1.png)



```python
X, y = centrifuge(5,'clf')
multi_grader_LinearSVC(20,100)
```




    [Text(0,0.5,u'Grade (%)'), Text(0.5,0,u'Training Fraction')]




![png](output_33_1.png)


Here are the main observations from the above set of experiments:

1. The best results are usually obtained with a training/testing split of 50%. However, there are significant gains in 'Grade' when the training fraction is increased from 5% to about 20%. After that there are strongly diminishing returns in 'Grade' when further increasing the training fraction.

2. The classification accuracy of the model decreases significantly with each increase in number of enrichment regimes. However, in all situations, the models perform much better than random guessing.

# Regression Model

The first regression model that we will test is the Lasso.


```python
reg = linear_model.Lasso(alpha=0.1)
X, y = centrifuge(3,'reg')
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
    X, y, test_size=0.5)
reg.fit(X_train,y_train)
pred = reg.predict(X_test)
score = grader(pred,y_test)
errors=abs(pred-y_test)
```


```python
avg_error = np.average(errors)
avg_error
```




    0.044737565882256886




```python
plt.hist(errors,bins=100)
```




    (array([ 11.,   8.,   9.,   9.,   6.,   9.,   6.,   5.,   4.,   5.,   7.,
              4.,  10., 101.,   8.,   5.,   8.,   4.,   7.,  11.,   1.,  10.,
             15.,   4.,   5.,   4.,   3.,   6.,   4.,   3.,   5.,   6.,   3.,
              2.,   2.,   3.,   4.,   3.,   1.,   2.,   5.,   3.,   0.,   0.,
              1.,   0.,   1.,   1.,   0.,   2.,   1.,   0.,   1.,   1.,   1.,
              1.,   2.,   1.,   0.,   1.,   0.,   1.,   2.,   1.,   0.,   0.,
              0.,   0.,   0.,   0.,   0.,   1.,   0.,   0.,   1.,   0.,   0.,
              0.,   0.,   1.,   0.,   0.,   1.,   0.,   1.,   0.,   1.,   0.,
              1.,   0.,   0.,   1.,   0.,   0.,   1.,   0.,   1.,   0.,   0.,
              1.]),
     array([1.63744205e-04, 2.35538994e-03, 4.54703567e-03, 6.73868141e-03,
            8.93032714e-03, 1.11219729e-02, 1.33136186e-02, 1.55052643e-02,
            1.76969101e-02, 1.98885558e-02, 2.20802015e-02, 2.42718473e-02,
            2.64634930e-02, 2.86551387e-02, 3.08467845e-02, 3.30384302e-02,
            3.52300759e-02, 3.74217217e-02, 3.96133674e-02, 4.18050131e-02,
            4.39966589e-02, 4.61883046e-02, 4.83799503e-02, 5.05715961e-02,
            5.27632418e-02, 5.49548875e-02, 5.71465333e-02, 5.93381790e-02,
            6.15298247e-02, 6.37214705e-02, 6.59131162e-02, 6.81047619e-02,
            7.02964077e-02, 7.24880534e-02, 7.46796991e-02, 7.68713449e-02,
            7.90629906e-02, 8.12546363e-02, 8.34462821e-02, 8.56379278e-02,
            8.78295735e-02, 9.00212193e-02, 9.22128650e-02, 9.44045107e-02,
            9.65961565e-02, 9.87878022e-02, 1.00979448e-01, 1.03171094e-01,
            1.05362739e-01, 1.07554385e-01, 1.09746031e-01, 1.11937677e-01,
            1.14129322e-01, 1.16320968e-01, 1.18512614e-01, 1.20704260e-01,
            1.22895905e-01, 1.25087551e-01, 1.27279197e-01, 1.29470842e-01,
            1.31662488e-01, 1.33854134e-01, 1.36045780e-01, 1.38237425e-01,
            1.40429071e-01, 1.42620717e-01, 1.44812363e-01, 1.47004008e-01,
            1.49195654e-01, 1.51387300e-01, 1.53578946e-01, 1.55770591e-01,
            1.57962237e-01, 1.60153883e-01, 1.62345528e-01, 1.64537174e-01,
            1.66728820e-01, 1.68920466e-01, 1.71112111e-01, 1.73303757e-01,
            1.75495403e-01, 1.77687049e-01, 1.79878694e-01, 1.82070340e-01,
            1.84261986e-01, 1.86453632e-01, 1.88645277e-01, 1.90836923e-01,
            1.93028569e-01, 1.95220214e-01, 1.97411860e-01, 1.99603506e-01,
            2.01795152e-01, 2.03986797e-01, 2.06178443e-01, 2.08370089e-01,
            2.10561735e-01, 2.12753380e-01, 2.14945026e-01, 2.17136672e-01,
            2.19328318e-01]),
     <a list of 100 Patch objects>)




![png](output_39_1.png)


Let's look at another regression model: Ridge Regression

This function runs ridge regression for a specified number of enrichment levels, test fraction, and iterations at each set of parameters. At each enrichment level, a boxplot will be made showing the distribution of average errors each time the code iterates at that value. The result will be a boxplot showing distribution of average errors for n iterations as a function of w enrichment levels.


```python
def ridge_regression_variations(w_max, test_fraction, n):
    enrichments = np.array([])
    avg_error = np.array([])
    for i in range(1,w_max+1):
        X, y = centrifuge(i,'reg')
        reg = linear_model.Ridge()
        for j in range(0,n):
            X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
                X, y, test_size = test_fraction)
            reg.fit(X_train, y_train)
            pred = reg.predict(X_test)
            errors=abs(pred-y_test)
            average = np.average(errors)
            avg_error = np.append(avg_error, average)
            enrichments = np.append(enrichments, i+1)
            figure = sns.boxplot(x=enrichments, y=avg_error).set(xlabel='Number of enrichment levels', ylabel='Average error in prediction', yscale='log', title='Number of replicates:' + str(n))
    return figure
```


```python
ridge_regression_variations(8,0.5,10)
```

    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 1.02514549765e-22 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 1.5416927806e-22 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 1.23868097574e-22 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 4.26265311552e-22 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 1.0441630032e-22 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 7.34811046129e-23 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 3.45773963873e-23 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 2.20780718632e-23 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 3.12135146386e-23 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 7.23665903605e-23 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 4.08930875118e-23 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 3.22584844816e-23 / 1.11022302463e-16
      RuntimeWarning)
    /Users/adam/anaconda2/lib/python2.7/site-packages/scipy/linalg/basic.py:40: RuntimeWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number/precision: 3.87299301303e-23 / 1.11022302463e-16
      RuntimeWarning)





    [None,
     Text(0,0.5,u'Average error in prediction'),
     Text(0.5,0,u'Number of enrichment levels'),
     Text(0.5,1,u'Number of replicates:10')]




![png](output_43_2.png)



```python
multi_grader_RidgeRegression(4,10,0.5)
```

    /Users/adam/anaconda2/lib/python2.7/site-packages/sklearn/linear_model/ridge.py:125: LinAlgWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number1.002149e-19
      overwrite_a=True).T
    /Users/adam/anaconda2/lib/python2.7/site-packages/sklearn/linear_model/ridge.py:125: LinAlgWarning: scipy.linalg.solve
    Ill-conditioned matrix detected. Result is not guaranteed to be accurate.
    Reciprocal condition number5.134998e-21
      overwrite_a=True).T





    [Text(0,0.5,u'Error Absolute Values'), Text(0.5,0,u'Regularization Parameter')]




![png](output_44_2.png)



```python
def single_ridge_regression(w, test_fraction, n_bins, xmin, xmax, solver):
    X, y = centrifuge(w,'reg')
    reg = linear_model.Ridge(solver=solver)
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
        X, y, test_size = test_fraction)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    errors = abs(pred - y_test)
    avg_error = np.average(errors)
    plt.hist(errors, bins = n_bins)
    plt.xlim(xmin, xmax)
    plt.xlabel('Absolute error in prediction')
    plt.ylabel('Frequency')
    plt.title('Distribution of absolute errors')
    return avg_error
```

The promising solvers appear to be: svd, cholesky, and sparse_cg


```python
rr_svd = single_ridge_regression(5,0.5,500, 0, 0.1, 'svd')
```


![png](output_47_0.png)



```python
n_alphas = 200
alphas = np.logspace(-4,2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
    
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) # reverse axis
plt.xlabel('alphas')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
```


![png](output_48_0.png)



```python
single_ridge_regression(5,0.5,500, 0, 0.1, 'cholesky')
```




    0.000652754531033244




![png](output_49_1.png)



```python
single_ridge_regression(5,0.5,500, 0, 0.1, 'lsqr')
```




    0.14138069286851848




![png](output_50_1.png)



```python
single_ridge_regression(5,0.5,500, 0, 0.1, 'sparse_cg')
```




    0.022608572143524814




![png](output_51_1.png)



```python
single_ridge_regression(5,0.5,500, 0, 0.1, 'sag')
```




    0.23120437427469406




![png](output_52_1.png)



```python
single_ridge_regression(5,0.5,500, 0, 0.1, 'saga')
```




    0.24085253466676013




![png](output_53_1.png)



```python
single_ridge_regression(5, 0.5, 500, 0, 0.1, 'auto')
```




    0.0005148026476750011




![png](output_54_1.png)


^These are the results of RidgeRegression when presented with 4 enrichment levels. Almost all of the errors are less than 1% off!

Let's build a multi-variate ridge regression model. For this experiment, we will simultaneously perform predictions on two parameters: the enrichment and decay time of the uranium. We must therefore construct a new targets array y that includes decay times...

We can also test some of the other features in ridge regression:
1. We still need to look at logarithmically spaced regularization parameters.
2. We should investigate the different solver options: solver= auto, svd, cholesky, lsqr, sparse_cg, sag, saga

Let's build the multivariate labels array y for simultaneous predictions on decay time and enrichment.

Here we begin utilizing the full capabilities of pandas for dataframe manipulation, which will eventually lead to a much smoother and more optimized workflow that what we have had up to this point, utilizing numpy arrays and manual tracking of data labels.


```python
datau235 = pd.read_csv('u235.csv')
datau238 = pd.read_csv('u238.csv')
dfu235 = pd.DataFrame(datau235)
dfu238 = pd.DataFrame(datau235)
dfu235 = dfu235.T
dfu238 = dfu238.T
dfu235.columns = dfu235.iloc[0]
dfu238.columns = dfu238.iloc[0]
dfu235 = dfu235.drop('Energy')
dfu238 = dfu238.drop('Energy')
dfu235.index.name = 'DecayTimeDays'
dfu238.index.name = 'DecayTimeDays'
dfu235 = pd.concat([dfu235], keys=['1'], names=['Enrichment'])
dfu238 = pd.concat([dfu238], keys=['0'], names=['Enrichment'])
dfu = pd.concat([dfu235, dfu238])
index = dfu.index
columns = dfu.columns
values = dfu.values
dfu
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Energy</th>
      <th>1.599</th>
      <th>1.5975</th>
      <th>1.596</th>
      <th>1.5945</th>
      <th>1.593</th>
      <th>1.591</th>
      <th>1.5895</th>
      <th>1.588</th>
      <th>1.5865</th>
      <th>1.585</th>
      <th>...</th>
      <th>0.01678</th>
      <th>0.015185</th>
      <th>0.01359</th>
      <th>0.01199</th>
      <th>0.01039</th>
      <th>0.008791</th>
      <th>0.007193</th>
      <th>0.0055945</th>
      <th>0.003996</th>
      <th>0.0023975</th>
    </tr>
    <tr>
      <th>Enrichment</th>
      <th>DecayTimeDays</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="30" valign="top">1</th>
      <th>0</th>
      <td>5.78e+07</td>
      <td>1330</td>
      <td>3.5e+09</td>
      <td>7.24e-09</td>
      <td>1.17e+07</td>
      <td>7210</td>
      <td>2.32e+06</td>
      <td>317</td>
      <td>0.0136</td>
      <td>134000</td>
      <td>...</td>
      <td>1.28e+10</td>
      <td>1.44e+10</td>
      <td>2.11e+10</td>
      <td>1.24e+10</td>
      <td>2.93e+10</td>
      <td>2.76e+10</td>
      <td>3.55e+10</td>
      <td>7.26e+10</td>
      <td>4.75e+09</td>
      <td>5.06e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.18e+06</td>
      <td>0.0021</td>
      <td>6.71e+09</td>
      <td>3.94e-12</td>
      <td>7.59e+06</td>
      <td>6240</td>
      <td>469000</td>
      <td>2.07e-12</td>
      <td>1.84e-12</td>
      <td>0.0176</td>
      <td>...</td>
      <td>2.08e+09</td>
      <td>2.26e+09</td>
      <td>3.58e+09</td>
      <td>2.04e+09</td>
      <td>4.93e+09</td>
      <td>4.66e+09</td>
      <td>6.07e+09</td>
      <td>1.43e+10</td>
      <td>3.14e+09</td>
      <td>2.46e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>265000</td>
      <td>3.82e-09</td>
      <td>7.58e+09</td>
      <td>3.22e-14</td>
      <td>4.95e+06</td>
      <td>5410</td>
      <td>96400</td>
      <td>3.49e-12</td>
      <td>1.39e-14</td>
      <td>2.76e-09</td>
      <td>...</td>
      <td>1.02e+09</td>
      <td>1.13e+09</td>
      <td>1.87e+09</td>
      <td>1.04e+09</td>
      <td>2.52e+09</td>
      <td>2.38e+09</td>
      <td>3.11e+09</td>
      <td>7.62e+09</td>
      <td>2.35e+09</td>
      <td>1.68e+08</td>
    </tr>
    <tr>
      <th>6</th>
      <td>208000</td>
      <td>6.96e-15</td>
      <td>7.49e+09</td>
      <td>3.28e-15</td>
      <td>3.22e+06</td>
      <td>4700</td>
      <td>19800</td>
      <td>4.99e-12</td>
      <td>4.05e-16</td>
      <td>4.32e-16</td>
      <td>...</td>
      <td>7.08e+08</td>
      <td>7.95e+08</td>
      <td>1.33e+09</td>
      <td>7.32e+08</td>
      <td>1.77e+09</td>
      <td>1.67e+09</td>
      <td>2.19e+09</td>
      <td>5.38e+09</td>
      <td>1.78e+09</td>
      <td>1.26e+08</td>
    </tr>
    <tr>
      <th>7.9</th>
      <td>192000</td>
      <td>9e-19</td>
      <td>7.03e+09</td>
      <td>2.93e-15</td>
      <td>2.1e+06</td>
      <td>4070</td>
      <td>4070</td>
      <td>6.48e-12</td>
      <td>2.9e-16</td>
      <td>1.84e-19</td>
      <td>...</td>
      <td>5.56e+08</td>
      <td>6.24e+08</td>
      <td>1.06e+09</td>
      <td>5.74e+08</td>
      <td>1.39e+09</td>
      <td>1.31e+09</td>
      <td>1.72e+09</td>
      <td>4.24e+09</td>
      <td>1.36e+09</td>
      <td>9.96e+07</td>
    </tr>
    <tr>
      <th>9.9</th>
      <td>183000</td>
      <td>1.46e-18</td>
      <td>6.45e+09</td>
      <td>2.8e-15</td>
      <td>1.37e+06</td>
      <td>3530</td>
      <td>837</td>
      <td>7.97e-12</td>
      <td>2.73e-16</td>
      <td>1.5e-19</td>
      <td>...</td>
      <td>4.62e+08</td>
      <td>5.16e+08</td>
      <td>8.8e+08</td>
      <td>4.75e+08</td>
      <td>1.15e+09</td>
      <td>1.09e+09</td>
      <td>1.42e+09</td>
      <td>3.53e+09</td>
      <td>1.06e+09</td>
      <td>8.3e+07</td>
    </tr>
    <tr>
      <th>11.9</th>
      <td>175000</td>
      <td>2.24e-18</td>
      <td>5.85e+09</td>
      <td>2.69e-15</td>
      <td>893000</td>
      <td>3070</td>
      <td>172</td>
      <td>9.46e-12</td>
      <td>2.58e-16</td>
      <td>1.22e-19</td>
      <td>...</td>
      <td>3.96e+08</td>
      <td>4.41e+08</td>
      <td>7.55e+08</td>
      <td>4.06e+08</td>
      <td>9.81e+08</td>
      <td>9.29e+08</td>
      <td>1.21e+09</td>
      <td>3.03e+09</td>
      <td>8.37e+08</td>
      <td>7.22e+07</td>
    </tr>
    <tr>
      <th>13.9</th>
      <td>170000</td>
      <td>3.22e-18</td>
      <td>5.28e+09</td>
      <td>2.59e-15</td>
      <td>582000</td>
      <td>2660</td>
      <td>35.4</td>
      <td>1.09e-11</td>
      <td>2.44e-16</td>
      <td>9.98e-20</td>
      <td>...</td>
      <td>3.48e+08</td>
      <td>3.86e+08</td>
      <td>6.61e+08</td>
      <td>3.55e+08</td>
      <td>8.57e+08</td>
      <td>8.11e+08</td>
      <td>1.06e+09</td>
      <td>2.66e+09</td>
      <td>6.71e+08</td>
      <td>6.49e+07</td>
    </tr>
    <tr>
      <th>15.8</th>
      <td>166000</td>
      <td>4.41e-18</td>
      <td>4.75e+09</td>
      <td>2.52e-15</td>
      <td>379000</td>
      <td>2310</td>
      <td>7.27</td>
      <td>1.24e-11</td>
      <td>2.31e-16</td>
      <td>8.13e-20</td>
      <td>...</td>
      <td>3.1e+08</td>
      <td>3.42e+08</td>
      <td>5.88e+08</td>
      <td>3.15e+08</td>
      <td>7.61e+08</td>
      <td>7.2e+08</td>
      <td>9.39e+08</td>
      <td>2.36e+09</td>
      <td>5.45e+08</td>
      <td>5.98e+07</td>
    </tr>
    <tr>
      <th>17.8</th>
      <td>162000</td>
      <td>5.8e-18</td>
      <td>4.28e+09</td>
      <td>2.46e-15</td>
      <td>247000</td>
      <td>2000</td>
      <td>1.49</td>
      <td>1.39e-11</td>
      <td>2.19e-16</td>
      <td>6.63e-20</td>
      <td>...</td>
      <td>2.8e+08</td>
      <td>3.08e+08</td>
      <td>5.28e+08</td>
      <td>2.83e+08</td>
      <td>6.84e+08</td>
      <td>6.47e+08</td>
      <td>8.43e+08</td>
      <td>2.12e+09</td>
      <td>4.48e+08</td>
      <td>5.6e+07</td>
    </tr>
    <tr>
      <th>19.8</th>
      <td>1.59E+05</td>
      <td>7.42E-18</td>
      <td>3.84E+09</td>
      <td>2.41E-15</td>
      <td>1.61E+05</td>
      <td>1.74E+03</td>
      <td>3.07E-01</td>
      <td>1.54E-11</td>
      <td>2.07E-16</td>
      <td>5.40E-20</td>
      <td>...</td>
      <td>2.55E+08</td>
      <td>2.80E+08</td>
      <td>4.79E+08</td>
      <td>2.57E+08</td>
      <td>6.21E+08</td>
      <td>5.87E+08</td>
      <td>7.65E+08</td>
      <td>1.92E+09</td>
      <td>3.72E+08</td>
      <td>5.30E+07</td>
    </tr>
    <tr>
      <th>21.8</th>
      <td>1.57E+05</td>
      <td>9.24E-18</td>
      <td>3.45E+09</td>
      <td>2.39E-15</td>
      <td>1.05E+05</td>
      <td>1.51E+03</td>
      <td>6.31E-02</td>
      <td>1.69E-11</td>
      <td>1.96E-16</td>
      <td>4.40E-20</td>
      <td>...</td>
      <td>2.34E+08</td>
      <td>2.56E+08</td>
      <td>4.37E+08</td>
      <td>2.35E+08</td>
      <td>5.68E+08</td>
      <td>5.37E+08</td>
      <td>6.99E+08</td>
      <td>1.75E+09</td>
      <td>3.12E+08</td>
      <td>5.05E+07</td>
    </tr>
    <tr>
      <th>23.8</th>
      <td>1.55E+05</td>
      <td>1.13E-17</td>
      <td>3.10E+09</td>
      <td>2.37E-15</td>
      <td>6.85E+04</td>
      <td>1.31E+03</td>
      <td>1.30E-02</td>
      <td>1.84E-11</td>
      <td>1.86E-16</td>
      <td>3.59E-20</td>
      <td>...</td>
      <td>2.15E+08</td>
      <td>2.35E+08</td>
      <td>4.02E+08</td>
      <td>2.16E+08</td>
      <td>5.22E+08</td>
      <td>4.93E+08</td>
      <td>6.43E+08</td>
      <td>1.60E+09</td>
      <td>2.64E+08</td>
      <td>4.84E+07</td>
    </tr>
    <tr>
      <th>25.7</th>
      <td>1.53E+05</td>
      <td>1.36E-17</td>
      <td>2.78E+09</td>
      <td>2.38E-15</td>
      <td>4.47E+04</td>
      <td>1.13E+03</td>
      <td>2.67E-03</td>
      <td>1.99E-11</td>
      <td>1.76E-16</td>
      <td>2.92E-20</td>
      <td>...</td>
      <td>2.00E+08</td>
      <td>2.18E+08</td>
      <td>3.70E+08</td>
      <td>2.00E+08</td>
      <td>4.83E+08</td>
      <td>4.56E+08</td>
      <td>5.94E+08</td>
      <td>1.47E+09</td>
      <td>2.26E+08</td>
      <td>4.65E+07</td>
    </tr>
    <tr>
      <th>27.7</th>
      <td>1.51E+05</td>
      <td>1.60E-17</td>
      <td>2.50E+09</td>
      <td>2.40E-15</td>
      <td>2.91E+04</td>
      <td>9.82E+02</td>
      <td>5.48E-04</td>
      <td>2.13E-11</td>
      <td>1.67E-16</td>
      <td>2.38E-20</td>
      <td>...</td>
      <td>1.86E+08</td>
      <td>2.02E+08</td>
      <td>3.43E+08</td>
      <td>1.85E+08</td>
      <td>4.48E+08</td>
      <td>4.23E+08</td>
      <td>5.51E+08</td>
      <td>1.35E+09</td>
      <td>1.94E+08</td>
      <td>4.48E+07</td>
    </tr>
    <tr>
      <th>29.7</th>
      <td>1.50E+05</td>
      <td>1.87E-17</td>
      <td>2.25E+09</td>
      <td>2.43E-15</td>
      <td>1.90E+04</td>
      <td>8.52E+02</td>
      <td>1.13E-04</td>
      <td>2.28E-11</td>
      <td>1.58E-16</td>
      <td>1.94E-20</td>
      <td>...</td>
      <td>1.74E+08</td>
      <td>1.88E+08</td>
      <td>3.18E+08</td>
      <td>1.73E+08</td>
      <td>4.17E+08</td>
      <td>3.94E+08</td>
      <td>5.13E+08</td>
      <td>1.25E+09</td>
      <td>1.68E+08</td>
      <td>4.32E+07</td>
    </tr>
    <tr>
      <th>31.7</th>
      <td>1.48E+05</td>
      <td>2.17E-17</td>
      <td>2.02E+09</td>
      <td>2.48E-15</td>
      <td>1.24E+04</td>
      <td>7.39E+02</td>
      <td>2.32E-05</td>
      <td>2.43E-11</td>
      <td>1.49E-16</td>
      <td>1.58E-20</td>
      <td>...</td>
      <td>1.63E+08</td>
      <td>1.76E+08</td>
      <td>2.97E+08</td>
      <td>1.61E+08</td>
      <td>3.90E+08</td>
      <td>3.68E+08</td>
      <td>4.79E+08</td>
      <td>1.16E+09</td>
      <td>1.46E+08</td>
      <td>4.16E+07</td>
    </tr>
    <tr>
      <th>33.7</th>
      <td>1.47E+05</td>
      <td>2.48E-17</td>
      <td>1.81E+09</td>
      <td>2.55E-15</td>
      <td>8.06E+03</td>
      <td>6.41E+02</td>
      <td>4.76E-06</td>
      <td>2.58E-11</td>
      <td>1.42E-16</td>
      <td>1.29E-20</td>
      <td>...</td>
      <td>1.53E+08</td>
      <td>1.65E+08</td>
      <td>2.77E+08</td>
      <td>1.51E+08</td>
      <td>3.65E+08</td>
      <td>3.45E+08</td>
      <td>4.49E+08</td>
      <td>1.08E+09</td>
      <td>1.28E+08</td>
      <td>4.02E+07</td>
    </tr>
    <tr>
      <th>35.6</th>
      <td>1.46E+05</td>
      <td>2.81E-17</td>
      <td>1.63E+09</td>
      <td>2.62E-15</td>
      <td>5.25E+03</td>
      <td>5.56E+02</td>
      <td>9.78E-07</td>
      <td>2.73E-11</td>
      <td>1.34E-16</td>
      <td>1.05E-20</td>
      <td>...</td>
      <td>1.44E+08</td>
      <td>1.55E+08</td>
      <td>2.60E+08</td>
      <td>1.42E+08</td>
      <td>3.43E+08</td>
      <td>3.24E+08</td>
      <td>4.21E+08</td>
      <td>1.01E+09</td>
      <td>1.12E+08</td>
      <td>3.88E+07</td>
    </tr>
    <tr>
      <th>37.6</th>
      <td>1.45E+05</td>
      <td>3.17E-17</td>
      <td>1.46E+09</td>
      <td>2.72E-15</td>
      <td>3.43E+03</td>
      <td>4.82E+02</td>
      <td>2.01E-07</td>
      <td>2.87E-11</td>
      <td>1.27E-16</td>
      <td>8.56E-21</td>
      <td>...</td>
      <td>1.36E+08</td>
      <td>1.46E+08</td>
      <td>2.44E+08</td>
      <td>1.34E+08</td>
      <td>3.23E+08</td>
      <td>3.05E+08</td>
      <td>3.97E+08</td>
      <td>9.41E+08</td>
      <td>9.88E+07</td>
      <td>3.75E+07</td>
    </tr>
    <tr>
      <th>39.6</th>
      <td>1.43E+05</td>
      <td>3.55E-17</td>
      <td>1.31E+09</td>
      <td>2.82E-15</td>
      <td>2.23E+03</td>
      <td>4.18E+02</td>
      <td>4.13E-08</td>
      <td>3.02E-11</td>
      <td>1.21E-16</td>
      <td>6.98E-21</td>
      <td>...</td>
      <td>1.29E+08</td>
      <td>1.38E+08</td>
      <td>2.29E+08</td>
      <td>1.26E+08</td>
      <td>3.05E+08</td>
      <td>2.88E+08</td>
      <td>3.74E+08</td>
      <td>8.81E+08</td>
      <td>8.74E+07</td>
      <td>3.62E+07</td>
    </tr>
    <tr>
      <th>41.6</th>
      <td>1.42E+05</td>
      <td>3.95E-17</td>
      <td>1.18E+09</td>
      <td>2.94E-15</td>
      <td>1.46E+03</td>
      <td>3.63E+02</td>
      <td>8.50E-09</td>
      <td>3.17E-11</td>
      <td>1.14E-16</td>
      <td>5.69E-21</td>
      <td>...</td>
      <td>1.22E+08</td>
      <td>1.30E+08</td>
      <td>2.16E+08</td>
      <td>1.20E+08</td>
      <td>2.89E+08</td>
      <td>2.73E+08</td>
      <td>3.54E+08</td>
      <td>8.27E+08</td>
      <td>7.75E+07</td>
      <td>3.49E+07</td>
    </tr>
    <tr>
      <th>43.5</th>
      <td>1.41E+05</td>
      <td>4.38E-17</td>
      <td>1.06E+09</td>
      <td>3.08E-15</td>
      <td>9.49E+02</td>
      <td>3.15E+02</td>
      <td>1.75E-09</td>
      <td>3.32E-11</td>
      <td>1.09E-16</td>
      <td>4.63E-21</td>
      <td>...</td>
      <td>1.16E+08</td>
      <td>1.24E+08</td>
      <td>2.04E+08</td>
      <td>1.13E+08</td>
      <td>2.74E+08</td>
      <td>2.59E+08</td>
      <td>3.36E+08</td>
      <td>7.78E+08</td>
      <td>6.89E+07</td>
      <td>3.37E+07</td>
    </tr>
    <tr>
      <th>45.5</th>
      <td>1.40E+05</td>
      <td>4.82E-17</td>
      <td>9.50E+08</td>
      <td>3.22E-15</td>
      <td>6.18E+02</td>
      <td>2.73E+02</td>
      <td>3.59E-10</td>
      <td>3.47E-11</td>
      <td>1.03E-16</td>
      <td>3.78E-21</td>
      <td>...</td>
      <td>1.11E+08</td>
      <td>1.17E+08</td>
      <td>1.94E+08</td>
      <td>1.08E+08</td>
      <td>2.60E+08</td>
      <td>2.46E+08</td>
      <td>3.19E+08</td>
      <td>7.33E+08</td>
      <td>6.15E+07</td>
      <td>3.26E+07</td>
    </tr>
    <tr>
      <th>47.5</th>
      <td>1.39E+05</td>
      <td>5.29E-17</td>
      <td>8.53E+08</td>
      <td>3.38E-15</td>
      <td>4.03E+02</td>
      <td>2.37E+02</td>
      <td>7.38E-11</td>
      <td>3.61E-11</td>
      <td>9.78E-17</td>
      <td>3.08E-21</td>
      <td>...</td>
      <td>1.06E+08</td>
      <td>1.12E+08</td>
      <td>1.84E+08</td>
      <td>1.03E+08</td>
      <td>2.47E+08</td>
      <td>2.34E+08</td>
      <td>3.03E+08</td>
      <td>6.93E+08</td>
      <td>5.49E+07</td>
      <td>3.15E+07</td>
    </tr>
    <tr>
      <th>49.5</th>
      <td>1.38E+05</td>
      <td>5.78E-17</td>
      <td>7.66E+08</td>
      <td>3.55E-15</td>
      <td>2.63E+02</td>
      <td>2.05E+02</td>
      <td>1.52E-11</td>
      <td>3.76E-11</td>
      <td>9.29E-17</td>
      <td>2.51E-21</td>
      <td>...</td>
      <td>1.01E+08</td>
      <td>1.07E+08</td>
      <td>1.75E+08</td>
      <td>9.78E+07</td>
      <td>2.36E+08</td>
      <td>2.23E+08</td>
      <td>2.89E+08</td>
      <td>6.55E+08</td>
      <td>4.92E+07</td>
      <td>3.04E+07</td>
    </tr>
    <tr>
      <th>51.5</th>
      <td>1.38E+05</td>
      <td>6.29E-17</td>
      <td>6.88E+08</td>
      <td>3.74E-15</td>
      <td>1.71E+02</td>
      <td>1.78E+02</td>
      <td>3.12E-12</td>
      <td>3.91E-11</td>
      <td>8.82E-17</td>
      <td>2.04E-21</td>
      <td>...</td>
      <td>9.65E+07</td>
      <td>1.02E+08</td>
      <td>1.66E+08</td>
      <td>9.34E+07</td>
      <td>2.25E+08</td>
      <td>2.13E+08</td>
      <td>2.76E+08</td>
      <td>6.21E+08</td>
      <td>4.41E+07</td>
      <td>2.94E+07</td>
    </tr>
    <tr>
      <th>53.4</th>
      <td>1.37E+05</td>
      <td>6.83E-17</td>
      <td>6.18E+08</td>
      <td>3.94E-15</td>
      <td>1.12E+02</td>
      <td>1.55E+02</td>
      <td>6.41E-13</td>
      <td>4.06E-11</td>
      <td>8.38E-17</td>
      <td>1.67E-21</td>
      <td>...</td>
      <td>9.25E+07</td>
      <td>9.74E+07</td>
      <td>1.59E+08</td>
      <td>8.93E+07</td>
      <td>2.16E+08</td>
      <td>2.04E+08</td>
      <td>2.64E+08</td>
      <td>5.90E+08</td>
      <td>3.96E+07</td>
      <td>2.84E+07</td>
    </tr>
    <tr>
      <th>55.4</th>
      <td>1.36E+05</td>
      <td>7.39E-17</td>
      <td>5.55E+08</td>
      <td>4.15E-15</td>
      <td>7.28E+01</td>
      <td>1.34E+02</td>
      <td>1.32E-13</td>
      <td>4.20E-11</td>
      <td>7.96E-17</td>
      <td>1.36E-21</td>
      <td>...</td>
      <td>8.88E+07</td>
      <td>9.33E+07</td>
      <td>1.52E+08</td>
      <td>8.56E+07</td>
      <td>2.07E+08</td>
      <td>1.95E+08</td>
      <td>2.53E+08</td>
      <td>5.62E+08</td>
      <td>3.56E+07</td>
      <td>2.74E+07</td>
    </tr>
    <tr>
      <th>57.4</th>
      <td>135000</td>
      <td>7.96e-17</td>
      <td>4.98e+08</td>
      <td>4.37e-15</td>
      <td>47.4</td>
      <td>116</td>
      <td>2.71e-14</td>
      <td>4.35e-11</td>
      <td>7.56e-17</td>
      <td>1.11e-21</td>
      <td>...</td>
      <td>8.54e+07</td>
      <td>8.96e+07</td>
      <td>1.45e+08</td>
      <td>8.21e+07</td>
      <td>1.98e+08</td>
      <td>1.87e+08</td>
      <td>2.42e+08</td>
      <td>5.35e+08</td>
      <td>3.21e+07</td>
      <td>2.65e+07</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="30" valign="top">0</th>
      <th>296.7</th>
      <td>7.61E+04</td>
      <td>2.41E-15</td>
      <td>1.12E+03</td>
      <td>1.09E-13</td>
      <td>1.60E-06</td>
      <td>3.90E-06</td>
      <td>1.67E-16</td>
      <td>2.17E-10</td>
      <td>8.29E-18</td>
      <td>1.95E-32</td>
      <td>...</td>
      <td>1.17E+07</td>
      <td>1.25E+07</td>
      <td>1.94E+07</td>
      <td>1.14E+07</td>
      <td>2.74E+07</td>
      <td>2.57E+07</td>
      <td>3.32E+07</td>
      <td>6.49E+07</td>
      <td>4.22E+05</td>
      <td>4.29E+05</td>
    </tr>
    <tr>
      <th>298.7</th>
      <td>7.58E+04</td>
      <td>2.44E-15</td>
      <td>1.00E+03</td>
      <td>1.11E-13</td>
      <td>1.59E-06</td>
      <td>3.38E-06</td>
      <td>1.69E-16</td>
      <td>2.18E-10</td>
      <td>8.28E-18</td>
      <td>1.59E-32</td>
      <td>...</td>
      <td>1.16E+07</td>
      <td>1.24E+07</td>
      <td>1.93E+07</td>
      <td>1.13E+07</td>
      <td>2.71E+07</td>
      <td>2.55E+07</td>
      <td>3.29E+07</td>
      <td>6.43E+07</td>
      <td>4.20E+05</td>
      <td>4.15E+05</td>
    </tr>
    <tr>
      <th>300.7</th>
      <td>7.54E+04</td>
      <td>2.47E-15</td>
      <td>9.00E+02</td>
      <td>1.12E-13</td>
      <td>1.58E-06</td>
      <td>2.94E-06</td>
      <td>1.71E-16</td>
      <td>2.19E-10</td>
      <td>8.28E-18</td>
      <td>1.29E-32</td>
      <td>...</td>
      <td>1.15E+07</td>
      <td>1.23E+07</td>
      <td>1.91E+07</td>
      <td>1.12E+07</td>
      <td>2.69E+07</td>
      <td>2.53E+07</td>
      <td>3.26E+07</td>
      <td>6.38E+07</td>
      <td>4.18E+05</td>
      <td>4.02E+05</td>
    </tr>
    <tr>
      <th>302.7</th>
      <td>7.51E+04</td>
      <td>2.50E-15</td>
      <td>8.08E+02</td>
      <td>1.14E-13</td>
      <td>1.57E-06</td>
      <td>2.55E-06</td>
      <td>1.73E-16</td>
      <td>2.21E-10</td>
      <td>8.28E-18</td>
      <td>1.05E-32</td>
      <td>...</td>
      <td>1.14E+07</td>
      <td>1.22E+07</td>
      <td>1.90E+07</td>
      <td>1.11E+07</td>
      <td>2.67E+07</td>
      <td>2.51E+07</td>
      <td>3.23E+07</td>
      <td>6.32E+07</td>
      <td>4.16E+05</td>
      <td>3.89E+05</td>
    </tr>
    <tr>
      <th>304.7</th>
      <td>7.47E+04</td>
      <td>2.54E-15</td>
      <td>7.26E+02</td>
      <td>1.15E-13</td>
      <td>1.57E-06</td>
      <td>2.21E-06</td>
      <td>1.76E-16</td>
      <td>2.22E-10</td>
      <td>8.27E-18</td>
      <td>8.59E-33</td>
      <td>...</td>
      <td>1.13E+07</td>
      <td>1.21E+07</td>
      <td>1.88E+07</td>
      <td>1.10E+07</td>
      <td>2.65E+07</td>
      <td>2.48E+07</td>
      <td>3.20E+07</td>
      <td>6.27E+07</td>
      <td>4.14E+05</td>
      <td>3.76E+05</td>
    </tr>
    <tr>
      <th>306.6</th>
      <td>7.44E+04</td>
      <td>2.57E-15</td>
      <td>6.52E+02</td>
      <td>1.17E-13</td>
      <td>1.56E-06</td>
      <td>1.92E-06</td>
      <td>1.78E-16</td>
      <td>2.23E-10</td>
      <td>8.27E-18</td>
      <td>7.00E-33</td>
      <td>...</td>
      <td>1.12E+07</td>
      <td>1.20E+07</td>
      <td>1.86E+07</td>
      <td>1.09E+07</td>
      <td>2.62E+07</td>
      <td>2.46E+07</td>
      <td>3.18E+07</td>
      <td>6.22E+07</td>
      <td>4.11E+05</td>
      <td>3.64E+05</td>
    </tr>
    <tr>
      <th>308.6</th>
      <td>7.40E+04</td>
      <td>2.60E-15</td>
      <td>5.85E+02</td>
      <td>1.18E-13</td>
      <td>1.55E-06</td>
      <td>1.66E-06</td>
      <td>1.80E-16</td>
      <td>2.25E-10</td>
      <td>8.27E-18</td>
      <td>5.71E-33</td>
      <td>...</td>
      <td>1.11E+07</td>
      <td>1.19E+07</td>
      <td>1.85E+07</td>
      <td>1.08E+07</td>
      <td>2.60E+07</td>
      <td>2.44E+07</td>
      <td>3.15E+07</td>
      <td>6.17E+07</td>
      <td>4.09E+05</td>
      <td>3.53E+05</td>
    </tr>
    <tr>
      <th>310.6</th>
      <td>7.37E+04</td>
      <td>2.64E-15</td>
      <td>5.26E+02</td>
      <td>1.20E-13</td>
      <td>1.54E-06</td>
      <td>1.44E-06</td>
      <td>1.83E-16</td>
      <td>2.26E-10</td>
      <td>8.26E-18</td>
      <td>4.65E-33</td>
      <td>...</td>
      <td>1.10E+07</td>
      <td>1.18E+07</td>
      <td>1.83E+07</td>
      <td>1.07E+07</td>
      <td>2.58E+07</td>
      <td>2.42E+07</td>
      <td>3.13E+07</td>
      <td>6.12E+07</td>
      <td>4.07E+05</td>
      <td>3.41E+05</td>
    </tr>
    <tr>
      <th>312.6</th>
      <td>7.33E+04</td>
      <td>2.67E-15</td>
      <td>4.72E+02</td>
      <td>1.21E-13</td>
      <td>1.53E-06</td>
      <td>1.25E-06</td>
      <td>1.85E-16</td>
      <td>2.28E-10</td>
      <td>8.26E-18</td>
      <td>3.79E-33</td>
      <td>...</td>
      <td>1.09E+07</td>
      <td>1.17E+07</td>
      <td>1.82E+07</td>
      <td>1.07E+07</td>
      <td>2.56E+07</td>
      <td>2.40E+07</td>
      <td>3.10E+07</td>
      <td>6.07E+07</td>
      <td>4.05E+05</td>
      <td>3.31E+05</td>
    </tr>
    <tr>
      <th>314.5</th>
      <td>7.30E+04</td>
      <td>2.71E-15</td>
      <td>4.24E+02</td>
      <td>1.23E-13</td>
      <td>1.52E-06</td>
      <td>1.08E-06</td>
      <td>1.88E-16</td>
      <td>2.29E-10</td>
      <td>8.26E-18</td>
      <td>3.09E-33</td>
      <td>...</td>
      <td>1.08E+07</td>
      <td>1.16E+07</td>
      <td>1.81E+07</td>
      <td>1.06E+07</td>
      <td>2.54E+07</td>
      <td>2.39E+07</td>
      <td>3.08E+07</td>
      <td>6.02E+07</td>
      <td>4.03E+05</td>
      <td>3.20E+05</td>
    </tr>
    <tr>
      <th>316.5</th>
      <td>7.27E+04</td>
      <td>2.74E-15</td>
      <td>3.81E+02</td>
      <td>1.24E-13</td>
      <td>1.52E-06</td>
      <td>9.41E-07</td>
      <td>1.90E-16</td>
      <td>2.30E-10</td>
      <td>8.26E-18</td>
      <td>2.52E-33</td>
      <td>...</td>
      <td>1.07E+07</td>
      <td>1.15E+07</td>
      <td>1.79E+07</td>
      <td>1.05E+07</td>
      <td>2.52E+07</td>
      <td>2.37E+07</td>
      <td>3.05E+07</td>
      <td>5.97E+07</td>
      <td>4.01E+05</td>
      <td>3.10E+05</td>
    </tr>
    <tr>
      <th>318.5</th>
      <td>7.23E+04</td>
      <td>2.78E-15</td>
      <td>3.42E+02</td>
      <td>1.26E-13</td>
      <td>1.51E-06</td>
      <td>8.16E-07</td>
      <td>1.92E-16</td>
      <td>2.32E-10</td>
      <td>8.26E-18</td>
      <td>2.05E-33</td>
      <td>...</td>
      <td>1.06E+07</td>
      <td>1.15E+07</td>
      <td>1.78E+07</td>
      <td>1.04E+07</td>
      <td>2.50E+07</td>
      <td>2.35E+07</td>
      <td>3.03E+07</td>
      <td>5.93E+07</td>
      <td>4.00E+05</td>
      <td>3.00E+05</td>
    </tr>
    <tr>
      <th>320.5</th>
      <td>7.20E+04</td>
      <td>2.81E-15</td>
      <td>3.07E+02</td>
      <td>1.28E-13</td>
      <td>1.50E-06</td>
      <td>7.08E-07</td>
      <td>1.95E-16</td>
      <td>2.33E-10</td>
      <td>8.25E-18</td>
      <td>1.67E-33</td>
      <td>...</td>
      <td>1.05E+07</td>
      <td>1.14E+07</td>
      <td>1.76E+07</td>
      <td>1.03E+07</td>
      <td>2.48E+07</td>
      <td>2.33E+07</td>
      <td>3.00E+07</td>
      <td>5.88E+07</td>
      <td>3.98E+05</td>
      <td>2.90E+05</td>
    </tr>
    <tr>
      <th>322.5</th>
      <td>7.17E+04</td>
      <td>2.85E-15</td>
      <td>2.76E+02</td>
      <td>1.29E-13</td>
      <td>1.49E-06</td>
      <td>6.14E-07</td>
      <td>1.97E-16</td>
      <td>2.35E-10</td>
      <td>8.25E-18</td>
      <td>1.36E-33</td>
      <td>...</td>
      <td>1.04E+07</td>
      <td>1.13E+07</td>
      <td>1.75E+07</td>
      <td>1.03E+07</td>
      <td>2.46E+07</td>
      <td>2.31E+07</td>
      <td>2.98E+07</td>
      <td>5.83E+07</td>
      <td>3.96E+05</td>
      <td>2.81E+05</td>
    </tr>
    <tr>
      <th>324.4</th>
      <td>7.13E+04</td>
      <td>2.88E-15</td>
      <td>2.48E+02</td>
      <td>1.31E-13</td>
      <td>1.48E-06</td>
      <td>5.33E-07</td>
      <td>2.00E-16</td>
      <td>2.36E-10</td>
      <td>8.25E-18</td>
      <td>1.11E-33</td>
      <td>...</td>
      <td>1.03E+07</td>
      <td>1.12E+07</td>
      <td>1.74E+07</td>
      <td>1.02E+07</td>
      <td>2.44E+07</td>
      <td>2.29E+07</td>
      <td>2.96E+07</td>
      <td>5.79E+07</td>
      <td>3.94E+05</td>
      <td>2.72E+05</td>
    </tr>
    <tr>
      <th>326.4</th>
      <td>7.10E+04</td>
      <td>2.92E-15</td>
      <td>2.22E+02</td>
      <td>1.32E-13</td>
      <td>1.47E-06</td>
      <td>4.62E-07</td>
      <td>2.02E-16</td>
      <td>2.37E-10</td>
      <td>8.25E-18</td>
      <td>9.05E-34</td>
      <td>...</td>
      <td>1.03E+07</td>
      <td>1.11E+07</td>
      <td>1.72E+07</td>
      <td>1.01E+07</td>
      <td>2.42E+07</td>
      <td>2.28E+07</td>
      <td>2.93E+07</td>
      <td>5.75E+07</td>
      <td>3.92E+05</td>
      <td>2.64E+05</td>
    </tr>
    <tr>
      <th>328.4</th>
      <td>7.07E+04</td>
      <td>2.95E-15</td>
      <td>2.00E+02</td>
      <td>1.34E-13</td>
      <td>1.47E-06</td>
      <td>4.01E-07</td>
      <td>2.05E-16</td>
      <td>2.39E-10</td>
      <td>8.25E-18</td>
      <td>7.37E-34</td>
      <td>...</td>
      <td>1.02E+07</td>
      <td>1.10E+07</td>
      <td>1.71E+07</td>
      <td>1.00E+07</td>
      <td>2.41E+07</td>
      <td>2.26E+07</td>
      <td>2.91E+07</td>
      <td>5.70E+07</td>
      <td>3.91E+05</td>
      <td>2.56E+05</td>
    </tr>
    <tr>
      <th>330.4</th>
      <td>7.04E+04</td>
      <td>2.99E-15</td>
      <td>1.79E+02</td>
      <td>1.36E-13</td>
      <td>1.46E-06</td>
      <td>3.48E-07</td>
      <td>2.07E-16</td>
      <td>2.40E-10</td>
      <td>8.24E-18</td>
      <td>6.01E-34</td>
      <td>...</td>
      <td>1.01E+07</td>
      <td>1.09E+07</td>
      <td>1.70E+07</td>
      <td>9.94E+06</td>
      <td>2.39E+07</td>
      <td>2.24E+07</td>
      <td>2.89E+07</td>
      <td>5.66E+07</td>
      <td>3.89E+05</td>
      <td>2.48E+05</td>
    </tr>
    <tr>
      <th>332.3</th>
      <td>7.00E+04</td>
      <td>3.03E-15</td>
      <td>1.61E+02</td>
      <td>1.37E-13</td>
      <td>1.45E-06</td>
      <td>3.02E-07</td>
      <td>2.10E-16</td>
      <td>2.41E-10</td>
      <td>8.24E-18</td>
      <td>4.90E-34</td>
      <td>...</td>
      <td>1.00E+07</td>
      <td>1.09E+07</td>
      <td>1.69E+07</td>
      <td>9.87E+06</td>
      <td>2.37E+07</td>
      <td>2.22E+07</td>
      <td>2.87E+07</td>
      <td>5.62E+07</td>
      <td>3.87E+05</td>
      <td>2.40E+05</td>
    </tr>
    <tr>
      <th>334.3</th>
      <td>6.97E+04</td>
      <td>3.06E-15</td>
      <td>1.45E+02</td>
      <td>1.39E-13</td>
      <td>1.44E-06</td>
      <td>2.62E-07</td>
      <td>2.12E-16</td>
      <td>2.43E-10</td>
      <td>8.24E-18</td>
      <td>3.99E-34</td>
      <td>...</td>
      <td>9.93E+06</td>
      <td>1.08E+07</td>
      <td>1.67E+07</td>
      <td>9.80E+06</td>
      <td>2.35E+07</td>
      <td>2.21E+07</td>
      <td>2.85E+07</td>
      <td>5.57E+07</td>
      <td>3.85E+05</td>
      <td>2.32E+05</td>
    </tr>
    <tr>
      <th>336.3</th>
      <td>6.94E+04</td>
      <td>3.10E-15</td>
      <td>1.30E+02</td>
      <td>1.41E-13</td>
      <td>1.43E-06</td>
      <td>2.27E-07</td>
      <td>2.15E-16</td>
      <td>2.44E-10</td>
      <td>8.24E-18</td>
      <td>3.25E-34</td>
      <td>...</td>
      <td>9.86E+06</td>
      <td>1.07E+07</td>
      <td>1.66E+07</td>
      <td>9.72E+06</td>
      <td>2.34E+07</td>
      <td>2.19E+07</td>
      <td>2.83E+07</td>
      <td>5.53E+07</td>
      <td>3.84E+05</td>
      <td>2.25E+05</td>
    </tr>
    <tr>
      <th>338.3</th>
      <td>6.91E+04</td>
      <td>3.14E-15</td>
      <td>1.17E+02</td>
      <td>1.42E-13</td>
      <td>1.43E-06</td>
      <td>1.97E-07</td>
      <td>2.17E-16</td>
      <td>2.46E-10</td>
      <td>8.24E-18</td>
      <td>2.65E-34</td>
      <td>...</td>
      <td>9.78E+06</td>
      <td>1.06E+07</td>
      <td>1.65E+07</td>
      <td>9.65E+06</td>
      <td>2.32E+07</td>
      <td>2.18E+07</td>
      <td>2.81E+07</td>
      <td>5.49E+07</td>
      <td>3.82E+05</td>
      <td>2.18E+05</td>
    </tr>
    <tr>
      <th>340.3</th>
      <td>6.88E+04</td>
      <td>3.18E-15</td>
      <td>1.05E+02</td>
      <td>1.44E-13</td>
      <td>1.42E-06</td>
      <td>1.71E-07</td>
      <td>2.20E-16</td>
      <td>2.47E-10</td>
      <td>8.24E-18</td>
      <td>2.16E-34</td>
      <td>...</td>
      <td>9.70E+06</td>
      <td>1.06E+07</td>
      <td>1.64E+07</td>
      <td>9.58E+06</td>
      <td>2.30E+07</td>
      <td>2.16E+07</td>
      <td>2.78E+07</td>
      <td>5.45E+07</td>
      <td>3.81E+05</td>
      <td>2.11E+05</td>
    </tr>
    <tr>
      <th>342.2</th>
      <td>6.84E+04</td>
      <td>3.21E-15</td>
      <td>9.41E+01</td>
      <td>1.46E-13</td>
      <td>1.41E-06</td>
      <td>1.48E-07</td>
      <td>2.23E-16</td>
      <td>2.48E-10</td>
      <td>8.24E-18</td>
      <td>1.76E-34</td>
      <td>...</td>
      <td>9.63E+06</td>
      <td>1.05E+07</td>
      <td>1.62E+07</td>
      <td>9.51E+06</td>
      <td>2.29E+07</td>
      <td>2.14E+07</td>
      <td>2.76E+07</td>
      <td>5.41E+07</td>
      <td>3.79E+05</td>
      <td>2.05E+05</td>
    </tr>
    <tr>
      <th>344.2</th>
      <td>6.81E+04</td>
      <td>3.25E-15</td>
      <td>8.46E+01</td>
      <td>1.47E-13</td>
      <td>1.40E-06</td>
      <td>1.28E-07</td>
      <td>2.25E-16</td>
      <td>2.50E-10</td>
      <td>8.24E-18</td>
      <td>1.43E-34</td>
      <td>...</td>
      <td>9.56E+06</td>
      <td>1.04E+07</td>
      <td>1.61E+07</td>
      <td>9.45E+06</td>
      <td>2.27E+07</td>
      <td>2.13E+07</td>
      <td>2.74E+07</td>
      <td>5.37E+07</td>
      <td>3.77E+05</td>
      <td>1.98E+05</td>
    </tr>
    <tr>
      <th>346.2</th>
      <td>6.78E+04</td>
      <td>3.29E-15</td>
      <td>7.59E+01</td>
      <td>1.49E-13</td>
      <td>1.40E-06</td>
      <td>1.11E-07</td>
      <td>2.28E-16</td>
      <td>2.51E-10</td>
      <td>8.24E-18</td>
      <td>1.17E-34</td>
      <td>...</td>
      <td>9.49E+06</td>
      <td>1.03E+07</td>
      <td>1.60E+07</td>
      <td>9.38E+06</td>
      <td>2.25E+07</td>
      <td>2.11E+07</td>
      <td>2.72E+07</td>
      <td>5.34E+07</td>
      <td>3.76E+05</td>
      <td>1.92E+05</td>
    </tr>
    <tr>
      <th>348.2</th>
      <td>6.75E+04</td>
      <td>3.33E-15</td>
      <td>6.82E+01</td>
      <td>1.51E-13</td>
      <td>1.39E-06</td>
      <td>9.66E-08</td>
      <td>2.30E-16</td>
      <td>2.52E-10</td>
      <td>8.23E-18</td>
      <td>9.53E-35</td>
      <td>...</td>
      <td>9.41E+06</td>
      <td>1.03E+07</td>
      <td>1.59E+07</td>
      <td>9.31E+06</td>
      <td>2.24E+07</td>
      <td>2.10E+07</td>
      <td>2.71E+07</td>
      <td>5.30E+07</td>
      <td>3.74E+05</td>
      <td>1.86E+05</td>
    </tr>
    <tr>
      <th>350.2</th>
      <td>6.72E+04</td>
      <td>3.36E-15</td>
      <td>6.12E+01</td>
      <td>1.53E-13</td>
      <td>1.38E-06</td>
      <td>8.38E-08</td>
      <td>2.33E-16</td>
      <td>2.54E-10</td>
      <td>8.23E-18</td>
      <td>7.76E-35</td>
      <td>...</td>
      <td>9.34E+06</td>
      <td>1.02E+07</td>
      <td>1.58E+07</td>
      <td>9.25E+06</td>
      <td>2.22E+07</td>
      <td>2.08E+07</td>
      <td>2.69E+07</td>
      <td>5.26E+07</td>
      <td>3.73E+05</td>
      <td>1.81E+05</td>
    </tr>
    <tr>
      <th>352.1</th>
      <td>6.69E+04</td>
      <td>3.40E-15</td>
      <td>5.50E+01</td>
      <td>1.54E-13</td>
      <td>1.37E-06</td>
      <td>7.27E-08</td>
      <td>2.36E-16</td>
      <td>2.55E-10</td>
      <td>8.23E-18</td>
      <td>6.33E-35</td>
      <td>...</td>
      <td>9.28E+06</td>
      <td>1.01E+07</td>
      <td>1.57E+07</td>
      <td>9.18E+06</td>
      <td>2.21E+07</td>
      <td>2.07E+07</td>
      <td>2.67E+07</td>
      <td>5.22E+07</td>
      <td>3.71E+05</td>
      <td>1.75E+05</td>
    </tr>
    <tr>
      <th>354.1</th>
      <td>6.66E+04</td>
      <td>3.44E-15</td>
      <td>4.94E+01</td>
      <td>1.56E-13</td>
      <td>1.37E-06</td>
      <td>6.31E-08</td>
      <td>2.38E-16</td>
      <td>2.57E-10</td>
      <td>8.23E-18</td>
      <td>5.16E-35</td>
      <td>...</td>
      <td>9.21E+06</td>
      <td>1.00E+07</td>
      <td>1.56E+07</td>
      <td>9.12E+06</td>
      <td>2.19E+07</td>
      <td>2.05E+07</td>
      <td>2.65E+07</td>
      <td>5.19E+07</td>
      <td>3.70E+05</td>
      <td>1.70E+05</td>
    </tr>
  </tbody>
</table>
<p>360 rows × 1000 columns</p>
</div>



The above dataframe contains the multi-index pandas dataframe for gamma-ray emissions from u235 and u238 across a range of labeled decay times. We must now build a similar function to centrifuge which creates multi-label target arrays y.


```python
test = index.values
```


```python
test
```




    array([('1', '0'), ('1', '2'), ('1', '4'), ('1', '6'), ('1', '7.9'),
           ('1', '9.9'), ('1', '11.9'), ('1', '13.9'), ('1', '15.8'),
           ('1', '17.8'), ('1', '19.8'), ('1', '21.8'), ('1', '23.8'),
           ('1', '25.7'), ('1', '27.7'), ('1', '29.7'), ('1', '31.7'),
           ('1', '33.7'), ('1', '35.6'), ('1', '37.6'), ('1', '39.6'),
           ('1', '41.6'), ('1', '43.5'), ('1', '45.5'), ('1', '47.5'),
           ('1', '49.5'), ('1', '51.5'), ('1', '53.4'), ('1', '55.4'),
           ('1', '57.4'), ('1', '59.4'), ('1', '61.3'), ('1', '63.3'),
           ('1', '65.3'), ('1', '67.3'), ('1', '69.3'), ('1', '71.2'),
           ('1', '73.2'), ('1', '75.2'), ('1', '77.2'), ('1', '79.1'),
           ('1', '81.1'), ('1', '83.1'), ('1', '85.1'), ('1', '87.1'),
           ('1', '89'), ('1', '91'), ('1', '93'), ('1', '95'), ('1', '97'),
           ('1', '98.9'), ('1', '100.9'), ('1', '102.9'), ('1', '104.9'),
           ('1', '106.8'), ('1', '108.8'), ('1', '110.8'), ('1', '112.8'),
           ('1', '114.8'), ('1', '116.7'), ('1', '118.7'), ('1', '120.7'),
           ('1', '122.7'), ('1', '124.6'), ('1', '126.6'), ('1', '128.6'),
           ('1', '130.6'), ('1', '132.6'), ('1', '134.5'), ('1', '136.5'),
           ('1', '138.5'), ('1', '140.5'), ('1', '142.4'), ('1', '144.4'),
           ('1', '146.4'), ('1', '148.4'), ('1', '150.4'), ('1', '152.3'),
           ('1', '154.3'), ('1', '156.3'), ('1', '158.3'), ('1', '160.3'),
           ('1', '162.2'), ('1', '164.2'), ('1', '166.2'), ('1', '168.2'),
           ('1', '170.1'), ('1', '172.1'), ('1', '174.1'), ('1', '176.1'),
           ('1', '178.1'), ('1', '180'), ('1', '182'), ('1', '184'),
           ('1', '186'), ('1', '187.9'), ('1', '189.9'), ('1', '191.9'),
           ('1', '193.9'), ('1', '195.9'), ('1', '197.8'), ('1', '199.8'),
           ('1', '201.8'), ('1', '203.8'), ('1', '205.7'), ('1', '207.7'),
           ('1', '209.7'), ('1', '211.7'), ('1', '213.7'), ('1', '215.6'),
           ('1', '217.6'), ('1', '219.6'), ('1', '221.6'), ('1', '223.6'),
           ('1', '225.5'), ('1', '227.5'), ('1', '229.5'), ('1', '231.5'),
           ('1', '233.4'), ('1', '235.4'), ('1', '237.4'), ('1', '239.4'),
           ('1', '241.4'), ('1', '243.3'), ('1', '245.3'), ('1', '247.3'),
           ('1', '249.3'), ('1', '251.2'), ('1', '253.2'), ('1', '255.2'),
           ('1', '257.2'), ('1', '259.2'), ('1', '261.1'), ('1', '263.1'),
           ('1', '265.1'), ('1', '267.1'), ('1', '269'), ('1', '271'),
           ('1', '273'), ('1', '275'), ('1', '277'), ('1', '278.9'),
           ('1', '280.9'), ('1', '282.9'), ('1', '284.9'), ('1', '286.9'),
           ('1', '288.8'), ('1', '290.8'), ('1', '292.8'), ('1', '294.8'),
           ('1', '296.7'), ('1', '298.7'), ('1', '300.7'), ('1', '302.7'),
           ('1', '304.7'), ('1', '306.6'), ('1', '308.6'), ('1', '310.6'),
           ('1', '312.6'), ('1', '314.5'), ('1', '316.5'), ('1', '318.5'),
           ('1', '320.5'), ('1', '322.5'), ('1', '324.4'), ('1', '326.4'),
           ('1', '328.4'), ('1', '330.4'), ('1', '332.3'), ('1', '334.3'),
           ('1', '336.3'), ('1', '338.3'), ('1', '340.3'), ('1', '342.2'),
           ('1', '344.2'), ('1', '346.2'), ('1', '348.2'), ('1', '350.2'),
           ('1', '352.1'), ('1', '354.1'), ('0', '0'), ('0', '2'), ('0', '4'),
           ('0', '6'), ('0', '7.9'), ('0', '9.9'), ('0', '11.9'),
           ('0', '13.9'), ('0', '15.8'), ('0', '17.8'), ('0', '19.8'),
           ('0', '21.8'), ('0', '23.8'), ('0', '25.7'), ('0', '27.7'),
           ('0', '29.7'), ('0', '31.7'), ('0', '33.7'), ('0', '35.6'),
           ('0', '37.6'), ('0', '39.6'), ('0', '41.6'), ('0', '43.5'),
           ('0', '45.5'), ('0', '47.5'), ('0', '49.5'), ('0', '51.5'),
           ('0', '53.4'), ('0', '55.4'), ('0', '57.4'), ('0', '59.4'),
           ('0', '61.3'), ('0', '63.3'), ('0', '65.3'), ('0', '67.3'),
           ('0', '69.3'), ('0', '71.2'), ('0', '73.2'), ('0', '75.2'),
           ('0', '77.2'), ('0', '79.1'), ('0', '81.1'), ('0', '83.1'),
           ('0', '85.1'), ('0', '87.1'), ('0', '89'), ('0', '91'),
           ('0', '93'), ('0', '95'), ('0', '97'), ('0', '98.9'),
           ('0', '100.9'), ('0', '102.9'), ('0', '104.9'), ('0', '106.8'),
           ('0', '108.8'), ('0', '110.8'), ('0', '112.8'), ('0', '114.8'),
           ('0', '116.7'), ('0', '118.7'), ('0', '120.7'), ('0', '122.7'),
           ('0', '124.6'), ('0', '126.6'), ('0', '128.6'), ('0', '130.6'),
           ('0', '132.6'), ('0', '134.5'), ('0', '136.5'), ('0', '138.5'),
           ('0', '140.5'), ('0', '142.4'), ('0', '144.4'), ('0', '146.4'),
           ('0', '148.4'), ('0', '150.4'), ('0', '152.3'), ('0', '154.3'),
           ('0', '156.3'), ('0', '158.3'), ('0', '160.3'), ('0', '162.2'),
           ('0', '164.2'), ('0', '166.2'), ('0', '168.2'), ('0', '170.1'),
           ('0', '172.1'), ('0', '174.1'), ('0', '176.1'), ('0', '178.1'),
           ('0', '180'), ('0', '182'), ('0', '184'), ('0', '186'),
           ('0', '187.9'), ('0', '189.9'), ('0', '191.9'), ('0', '193.9'),
           ('0', '195.9'), ('0', '197.8'), ('0', '199.8'), ('0', '201.8'),
           ('0', '203.8'), ('0', '205.7'), ('0', '207.7'), ('0', '209.7'),
           ('0', '211.7'), ('0', '213.7'), ('0', '215.6'), ('0', '217.6'),
           ('0', '219.6'), ('0', '221.6'), ('0', '223.6'), ('0', '225.5'),
           ('0', '227.5'), ('0', '229.5'), ('0', '231.5'), ('0', '233.4'),
           ('0', '235.4'), ('0', '237.4'), ('0', '239.4'), ('0', '241.4'),
           ('0', '243.3'), ('0', '245.3'), ('0', '247.3'), ('0', '249.3'),
           ('0', '251.2'), ('0', '253.2'), ('0', '255.2'), ('0', '257.2'),
           ('0', '259.2'), ('0', '261.1'), ('0', '263.1'), ('0', '265.1'),
           ('0', '267.1'), ('0', '269'), ('0', '271'), ('0', '273'),
           ('0', '275'), ('0', '277'), ('0', '278.9'), ('0', '280.9'),
           ('0', '282.9'), ('0', '284.9'), ('0', '286.9'), ('0', '288.8'),
           ('0', '290.8'), ('0', '292.8'), ('0', '294.8'), ('0', '296.7'),
           ('0', '298.7'), ('0', '300.7'), ('0', '302.7'), ('0', '304.7'),
           ('0', '306.6'), ('0', '308.6'), ('0', '310.6'), ('0', '312.6'),
           ('0', '314.5'), ('0', '316.5'), ('0', '318.5'), ('0', '320.5'),
           ('0', '322.5'), ('0', '324.4'), ('0', '326.4'), ('0', '328.4'),
           ('0', '330.4'), ('0', '332.3'), ('0', '334.3'), ('0', '336.3'),
           ('0', '338.3'), ('0', '340.3'), ('0', '342.2'), ('0', '344.2'),
           ('0', '346.2'), ('0', '348.2'), ('0', '350.2'), ('0', '352.1'),
           ('0', '354.1')], dtype=object)


