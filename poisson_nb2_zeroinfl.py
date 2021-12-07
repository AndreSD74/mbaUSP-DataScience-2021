##
# https://timeseriesreasoning.com/contents/negative-binomial-regression-model/
# 
# https://github.com/bambinos/formulae/blob/master/docs/notebooks/getting_started.ipynb
# 
# https://timeseriesreasoning.com/contents/zero-inflated-poisson-regression-model/
# 
##

import pyreadr
import pandas as pd
import numpy as np
import math as math
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

# lendo arquivo de dados em formato RData 

arq = pyreadr.read_r('data/corruption.RData') 
paises = arq['corruption']

# formula: variavel y (violations) depende das variáveis corruption, post e staff
expr = 'violations ~ corruption + post + staff'

# gera matrices 
Y, x = dmatrices(expr, paises, return_type='dataframe')

print("POISSON usando GLM")

#Usando  statsmodels GLM para criar um modelo Poisson
poisson_model = sm.GLM(Y, x, family=sm.families.Poisson()).fit()

print(poisson_model.summary())

print("POISSON usando a função Poisson")

poisson2_model = sm.Poisson(endog=Y, exog=x).fit()

print(poisson2_model.summary())

print("Binomial Negativa usando GLM")

paises['LAMBDA'] = poisson_model.mu

#add a derived column called 'AUX_OLS_DEP' to the pandas Data Frame. 
# This new column will store the values of the dependent variable of the OLS regression
paises['AUX_OLS_DEP'] = paises.apply(lambda x: ((x['violations'] - x['LAMBDA'])**2 - x['LAMBDA']) / x['LAMBDA'], axis=1)

#use patsy to form the model specification for the OLSR
ols_expr = 'AUX_OLS_DEP ~ LAMBDA - 1'

#Configure and fit the OLSR model
aux_olsr_results = smf.ols(ols_expr, paises).fit()

#Print the regression params
print(aux_olsr_results.params)

#train the NB2 model on the training data set
nb2_model = sm.GLM(Y, x, family=sm.families.NegativeBinomial(alpha=aux_olsr_results.params[0])).fit()

#print the training summary
print(nb2_model.summary())

print("NEGATIVE BINOMIAL")

nb_model = sm.NegativeBinomial(Y, x).fit()

print(nb_model.summary())

print("POISSON ZERO INFLATED")

import statsmodels.discrete.count_model as cm

zip_model = cm.ZeroInflatedPoisson(Y, x, exog_infl=x).fit(maxiter=50)

print(zip_model.summary())

zip_model = cm.ZeroInflatedPoisson(Y, x, exog_infl=x).fit_regularized(maxiter=50)

print(zip_model.summary())

zip_model.predict([1, 0, 1.5, 43], exog_infl=np.ones(4))

print("NEGATIVE BINOMIAL ZERO INFLATED")

zinb_model = cm.ZeroInflatedNegativeBinomialP(Y, x, exog_infl=x).fit(maxiter=50)

print(zinb_model.summary())