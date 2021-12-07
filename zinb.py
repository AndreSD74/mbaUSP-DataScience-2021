import pyreadr
import pandas as pd

arq = pyreadr.read_r('data/corruption.RData') 
paises = arq['corruption']

print(paises.head())

# em R seria
################################################################################
#        ESTIMAÇÃO DO MODELO ZERO-INFLATED BINOMIAL NEGATIVO (ZINB)            #
################################################################################
#Estimação do modelo ZINB pela função zeroinfl do pacote pscl
# modelo_zinb <- zeroinfl(formula = violations ~ corruption + post + staff
#                         | corruption,
#                         data = corruption,
#                         dist = "negbin")

# em Python:
# https://timeseriesreasoning.com/contents/zero-inflated-poisson-regression-model/


from matplotlib import pyplot as plt
import numpy as np
import math as math
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from patsy import dmatrices

# expressao correta que não funciona com patsy dmatrices
# expr = 'violations ~ corruption + post + staff | corruption'

expr = 'violations ~ corruption + post + staff'

Y, x = dmatrices(expr, paises, return_type='dataframe')

print("POISSON")

poisson_model = sm.Poisson(endog=Y, exog=x).fit()

print(poisson_model.summary())

print(poisson_model.predict())

print("ZERO INFLATED POISSON")

zip_model = sm.ZeroInflatedPoisson(endog=Y, exog=x, exog_infl=x, inflation='logit').fit()

print(zip_model.summary())

print("NEGATIVE BINOMIAL")

nb_model = sm.NegativeBinomial(Y, x).fit()

print(nb_model.summary())

print("ZERO INFLATED NEGATIVE BINOMIAL")

# zinb_model = sm.ZeroInflatedNegativeBinomialP(endog=Y, exog=x, exog_infl=x, inflation='logit', p=2).fit()
# com o parametro exog_infl=x fica tudo nan
# sem o parametro exog_infl=x fica igual ao NegativeBinomial
# zinb_model = sm.ZeroInflatedNegativeBinomialP(endog=Y, exog=x, inflation='logit', p=2).fit()

zinb_model = sm.ZeroInflatedNegativeBinomialP(endog=Y, exog=x, inflation='logit', p=2).fit(method="nm", maxiter=50)

print(zinb_model.summary())


# testando expressao correta com Formula: ainda não está implementado interpretar o | na formula
# from formulaic import Formula
# expr = 'violations ~ corruption + post + staff | corruption'
# Y, x = Formula(expr).get_model_matrix(paises)

from formulae import design_matrices

expr = 'violations ~ corruption + post + staff | corruption'
dm = design_matrices(expr, paises)

teste1 = dm.common.as_dataframe()
print(teste1.head())
# zinb_model = sm.ZeroInflatedNegativeBinomialP(endog=Y, exog=x, inflation='logit', p=2).fit(method="nm", maxiter=50)

# print(zinb_model.summary())