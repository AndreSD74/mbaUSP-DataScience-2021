import pandas
from formulaic import Formula

df = pandas.DataFrame({
    'y': [0,1,2],
    'x': ['A', 'B', 'C'],
    'z': [0.3, 0.1, 0.2],
})

y, X = Formula('y ~ x + z').get_model_matrix(df)

print(X)

print(y)
