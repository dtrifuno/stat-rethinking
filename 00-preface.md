---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.3
  kernelspec:
    display_name: ml38
    language: python
    name: ml38
---

# Preface

```python
%%capture
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
```

## Code


### 0.1

```python
print('All models are wrong, but some are useful.')
```

### 0.2

```python
x = np.linspace(1, 2, 2)
x = x * 10
x = np.log(x)
x = np.sum(x)
x = np.exp(x)
x
```

### 0.3

```python
np.log(0.01**200), 200 * np.log(0.01)
```

### 0.4

```python
from statsmodels.formula.api import ols

# Load the data:
# car braking distances in feet paired with speeds in km/h
cars = sm.datasets.get_rdataset('cars').data

# fit a linear regression of distance on speed
cars_model = ols("dist ~ speed", data=cars).fit()

# estimated coefficients from the model
print(cars_model.params)

# plot residuals against speed
ax = plt.gca()
ax.plot(cars['dist'], cars_model.resid, 'o')
ax.axhline(y=0, color='black')
ax.set(xlabel='speed', ylabel='residual')
ax.set_title('Residuals versus speed', fontsize='large')
plt.show()
```

### 0.5

```python
# this is an R specific requirement
```
