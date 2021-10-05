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

<!-- #region tags=[] -->
# Chapter 3: Sampling the Imaginary
<!-- #endregion -->

```python tags=[]
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pymc3 as pm
import arviz as az

from unthinking import *

np.random.seed(42)
random_state = np.random.RandomState(42)
```

<!-- #region tags=[] -->
## Code
<!-- #endregion -->

<!-- #region tags=[] -->
### 3.1
<!-- #endregion -->

```python tags=[]
p_positive_vampire = 0.95
p_positive_mortal = 0.01
p_vampire = 0.001
p_positive = p_positive_vampire * p_vampire + p_positive_mortal * (1 - p_vampire)
p_vampire_positive = p_positive_vampire * p_vampire / p_positive
p_vampire_positive
```

<!-- #region tags=[] -->
### 3.2
<!-- #endregion -->

```python tags=[]
p_grid = np.linspace(0, 1, 1000)
prior = np.repeat(1, 1000)
likelihood = stats.binom.pmf(6, 9, p_grid)
unstd_posterior = likelihood * prior
posterior = unstd_posterior / sum(unstd_posterior)
```

<!-- #region tags=[] -->
### 3.3
<!-- #endregion -->

```python tags=[]
samples = np.random.choice(p_grid, p=posterior, size=10_000, replace=True)
```

<!-- #region tags=[] -->
### 3.4
<!-- #endregion -->

```python tags=[]
ax = plot(
    x=range(0, len(samples)),
    y=samples,
    plot_fn=sns.scatterplot,
    alpha=0.2,
    linewidth=0,
    xlabel="sample number",
    ylabel="proportion water (p)",
)
```

<!-- #region tags=[] -->
### 3.5
<!-- #endregion -->

```python tags=[]
ax = dens(samples, xlabel="proportion water (p)")
```

<!-- #region tags=[] -->
### 3.6
<!-- #endregion -->

```python tags=[]
# add up posterior probability where p < 0.5
sum(posterior[p_grid < 0.5])
```

<!-- #region tags=[] -->
### 3.7
<!-- #endregion -->

```python tags=[]
sum(samples < 0.5) / len(samples)
```

<!-- #region tags=[] -->
### 3.8
<!-- #endregion -->

```python tags=[]
sum((samples > 0.5) & (samples < 0.75)) / len(samples)
```

<!-- #region tags=[] -->
### 3.9
<!-- #endregion -->

```python tags=[]
np.quantile(samples, 0.8)
```

<!-- #region tags=[] -->
### 3.10
<!-- #endregion -->

```python tags=[]
np.quantile(samples, (0.1, 0.9))
```

<!-- #region tags=[] -->
### 3.11
<!-- #endregion -->

```python tags=[]
p_grid = np.linspace(0, 1, 1000)
prior = np.repeat(1, 1000)
likelihood = stats.binom.pmf(3, 3, p_grid)
posterior = likelihood * prior
posterior = posterior / sum(posterior)
samples = np.random.choice(p_grid, p=posterior, size=10_000, replace=True)
```

<!-- #region tags=[] -->
### 3.12
<!-- #endregion -->

```python tags=[]
pi(samples, prob=0.5)
```

<!-- #region tags=[] -->
### 3.13
<!-- #endregion -->

```python tags=[]
hdpi(samples, prob=0.5)
```

<!-- #region tags=[] -->
### 3.14
<!-- #endregion -->

```python tags=[]
p_grid[np.argmax(posterior)]
```

<!-- #region tags=[] -->
### 3.15
<!-- #endregion -->

```python tags=[]
stats.mode(samples)[0]
```

<!-- #region tags=[] -->
### 3.16
<!-- #endregion -->

```python tags=[]
np.mean(samples), np.median(samples)
```

<!-- #region tags=[] -->
### 3.17
<!-- #endregion -->

```python tags=[]
sum(posterior * np.abs(0.5 - p_grid))
```

<!-- #region tags=[] -->
### 3.18
<!-- #endregion -->

```python tags=[]
loss = np.vectorize(lambda x: sum(posterior * np.abs(x - p_grid)))(p_grid)
```

<!-- #region tags=[] -->
### 3.19
<!-- #endregion -->

```python tags=[]
p_grid[np.argmin(loss)]
```

<!-- #region tags=[] -->
### 3.20
<!-- #endregion -->

```python tags=[]
stats.binom.pmf(range(0, 3), 2, p=0.7)
```

<!-- #region tags=[] -->
### 3.21
<!-- #endregion -->

```python tags=[]
stats.binom.rvs(2, p=0.7, random_state=random_state)
```

<!-- #region tags=[] -->
### 3.22
<!-- #endregion -->

```python tags=[]
stats.binom.rvs(2, p=0.7, size=10, random_state=random_state)
```

<!-- #region tags=[] -->
### 3.23
<!-- #endregion -->

```python tags=[]
np.bincount(stats.binom.rvs(2, p=0.7, size=10_000, random_state=random_state)) / 10_000
```

<!-- #region tags=[] -->
### 3.24
<!-- #endregion -->

```python tags=[]
dummy_w = stats.binom.rvs(9, p=0.7, size=10_000, random_state=random_state)
ax = simplehist(dummy_w, xlabel="dummy water count")
```

<!-- #region tags=[] -->
### 3.25
<!-- #endregion -->

```python tags=[]
w = stats.binom.rvs(9, p=0.6, size=10_000, random_state=random_state)
```

<!-- #region tags=[] -->
### 3.26
<!-- #endregion -->

```python tags=[]
w = stats.binom.rvs(9, p=samples, random_state=random_state)
```

<!-- #region tags=[] -->
### 3.27
<!-- #endregion -->

```python tags=[]
p_grid = np.linspace(0, 1, 1000)
prior = np.repeat(1, 1000)
likelihood = stats.binom.pmf(6, 9, p_grid)
posterior = likelihood * prior
posterior = posterior / sum(posterior)
samples = np.random.choice(p_grid, p=posterior, size=10_000, replace=True)
```

<!-- #region tags=[] -->
### 3.28
<!-- #endregion -->

```python tags=[]
# fmt: off
birth1 = np.array([1,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0,
0,0,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,
1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,1,1,0,
1,0,1,1,1,0,1,1,1,1])

birth2 = np.array([0,1,0,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,
1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,1,
0,0,0,1,1,1,0,0,0,0])
# fmt: on
```

<!-- #region tags=[] -->
### 3.29
<!-- #endregion -->

```python tags=[]
# Execute cell 3.28 instead to load the data.
```

<!-- #region tags=[] -->
### 3.30
<!-- #endregion -->

```python tags=[]
sum(birth1) + sum(birth2)
```

<!-- #region tags=[] -->
## Practice
<!-- #endregion -->

<!-- #region tags=[] -->
### 3E1.
<!-- #endregion -->

```python tags=[]
sum(samples < 0.2) / len(samples)
```

<!-- #region tags=[] -->
### 3E2.
<!-- #endregion -->

```python tags=[]
sum(samples > 0.8) / len(samples)
```

<!-- #region tags=[] -->
### 3E3.
<!-- #endregion -->

```python tags=[]
sum((samples > 0.2) & (samples < 0.8)) / len(samples)
```

<!-- #region tags=[] -->
### 3E4.
<!-- #endregion -->

```python tags=[]
np.percentile(samples, 20)
```

<!-- #region tags=[] -->
### 3E5.
<!-- #endregion -->

```python tags=[]
np.percentile(samples, 100 - 20)
```

<!-- #region tags=[] -->
### 3E6.
<!-- #endregion -->

```python tags=[]
hdpi(samples, prob=0.66)
```

<!-- #region tags=[] -->
### 3E7.
<!-- #endregion -->

```python tags=[]
pi(samples, prob=0.66)
```

<!-- #region tags=[] -->
### 3M1.
<!-- #endregion -->

```python tags=[]
p_grid = np.linspace(0, 1, 1000)
prior = np.repeat(1, 1000)
likelihood = stats.binom.pmf(8, 15, p_grid)
posterior = likelihood * prior
posterior = posterior / sum(posterior)
```

<!-- #region tags=[] -->
### 3M2.
<!-- #endregion -->

```python tags=[]
samples = np.random.choice(p_grid, p=posterior, size=10_000, replace=True)
hdpi(samples, prob=0.9)
```

<!-- #region tags=[] -->
### 3M3.
<!-- #endregion -->

```python tags=[]
w = stats.binom.rvs(15, p=samples)
sum(w == 8) / len(w)
```

<!-- #region tags=[] -->
### 3M4.
<!-- #endregion -->

```python tags=[]
w = stats.binom.rvs(9, p=samples)
sum(w == 6) / len(w)
```

<!-- #region tags=[] -->
### 3M5.
<!-- #endregion -->

```python tags=[]
p_grid = np.linspace(0, 1, 1000)
prior = np.concatenate((np.repeat(0, 500), np.repeat(2, 500)))
likelihood = stats.binom.pmf(8, 15, p_grid)
posterior = likelihood * prior
posterior = posterior / sum(posterior)

samples = np.random.choice(p_grid, p=posterior, size=10_000, replace=True)
print(hdpi(samples, prob=0.9))

w = stats.binom.rvs(15, p=samples)
print(sum(w == 8) / len(w))

w = stats.binom.rvs(9, p=samples)
print(sum(w == 6) / len(w))
```

<!-- #region tags=[] -->
### 3M6.
<!-- #endregion -->

```python tags=[]
true_p = 0.5  # expect n to be largest the closer true_p is to 0.5

n = 10
while True:
    data = stats.binom.rvs(n, true_p, random_state=random_state)
    p_grid = np.linspace(0, 1, 1000)
    prior = np.repeat(1, 1000)
    likelihood = stats.binom.pmf(data, n, p_grid)
    posterior = likelihood * prior
    posterior = posterior / sum(posterior)
    samples = np.random.choice(p_grid, p=posterior, size=10_000, replace=True)

    interval = pi(samples, 0.99)
    width = interval[1] - interval[0]
    if width < 0.05:
        print(n)
        print(interval)
        break
    n = int(n * 1.25)
```

<!-- #region tags=[] -->
### 3H1.
<!-- #endregion -->

```python tags=[]
k = sum(birth1) + sum(birth2)
n = len(birth1) + len(birth2)

p_grid = np.linspace(0, 1, 1000)
prior = np.repeat(1, 1000)
likelihood = stats.binom.pmf(k, n, p=p_grid)
posterior = likelihood * prior
posterior = posterior / sum(posterior)
p_grid[np.argmax(posterior)]
```

<!-- #region tags=[] -->
### 3H2.
<!-- #endregion -->

```python tags=[]
samples = np.random.choice(p_grid, p=posterior, size=10_000, replace=True)
for p in (0.50, 0.89, 0.97):
    print("{}\n".format(hdpi(samples, prob=p)))
```

<!-- #region tags=[] -->
### 3H3.
<!-- #endregion -->

```python tags=[]
w = stats.binom.rvs(n, p=samples, random_state=random_state)

plt.axvline(x=k, color="tab:red", linewidth=3)
ax = simplehist(w, xlabel="male births per 200")
legend = plt.legend(["actual", "predicted"], loc="upper left")
```

The actual number of births is right in the middle of the histogram and appears to be a quite frequent outcome. Hence the model fits the data well.

<!-- #region tags=[] -->
### 3H4.
<!-- #endregion -->

```python tags=[]
w = stats.binom.rvs(n // 2, p=samples, random_state=random_state)

plt.axvline(x=sum(birth1), color="tab:red", linewidth=3)
ax = simplehist(w, xlabel="male births per 100")
legend = plt.legend(["actual", "predicted"], loc="upper left")
```

The actual number of births is off center in the histogram, but still appears to be a fairly frequent outcome. The model fit no longer looks quite as strong, but it does not seem inappropriate either.

<!-- #region tags=[] -->
### 3H5.
<!-- #endregion -->

```python tags=[]
born_after_girl = birth2[birth1 == 0]

w = stats.binom.rvs(len(born_after_girl), p=samples, random_state=random_state)
plt.axvline(x=sum(born_after_girl), color="tab:red", linewidth=3)
ax = simplehist(w, xlabel="male births per 100")
legend = plt.legend(["actual", "predicted"], loc="upper left")
```

The actual result corresponds to what seems to be a very unlikely event in the model, so it seems probable that our assumption that the sex of the first child does not influence the sex of the second child was incorrect.
