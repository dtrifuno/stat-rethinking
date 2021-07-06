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

# Chapter 4: Geocentric Models

```python
%%capture
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pymc3 as pm
import arviz as az

from unthinking import *
```

## Code


### 4.1

```python
pos = stats.uniform.rvs(-1, 1, size=(1000, 16)).sum(axis=1)
dens(pos, xlabel='distance from start')
```

### 4.2

```python
(1 + stats.uniform.rvs(0, 0.1, size=12)).prod()
```

### 4.3

```python
growth = (1 + stats.uniform.rvs(0, 0.1, size=(1000, 12))).prod(axis=1)
dens(growth, xlabel='proportional growth', norm_comp=True)
```

### 4.4

```python
small = (1 + stats.uniform.rvs(0, 0.01, size=(1000, 12))).prod(axis=1)
big = (1 + stats.uniform.rvs(0, 0.5, size=(1000, 12))).prod(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
dens(small, ax=axes[0], norm_comp=True,
     xlabel='proportional growth', title='small growth effects')
dens(big, ax=axes[1], norm_comp=True,
     xlabel='proportional growth', title='large growth effects')
plt.show()
```

### 4.5

```python
log_big = np.log((1 + stats.uniform.rvs(0, 0.5, size=(1000, 12))).prod(axis=1))

dens(log_big, xlabel='log proportional growth', title='log growth effects')
```

### 4.6

```python
w = 6
n = 9
p_grid = np.linspace(0, 1, 100)
posterior = stats.binom.pmf(w, n, p_grid) * stats.uniform.pdf(p_grid, 0, 1)
posterior = posterior / sum(posterior)

plot(x=p_grid, y=posterior, xlabel='probability of water', ylabel='density')
plt.show()
```

### 4.7

```python
df = pd.read_csv('data/Howell1.csv',sep=';')
```

### 4.8

```python
df
```

### 4.9

```python
df.hist()
df.describe(percentiles=(0.055, 0.945))
```

<!-- #region tags=[] -->
### 4.10
<!-- #endregion -->

```python tags=[]
df['height']
```

<!-- #region tags=[] -->
### 4.11
<!-- #endregion -->

```python tags=[]
df2 = df[df['age'] >= 18]
```

```python tags=[]
ax = dens(df2['height'], xlabel='height')
```

<!-- #region tags=[] -->
### 4.12
<!-- #endregion -->

```python tags=[]
pdf = lambda x: stats.norm.pdf(x, 178, 20)
ax = curve(pdf, start=100, end=250, xlabel='mu', ylabel='density')
```

<!-- #region tags=[] -->
### 4.13
<!-- #endregion -->

```python tags=[]
pdf = lambda x: stats.uniform.pdf(x, 0, 50)
ax = curve(pdf, start=-10, end=60, xlabel='sigma', ylabel='density')
```

<!-- #region tags=[] -->
### 4.14
<!-- #endregion -->

```python tags=[]
n = 10_000
sample_mu = stats.norm.rvs(178, 20, n)
sample_sigma = stats.uniform.rvs(0, 50, n)
prior_h = stats.norm.rvs(sample_mu, sample_sigma)

ax = dens(prior_h, xlabel='height')
```

<!-- #region tags=[] -->
### 4.15
<!-- #endregion -->

```python tags=[]
sample_mu = stats.norm.rvs(178, 100, n)
prior_h = stats.norm.rvs(sample_mu, sample_sigma)
ax = az.plot_kde(prior_h)
ax.set(xlabel='height (cm)', ylabel='density')
plt.show()
```

<!-- #region tags=[] -->
### 4.16
<!-- #endregion -->

```python tags=[]
mu = np.linspace(150, 160, 300)
sigma = np.linspace(7, 9, 300)
post_mu, post_sigma = np.meshgrid(mu, sigma)
post_ll = np.zeros(post_mu.shape)

for i in range(len(mu)):
    for j in range(len(sigma)):
        post_ll[i, j] = np.sum(stats.norm.logpdf(df2['height'], post_mu[i, j], post_sigma[i, j]))

post_prod = post_ll + stats.norm.logpdf(post_mu, 178, 20) + stats.uniform.logpdf(post_sigma, 0, 50)
post_prob = np.exp(post_prod - np.max(post_prod))
```

<!-- #region tags=[] -->
### 4.17
<!-- #endregion -->

```python tags=[]
h = plt.contourf(post_mu, post_sigma, post_prob, levels=7)
plt.show()
```

<!-- #region tags=[] -->
### 4.18
<!-- #endregion -->

```python tags=[]
plt.pcolormesh(post_mu, post_sigma, post_prob, shading='auto')
plt.show()
```

<!-- #region tags=[] -->
### 4.19
<!-- #endregion -->

```python tags=[]
flat_mu = post_mu.reshape(-1)
flat_sigma = post_sigma.reshape(-1)
p = post_prob.reshape(-1)
p = p / p.sum()

n = 10_000
sample_i = np.random.choice(len(p), size=n, p=p)
sample_mu = flat_mu[sample_i]
sample_sigma = flat_sigma[sample_i]
```

<!-- #region tags=[] -->
### 4.20
<!-- #endregion -->

```python tags=[]
ax = sns.scatterplot(x=sample_mu, y=sample_sigma, alpha=0.3, s=5)
ax.set(xlabel='mu', ylabel='sigma')
plt.show()
```

<!-- #region tags=[] -->
### 4.21
<!-- #endregion -->

```python tags=[]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

az.plot_kde(sample_mu, ax=axes[0])
axes[0].set(xlabel='mu', ylabel='density')
az.plot_kde(sample_sigma, ax=axes[1])
axes[1].set(xlabel='sigma', ylabel='density')

fig.tight_layout()
plt.show()
```

<!-- #region tags=[] -->
### 4.22
<!-- #endregion -->

```python tags=[]
az.hdi(sample_mu, hdi_prob=0.94), az.hdi(sample_sigma, hdi_prob=0.94)
```

<!-- #region tags=[] -->
### 4.23
<!-- #endregion -->

```python tags=[]
df3 = df2.sample(20)
```

<!-- #region tags=[] -->
### 4.24
<!-- #endregion -->

```python tags=[]
post2_mu, post2_sigma = np.meshgrid(mu, sigma)
post2_ll = np.zeros(post2_mu.shape)

for i in range(len(mu)):
    for j in range(len(sigma)):
        post2_ll[i, j] = np.sum(stats.norm.logpdf(df3['height'], post2_mu[i, j], post2_sigma[i, j]))

post2_prod = post2_ll + stats.norm.logpdf(post2_mu, 178, 20) + stats.uniform.logpdf(post2_sigma, 0, 50)
post2_prob = np.exp(post2_prod - np.max(post2_prod))

flat2_mu = post2_mu.reshape(-1)
flat2_sigma = post2_sigma.reshape(-1)
p2 = post2_prob.reshape(-1)
p2 = p2 / p2.sum()

sample2_i = np.random.choice(len(p2), size=n, p=p2)
sample2_mu = flat2_mu[sample2_i]
sample2_sigma = flat2_sigma[sample2_i]

ax = sns.scatterplot(x=sample2_mu, y=sample2_sigma, alpha=0.3, s=5)
ax.set(xlabel='mu', ylabel='sigma')
plt.show()
```

<!-- #region tags=[] -->
### 4.25
<!-- #endregion -->

```python tags=[]
ax = az.plot_kde(sample2_sigma)
ax.set(xlabel='sigma', ylabel='density')
plt.show()
```

<!-- #region tags=[] -->
### 4.26
<!-- #endregion -->

```python tags=[]
df = pd.read_csv('data/Howell1.csv',sep=';')
df2 = df[df['age'] >= 18]
```

<!-- #region tags=[] -->
### 4.27-4.28
<!-- #endregion -->

```python tags=[]
with pm.Model() as model:
    mu = pm.Normal('mu', mu=178, sigma=20)
    sigma = pm.Uniform('sigma', 0, 50)
    height = pm.Normal('height', mu=mu, sigma=sigma, observed=df2['height'])
    result = quap()
```

<!-- #region tags=[] -->
### 4.29
<!-- #endregion -->

```python tags=[]
precis(result)
```

<!-- #region tags=[] -->
### 4.30
<!-- #endregion -->

```python tags=[]
with pm.Model() as model:
    mu = pm.Normal('mu', mu=178, sigma=20)
    sigma = pm.Uniform('sigma', 0, 50)
    height = pm.Normal('height', mu=mu, sigma=sigma, observed=df2['height'])
    precis(quap(start={'mu': df2['height'].mean(), 'sigma': df2['height'].std()}))
```

<!-- #region tags=[] -->
### 4.31
<!-- #endregion -->

```python tags=[]
with pm.Model() as model:
    mu = pm.Normal('mu', mu=178, sigma=0.1)
    sigma = pm.Uniform('sigma', 0, 50)
    height = pm.Normal('height', mu=mu, sigma=sigma, observed=df2['height'])
    precis(quap())
```

<!-- #region tags=[] -->
### 4.32
<!-- #endregion -->

```python tags=[]
cov = result.cov
cov
```

<!-- #region tags=[] -->
### 4.33
<!-- #endregion -->

```python tags=[]
np.diag(cov), cov_to_cor(cov)
```

<!-- #region tags=[] -->
### 4.34
<!-- #endregion -->

```python tags=[]
post = extract_samples(result, 10_000)
post.head()
```

<!-- #region tags=[] -->
### 4.35
<!-- #endregion -->

```python tags=[]
post.hist()
post.describe(percentiles=(0.055, 0.945))
```

<!-- #region tags=[] -->
### 4.36
<!-- #endregion -->

```python tags=[]
post = stats.multivariate_normal.rvs(mean=result.mean, cov=result.cov, size=10_000)
```

<!-- #region tags=[] -->
### 4.37
<!-- #endregion -->

```python tags=[]
df = pd.read_csv('data/Howell1.csv',sep=';')
df2 = df[df['age'] >= 18]
ax = sns.scatterplot(data=df2, y='height', x='weight')
```

<!-- #region tags=[] -->
### 4.38
<!-- #endregion -->

```python tags=[]
random_state = np.random.RandomState(2971)
n = 100
a = stats.norm.rvs(178, 20, size=n, random_state=random_state)
b = stats.norm.rvs(0, 10, size=n, random_state=random_state)
```

<!-- #region tags=[] -->
### 4.39
<!-- #endregion -->

```python tags=[]
xmin = df2['weight'].min()
xmax = df2['weight'].max()

xs = np.linspace(xmin, xmax, 100)
xbar = df2['weight'].mean()

ax = plt.gca()
ax.axhline(y=0, linewidth=1, linestyle='dashed')
ax.axhline(y=272, linewidth=1)

for alpha, beta in zip(a, b):
    ys = alpha + beta*(xs - xbar)
    sns.lineplot(x=xs, y=ys, color='black', linewidth=1, alpha=0.2, ax=ax)
ax.set(title='b ~ N(0, 10)', xlabel='weight', ylabel='height')
plt.show()
```

<!-- #region tags=[] -->
### 4.40
<!-- #endregion -->

```python tags=[]
b = stats.lognorm.rvs(1, size=10_000)
az.plot_kde(b[b < 5])
plt.show()
```

<!-- #region tags=[] -->
### 4.41
<!-- #endregion -->

```python tags=[]
random_state = np.random.RandomState(2971)
n = 100
a = stats.norm.rvs(178, 20, size=n, random_state=random_state)
b = stats.lognorm.rvs(1, size=n, random_state=random_state)
```

```python tags=[]
xmin = df2['weight'].min()
xmax = df2['weight'].max()

xs = np.linspace(xmin, xmax, 100)
xbar = df2['weight'].mean()

ax = plt.gca()
ax.axhline(y=0, linewidth=1, linestyle='dashed')
ax.axhline(y=272, linewidth=1)

for alpha, beta in zip(a, b):
    ys = alpha + beta*(xs - xbar)
    sns.lineplot(x=xs, y=ys, color='black', linewidth=1, alpha=0.2, ax=ax)
ax.set(title='log(b) ~ N(0, 1)', xlabel='weight', ylabel='height')
plt.show()
```

<!-- #region tags=[] -->
### 4.42
<!-- #endregion -->

```python tags=[]
df = pd.read_csv('data/Howell1.csv',sep=';')
df2 = df[df['age'] >= 18]

xbar = df2['weight'].mean()
x = np.asarray(df2['weight'] - xbar)

with pm.Model() as model:
    a = pm.Normal('a', mu=178, sigma=20)
    b = pm.Lognormal('b', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', 0, 50)
    height = pm.Normal('height', mu=a + b * x, sigma=sigma, observed=df2['height'])
    m43 = quap()
```

<!-- #region tags=[] -->
### 4.43
<!-- #endregion -->

```python tags=[]
# TODO: Fails for me on current version of PyMC3
```

<!-- #region tags=[] -->
### 4.44
<!-- #endregion -->

```python tags=[]
precis(m43)
```

<!-- #region tags=[] -->
### 4.45
<!-- #endregion -->

```python tags=[]
np.round(m43.cov, 3)
```

<!-- #region tags=[] -->
### 4.46
<!-- #endregion -->

```python tags=[]
sns.scatterplot(data=df2, x='weight', y='height')

post = extract_samples(m43, n=10_000)
a_map = post['a'].mean()
b_map = post['b'].mean()

xs = np.linspace(df2['weight'].min(), df2['weight'].max(), 100)
sns.lineplot(x=xs, y=a_map + b_map * (xs - xbar), color='tab:orange', linewidth=2)
plt.show()
```

<!-- #region tags=[] -->
### 4.47
<!-- #endregion -->

```python tags=[]
post = extract_samples(m43, n=10_000)
post.head(5)
```

<!-- #region tags=[] -->
### 4.48
<!-- #endregion -->

```python tags=[]
n = 10
dfn = df2[:n]

x = np.asarray(dfn['weight'] - dfn['weight'].mean())

with pm.Model() as model:
    a = pm.Normal('a', mu=178, sigma=20)
    b = pm.Lognormal('b', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', 0, 50)
    height = pm.Normal('height', mu=a + b * x, sigma=sigma, observed=dfn['height'])
    mn = quap()
```

<!-- #region tags=[] -->
### 4.49
<!-- #endregion -->

```python tags=[]
post = extract_samples(mn, n=20)
ax = plt.gca()
sns.scatterplot(data=dfn, x='weight', y='height')
ax.set(title=f'n = {n}')
for i in range(n):
    sns.lineplot(x=xs, y=post['a'][i] + post['b'][i] * (xs - xbar), color='black', alpha=0.2)
```

<!-- #region tags=[] -->
### 4.50
<!-- #endregion -->

```python tags=[]
post = extract_samples(m43, n=100_000)
mu_at_50 = post['a'] + post['b'] * (50 - xbar)
```

<!-- #region tags=[] -->
### 4.51
<!-- #endregion -->

```python tags=[]
ax = az.plot_kde(mu_at_50)
ax.set(xlabel='mu|weight=50', ylabel='density')
plt.show()
```

<!-- #region tags=[] -->
### 4.52
<!-- #endregion -->

```python tags=[]
mu_at_50.quantile((0.05, 0.94))
```

<!-- #region tags=[] -->
### 4.53
<!-- #endregion -->

```python tags=[]
samples = extract_samples(m43, n=1_000)
a = np.asarray(samples['a'])
b = np.asarray(samples['b'])
k = a + b * np.asarray((df2['weight'] - xbar)).reshape(-1,1)
k.shape
```

<!-- #region tags=[] -->
### 4.54
<!-- #endregion -->

```python tags=[]
samples = extract_samples(m43, n=1_000)
a = np.asarray(samples['a'])
b = np.asarray(samples['b'])
x = np.linspace(25, 70, 46)
mu = a + b * (x - xbar).reshape(-1,1)
mu.shape
```

<!-- #region tags=[] -->
### 4.55
<!-- #endregion -->

```python tags=[]
for x_, row in zip(mu, k):
    sns.scatterplot(x=x_*len(row), y=row, color='tab:blue', linewidth=0)
```

<!-- #region tags=[] -->
### 4.56
<!-- #endregion -->

```python tags=[]
mu_mean = mu.mean(axis=1)
mu_quantile = np.quantile(mu, (0.05, 0.94), axis=1)
```

<!-- #region tags=[] -->
### 4.57
<!-- #endregion -->

```python tags=[]
sns.scatterplot(data=df2, x='weight', y='height')
sns.lineplot(x=x, y=mu_mean, color='tab:red')
plt.fill_between(x, mu_quantile[0], mu_quantile[1], alpha=0.2)
plt.show()
```

<!-- #region tags=[] -->
### 4.58
<!-- #endregion -->

```python tags=[]
post = extract_samples(m43, n=10_000)
a = np.asarray(post['a'])
b = np.asarray(post['b'])
weight_seq = np.linspace(25, 70, 46)
mu = a + b * (x - xbar).reshape(-1,1)
mu_mean = mu.mean(axis=1)
mu_ci = np.quantile(mu, (0.05, 0.94), axis=1)
```

<!-- #region tags=[] -->
### 4.59
<!-- #endregion -->

```python tags=[]
samples = extract_samples(m43, n=1_000)
a = np.asarray(samples['a'])
b = np.asarray(samples['b'])
x = np.linspace(25, 70, 46)
mu = a + b * (x - xbar).reshape(-1,1)

simulated_heights = []
for m, s in zip(mu.T, samples['sigma']):
    simulated_heights.append(stats.norm.rvs(m, s))
    
simulated_heights = np.asarray(simulated_heights)
simulated_heights
```

<!-- #region tags=[] -->
### 4.60
<!-- #endregion -->

```python tags=[]
height_pi = np.quantile(simulated_heights, (0.05, 0.94), axis=0)
```

<!-- #region tags=[] -->
### 4.61
<!-- #endregion -->

```python tags=[]
sns.scatterplot(data=df2, x='weight', y='height')
sns.lineplot(x=x, y=mu_mean)
plt.fill_between(x, mu_quantile[0], mu_quantile[1], alpha=0.2)
plt.fill_between(x, height_pi[0], height_pi[1], alpha=0.2, color=['tab:green'])
plt.show()
```

<!-- #region tags=[] -->
### 4.62
<!-- #endregion -->

```python tags=[]
data = pd.DataFrame({'weight': np.linspace(25, 70, 46)})
sim_height = sim(m43, lambda df: df['a'] + df['b']*(df['weight'] - xbar), data)
pi = sim_height.groupby('weight').quantile([0.05, 0.94]).unstack()
pi.columns = ["low", "high"]

plt.fill_between(pi.index, pi["low"], pi["high"], alpha=0.2, color=['tab:green'])

```

```python
sim_height = sim
```

<!-- #region tags=[] -->
### 4.63
<!-- #endregion -->

```python tags=[]
samples = extract_samples(m43, n=10_000)
a = np.asarray(samples['a'])
b = np.asarray(samples['b'])
x = np.linspace(25, 70, 46).reshape(-1, 1)
mu = a + b * (x - xbar)

simulated_heights = []
for m, s in zip(mu.T, samples['sigma']):
    simulated_heights.append(stats.norm.rvs(m, s))
    
simulated_heights = np.asarray(simulated_heights)
height_pi = np.quantile(simulated_heights, (0.05, 0.94), axis=0)
```

<!-- #region tags=[] -->
### 4.64
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.65
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.66
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.67
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.68
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.69
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.70
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.71
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.72
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.73
<!-- #endregion -->

```python tags=[]
df = pd.read_csv('data/Howell1.csv',sep=';')
df2 = df[df['age'] >= 18]

xbar = df2['weight'].mean()
x = np.asarray(df2['weight'] - xbar)

with pm.Model() as model:
    a = pm.Normal('a', mu=178, sigma=20)
    b = pm.Lognormal('b', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', 0, 50)
    height = pm.Normal('height', mu=a + b * x, sigma=sigma, observed=df2['height'])
    m43 = quap()
```

```python tags=[]
x = model.observed_RVs.copy().pop()
```

```python tags=[]
dir(x)
```

<!-- #region tags=[] -->
### 4.74
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.75
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.76
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.77
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.78
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
### 4.79
<!-- #endregion -->

```python tags=[]

```

<!-- #region tags=[] -->
## Practice
<!-- #endregion -->

### 4E1.

The likelihood is the first line: $y_i = \text{Normal}(\mu, \sigma)$.


### 4E2. 

There are two parameters in the posterior distribution, $\mu$ and $\sigma$.


### 4E3.

$P(\mu, \sigma|y) = \frac{\prod_i \text{Normal}(y_i|\mu, \sigma) \, \text{Normal}(\mu|0, 10) \, \text{Exp}(\sigma|1)}{\iint \prod_i \text{Normal}(y_i|\mu, \sigma) \, \text{Normal}(\mu|0, 10) \, \text{Exp}(\sigma| 1)\, d\mu\, d\sigma }$


### 4E4.

The linear model is the second line: $\mu_i = \alpha + \beta x_i$.

<!-- #region tags=[] -->
### 4E5.

There are three parameters in the posterior distribution: $\alpha$, $\beta$ and $\sigma$.
<!-- #endregion -->

<!-- #region tags=[] -->
### 4M1.
<!-- #endregion -->

```python tags=[]
n = 10_000
mus = stats.norm.rvs(0, 10, n)
sigmas = stats.expon.rvs(size=n)
samples = stats.norm.rvs(loc=mus, scale=sigmas)
sns.histplot(samples)
plt.show()
```

<!-- #region tags=[] -->
### 4M2.
<!-- #endregion -->

```python tags=[]
fake_data = stats.norm.rvs(size=500)

with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=10)
    sigma = pm.Exponential('sigma', lam=1)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=fake_data)
    result = quap(model, vars=[mu, sigma])
```

<!-- #region tags=[] -->
### 4M3.

$$y_i \sim \text{Normal}(\mu, \sigma) $$
$$\mu_i = a + bx_i $$
$$a \sim \text{Normal}(0, 10) $$
$$b \sim \text{Uniform}(0, 1) $$
$$\sigma \sim \text{Exponential}(1) $$
<!-- #endregion -->

<!-- #region tags=[] -->
### 4M4.

$t$ - year of measurement

$t_0$ - year of first measurement

$$ y_{t} = \text{Normal}(\mu_t, \sigma) $$
$$ \mu_{t} = a + b(t - t_0) $$
$$ a \sim \text{Normal}(100, 40) $$
$$ b \sim \text{Logormal}(1.5) $$
$$ \sigma \sim \text{Uniform}(0, 70) $$
<!-- #endregion -->

<!-- #region tags=[] -->
### 4M5.

No, since that assumption is already implied by our use of a lognormal distribution for the parameter $b$. 
<!-- #endregion -->

<!-- #region tags=[] -->
### 4M6.

That means we can replace our prior on $\sigma$ with $\text{Uniform}(0, 63)$.
<!-- #endregion -->

<!-- #region tags=[] -->
### 4M7.
<!-- #endregion -->

```python tags=[]
df = pd.read_csv('data/Howell1.csv',sep=';')
df2 = df[df['age'] >= 18]

x = np.asarray(df2['weight'])

with pm.Model() as model:
    a = pm.Normal('a', mu=178, sigma=20)
    b = pm.Lognormal('b', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', 0, 50)
    height = pm.Normal('height', mu=a + b * x, sigma=sigma, observed=df2['height'])
    m_uncentered = quap(model, vars=[a, b, sigma])
    

uncentered_samples = extract_samples(m_uncentered, n=10_000)
m_uncentered.cov
```

```python tags=[]
m43_samples = extract_samples(m43, n=10_000)

weight_seq = np.linspace(35, 70, 46)
a = np.asarray(m43_samples['a'])
b = np.asarray(m43_samples['b'])

xbar = df2['weight'].mean()

m43_best_fit = (a + b * (weight_seq - xbar).reshape(-1, 1)).mean(axis=1)


a = np.asarray(uncentered_samples['a'])
b = np.asarray(uncentered_samples['b'])
uncentered_best_fit = (a + b * weight_seq.reshape(-1, 1)).mean(axis=1)

sns.lineplot(x=weight_seq, y=m43_best_fit)
sns.lineplot(x=weight_seq, y=uncentered_best_fit)
```

<!-- #region tags=[] -->
### 4M8.
<!-- #endregion -->

<!-- #region tags=[] -->
### 4H1.
<!-- #endregion -->

<!-- #region tags=[] -->
### 4H2.
<!-- #endregion -->

<!-- #region tags=[] -->
### 4H3.
<!-- #endregion -->

<!-- #region tags=[] -->
### 4H4.
<!-- #endregion -->

<!-- #region tags=[] -->
### 4H5.
<!-- #endregion -->

<!-- #region tags=[] -->
### 4H6.
<!-- #endregion -->

<!-- #region tags=[] -->
### 4H7.
<!-- #endregion -->

<!-- #region tags=[] -->
### 4H8.
<!-- #endregion -->
