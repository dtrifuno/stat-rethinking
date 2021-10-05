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

np.random.seed(42)
random_state = np.random.RandomState(42)
```

## Code


### 4.1

```python
pos = stats.uniform.rvs(-1, 1, size=(1000, 16), random_state=random_state).sum(axis=1)
ax = dens(pos, xlabel="distance from start")
```

### 4.2

```python
(1 + stats.uniform.rvs(0, 0.1, size=12, random_state=random_state)).prod()
```

### 4.3

```python
growth = (
    1 + stats.uniform.rvs(0, 0.1, size=(1000, 12), random_state=random_state)
).prod(axis=1)
ax = dens(growth, xlabel="proportional growth", norm_comp=True)
```

### 4.4

```python
small = (
    1 + stats.uniform.rvs(0, 0.01, size=(1000, 12), random_state=random_state)
).prod(axis=1)
big = (1 + stats.uniform.rvs(0, 0.5, size=(1000, 12), random_state=random_state)).prod(
    axis=1
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
dens(
    small,
    ax=axes[0],
    norm_comp=True,
    xlabel="proportional growth",
    title="small growth effects",
)
dens(
    big,
    ax=axes[1],
    norm_comp=True,
    xlabel="proportional growth",
    title="large growth effects",
)
fig.tight_layout()
```

### 4.5

```python
log_big = np.log(
    (1 + stats.uniform.rvs(0, 0.5, size=(1000, 12), random_state=random_state)).prod(
        axis=1
    )
)

ax = dens(
    log_big,
    xlabel="log proportional growth",
    title="log growth effects",
    norm_comp=True,
)
```

### 4.6

```python
w = 6
n = 9
p_grid = np.linspace(0, 1, 100)
posterior = stats.binom.pmf(w, n, p_grid) * stats.uniform.pdf(p_grid, 0, 1)
posterior = posterior / sum(posterior)

ax = plot(x=p_grid, y=posterior, xlabel="probability of water", ylabel="density")
```

### 4.7

```python
df = pd.read_csv("data/Howell1.csv", sep=";")
```

### 4.8

```python
df
```

### 4.9

```python
precis(df)
```

<!-- #region tags=[] -->
### 4.10
<!-- #endregion -->

```python tags=[]
df["height"]
```

<!-- #region tags=[] -->
### 4.11
<!-- #endregion -->

```python tags=[]
df2 = df[df["age"] >= 18]
```

```python tags=[]
ax = dens(df2["height"], xlabel="height")
```

<!-- #region tags=[] -->
### 4.12
<!-- #endregion -->

```python tags=[]
pdf = lambda x: stats.norm.pdf(x, 178, 20)
ax = curve(pdf, start=100, end=250, xlabel="mu", ylabel="density")
```

<!-- #region tags=[] -->
### 4.13
<!-- #endregion -->

```python tags=[]
pdf = lambda x: stats.uniform.pdf(x, 0, 50)
ax = curve(pdf, start=-10, end=60, xlabel="sigma", ylabel="density")
```

<!-- #region tags=[] -->
### 4.14
<!-- #endregion -->

```python tags=[]
n = 10_000
sample_mu = stats.norm.rvs(178, 20, n)
sample_sigma = stats.uniform.rvs(0, 50, n)
prior_h = stats.norm.rvs(sample_mu, sample_sigma)

ax = dens(prior_h, xlabel="height")
```

<!-- #region tags=[] -->
### 4.15
<!-- #endregion -->

```python tags=[]
sample_mu = stats.norm.rvs(178, 100, n)
prior_h = stats.norm.rvs(sample_mu, sample_sigma)
ax = dens(prior_h, xlabel="height")
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
        post_ll[i, j] = np.sum(
            stats.norm.logpdf(df2["height"], post_mu[i, j], post_sigma[i, j])
        )

post_prod = (
    post_ll
    + stats.norm.logpdf(post_mu, 178, 20)
    + stats.uniform.logpdf(post_sigma, 0, 50)
)
post_prob = np.exp(post_prod - np.max(post_prod))
```

<!-- #region tags=[] -->
### 4.17
<!-- #endregion -->

```python tags=[]
cs = plt.contourf(post_mu, post_sigma, post_prob, levels=7)
```

<!-- #region tags=[] -->
### 4.18
<!-- #endregion -->

```python tags=[]
qm = plt.pcolormesh(post_mu, post_sigma, post_prob, shading="auto")
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
ax = plot(
    x=sample_mu,
    y=sample_sigma,
    plot_fn=sns.scatterplot,
    alpha=0.3,
    s=5,
    xlabel="mu",
    ylabel="sigma",
)
```

<!-- #region tags=[] -->
### 4.21
<!-- #endregion -->

```python tags=[]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

dens(sample_mu, ax=axes[0], xlabel="mu")
dens(sample_sigma, ax=axes[1], xlabel="sigma")

fig.tight_layout()
```

<!-- #region tags=[] -->
### 4.22
<!-- #endregion -->

```python tags=[]
hdpi(sample_mu, prob=0.94), hdpi(sample_sigma, prob=0.94)
```

<!-- #region tags=[] -->
### 4.23
<!-- #endregion -->

```python tags=[]
df3 = df2.sample(20, random_state=random_state)
```

<!-- #region tags=[] -->
### 4.24
<!-- #endregion -->

```python tags=[]
post2_mu, post2_sigma = np.meshgrid(mu, sigma)
post2_ll = np.zeros(post2_mu.shape)

for i in range(len(mu)):
    for j in range(len(sigma)):
        post2_ll[i, j] = np.sum(
            stats.norm.logpdf(df3["height"], post2_mu[i, j], post2_sigma[i, j])
        )

post2_prod = (
    post2_ll
    + stats.norm.logpdf(post2_mu, 178, 20)
    + stats.uniform.logpdf(post2_sigma, 0, 50)
)
post2_prob = np.exp(post2_prod - np.max(post2_prod))

flat2_mu = post2_mu.reshape(-1)
flat2_sigma = post2_sigma.reshape(-1)
p2 = post2_prob.reshape(-1)
p2 = p2 / p2.sum()

sample2_i = np.random.choice(len(p2), size=n, p=p2)
sample2_mu = flat2_mu[sample2_i]
sample2_sigma = flat2_sigma[sample2_i]

ax = plot(
    x=sample2_mu,
    y=sample2_sigma,
    plot_fn=sns.scatterplot,
    alpha=0.3,
    s=5,
    xlabel="mu",
    ylabel="sigma",
)
```

<!-- #region tags=[] -->
### 4.25
<!-- #endregion -->

```python tags=[]
ax = dens(sample2_sigma, xlabel="sigma", norm_comp=True)
```

<!-- #region tags=[] -->
### 4.26
<!-- #endregion -->

```python tags=[]
df = pd.read_csv("data/Howell1.csv", sep=";")
df2 = df[df["age"] >= 18]
```

<!-- #region tags=[] -->
### 4.27-4.28
<!-- #endregion -->

```python tags=[]
with pm.Model() as model:
    mu = pm.Normal("mu", mu=178, sigma=20)
    sigma = pm.Uniform("sigma", 0, 50)
    height = pm.Normal("height", mu=mu, sigma=sigma, observed=df2["height"])
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
    mu = pm.Normal("mu", mu=178, sigma=20)
    sigma = pm.Uniform("sigma", 0, 50)
    height = pm.Normal("height", mu=mu, sigma=sigma, observed=df2["height"])
    precis(quap(start={"mu": df2["height"].mean(), "sigma": df2["height"].std()}))
```

<!-- #region tags=[] -->
### 4.31
<!-- #endregion -->

```python tags=[]
with pm.Model() as model:
    mu = pm.Normal("mu", mu=178, sigma=0.1)
    sigma = pm.Uniform("sigma", 0, 50)
    height = pm.Normal("height", mu=mu, sigma=sigma, observed=df2["height"])
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
post = extract_samples(result, 10_000, random_state=random_state)
post.head()
```

<!-- #region tags=[] -->
### 4.35
<!-- #endregion -->

```python tags=[]
precis(post)
```

<!-- #region tags=[] -->
### 4.36
<!-- #endregion -->

```python tags=[]
post = stats.multivariate_normal.rvs(
    mean=result.mean, cov=result.cov, size=10_000, random_state=random_state
)
```

<!-- #region tags=[] -->
### 4.37
<!-- #endregion -->

```python tags=[]
df = pd.read_csv("data/Howell1.csv", sep=";")
df2 = df[df["age"] >= 18]
ax = plot(data=df2, y="height", x="weight", plot_fn=sns.scatterplot)
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
xmin = df2["weight"].min()
xmax = df2["weight"].max()

xs = np.linspace(xmin, xmax, 100)
xbar = df2["weight"].mean()

ax = plt.gca()
ax.set(title="b ~ N(0, 10)", xlabel="weight", ylabel="height")
ax.axhline(y=0, linewidth=1, linestyle="dashed")
ax.axhline(y=272, linewidth=1)

for alpha, beta in zip(a, b):
    ys = alpha + beta * (xs - xbar)
    plot(x=xs, y=ys, plot_fn=sns.lineplot, color="black", linewidth=1, alpha=0.2, ax=ax)
```

<!-- #region tags=[] -->
### 4.40
<!-- #endregion -->

```python tags=[]
b = stats.lognorm.rvs(1, size=10_000, random_state=random_state)
ax = dens(b[b < 5], xlabel="b")
```

<!-- #region tags=[] -->
### 4.41
<!-- #endregion -->

```python tags=[]
n = 100
a = stats.norm.rvs(178, 20, size=n, random_state=random_state)
b = stats.lognorm.rvs(1, size=n, random_state=random_state)
```

```python tags=[]
xmin = df2["weight"].min()
xmax = df2["weight"].max()
xbar = df2["weight"].mean()

ax = plt.gca()
ax.set(title="log(b) ~ N(0, 1)", xlabel="weight", ylabel="height")
ax.axhline(y=0, linewidth=1, linestyle="dashed")
ax.axhline(y=272, linewidth=1)

for alpha, beta in zip(a, b):
    curve(
        lambda x: alpha + beta * (x - xbar),
        start=xmin,
        end=xmax,
        color="black",
        linewidth=1,
        alpha=0.2,
        ax=ax,
    )
```

<!-- #region tags=[] -->
### 4.42
<!-- #endregion -->

```python tags=[]
df = pd.read_csv("data/Howell1.csv", sep=";")
df2 = df[df["age"] >= 18]

xbar = df2["weight"].mean()
weight = np.asarray(df2["weight"])

with pm.Model() as model:
    a = pm.Normal("a", mu=178, sigma=20)
    b = pm.Lognormal("b", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", 0, 50)
    height = pm.Normal(
        "height", mu=a + b * (weight - xbar), sigma=sigma, observed=df2["height"]
    )
    m43 = quap()
```

<!-- #region tags=[] -->
### 4.43
<!-- #endregion -->

```python tags=[]
with pm.Model() as model:
    a = pm.Normal("a", mu=178, sigma=20)
    log_b = pm.Normal("log_b", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", 0, 50)
    height = pm.Normal(
        "height",
        mu=a + log_b.exp() * (weight - xbar),
        sigma=sigma,
        observed=df2["height"],
    )
    m43b = quap()
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
ax = plot(data=df2, x="weight", y="height", plot_fn=sns.scatterplot)

post = extract_samples(m43, n=10_000, random_state=random_state)
a_map = post["a"].mean()
b_map = post["b"].mean()
ax = curve(
    lambda x: a_map + b_map * (x - xbar),
    start=df2["weight"].min(),
    end=df2["weight"].max(),
    xlabel="weight",
    ylabel="height",
    ax=ax,
    color="tab:orange",
    linewidth=2,
)
```

<!-- #region tags=[] -->
### 4.47
<!-- #endregion -->

```python tags=[]
post = extract_samples(m43, n=10_000, random_state=random_state)
post.head(5)
```

<!-- #region tags=[] -->
### 4.48
<!-- #endregion -->

```python tags=[]
n = 10
dfn = df2[:n]

x = np.asarray(dfn["weight"] - dfn["weight"].mean())

with pm.Model() as model:
    a = pm.Normal("a", mu=178, sigma=20)
    b = pm.Lognormal("b", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", 0, 50)
    height = pm.Normal("height", mu=a + b * x, sigma=sigma, observed=dfn["height"])
    mn = quap()
```

<!-- #region tags=[] -->
### 4.49
<!-- #endregion -->

```python tags=[]
post = extract_samples(mn, n=20, random_state=random_state)
ax = plot(data=dfn, x="weight", y="height", plot_fn=sns.scatterplot, title=f"n = {n}")
for i in range(n):
    plot(
        x=xs,
        y=post["a"][i] + post["b"][i] * (xs - xbar),
        plot_fn=sns.lineplot,
        ax=ax,
        color="black",
        alpha=0.2,
    )
```

<!-- #region tags=[] -->
### 4.50
<!-- #endregion -->

```python tags=[]
post = extract_samples(m43, n=100_000, random_state=random_state)
mu_at_50 = post["a"] + post["b"] * (50 - xbar)
```

<!-- #region tags=[] -->
### 4.51
<!-- #endregion -->

```python tags=[]
ax = dens(mu_at_50, xlabel="mu|weight=50")
```

<!-- #region tags=[] -->
### 4.52
<!-- #endregion -->

```python tags=[]
pi(mu_at_50, prob=0.89)
```

<!-- #region tags=[] -->
### 4.53
<!-- #endregion -->

```python tags=[]
mu = link(
    m43, {"mu": lambda df: df["a"] + df["b"] * (df["weight"] - xbar)}, df2[["weight"]]
)
```

<!-- #region tags=[] -->
### 4.54
<!-- #endregion -->

```python tags=[]
weight_seq = np.linspace(25, 70, 46)
mu = link(
    m43,
    {"mu": lambda df: df["a"] + df["b"] * (df["weight"] - xbar)},
    pd.DataFrame({"weight": weight_seq}),
)
mu
```

<!-- #region tags=[] -->
### 4.55
<!-- #endregion -->

```python tags=[]
ax = plot(
    x="weight",
    y="mu",
    data=mu,
    plot_fn=sns.scatterplot,
    color="tab:orange",
    linewidth=0.5,
    alpha=0.3,
)
ax = sns.regplot(x="weight", y="height", data=df2, scatter=False, ax=ax, ci=None)
```

<!-- #region tags=[] -->
### 4.56
<!-- #endregion -->

```python tags=[]
mu_mean = mu.groupby("weight").mean()["mu"]
mu_quantile = mu.groupby("weight").quantile([0.05, 0.94])["mu"].unstack()
mu_quantile.columns = ["lower", "upper"]
```

<!-- #region tags=[] -->
### 4.57
<!-- #endregion -->

```python tags=[]
ax = plot(data=df2, x="weight", y="height", plot_fn=sns.scatterplot)
plot(x=mu_mean.index, y=mu_mean.values, plot_fn=sns.lineplot, ax=ax, color="tab:red")
p = plt.fill_between(weight_seq, mu_quantile["lower"], mu_quantile["upper"], alpha=0.2)
```

<!-- #region tags=[] -->
### 4.58
<!-- #endregion -->

```python tags=[]
post = extract_samples(m43, n=10_000)
weight_seq = np.linspace(25, 70, 46)
df = post.merge(pd.DataFrame({"weight": weight_seq}), how="cross")
df["mu"] = df["a"] + df["b"] * (df["weight"] - xbar)
mu_mean = df.groupby("weight").mean()["mu"]
mu_quantile = df.groupby("weight").quantile([0.05, 0.94])["mu"].unstack()
mu_quantile.columns = ["lower", "upper"]
```

<!-- #region tags=[] -->
### 4.59
<!-- #endregion -->

```python tags=[]
df = link(
    m43,
    {"mu": lambda df: df["a"] + df["b"] * (df["weight"] - xbar)},
    pd.DataFrame({"weight": weight_seq}),
)
df["height"] = stats.norm.rvs(df["mu"], df["sigma"])
df["height"]
```

<!-- #region tags=[] -->
### 4.60 FIXME
<!-- #endregion -->

```python tags=[]
# mu_quantile = df.groupby('weight').quantile([0.05, 0.94])['mu'].unstack()
# mu_quantile.columns = ['lower', 'upper']

mu_hpdi = df.groupby("weight").agg(lambda x: hdpi(np.asarray(x)))

# height_pi = df.groupby('weight')['height'].quantile((0.05, 0.94)).unstack()
# height_pi.columns = ['lower', 'upper']
```

<!-- #region tags=[] -->
### 4.61 FIXME
<!-- #endregion -->

```python tags=[]
# plot raw data
ax = plot(data=df2, x="weight", y="height", plot_fn=sns.scatterplot, alpha=0.8)

# draw MAP line
plot(x=weight_seq, y=mu_mean, plot_fn=sns.lineplot, ax=ax, linewidth=2, color="tab:red")

# draw HDPI region for line
plt.fill_between(
    weight_seq, mu_quantile["lower"], mu_quantile["upper"], alpha=0.2, color=["tab:red"]
)

# draw PI region for simulated heights
p = plt.fill_between(
    weight_seq, height_pi["lower"], height_pi["upper"], alpha=0.2, color=["gray"]
)
```

<!-- #region tags=[] -->
### 4.62 FIXME
<!-- #endregion -->

```python tags=[]
data = pd.DataFrame({"weight": np.linspace(25, 70, 46)})
sim_height = sim(m43, lambda df: df["a"] + df["b"] * (df["weight"] - xbar), data)
pi = sim_height.groupby("weight").quantile([0.05, 0.94]).unstack()
pi.columns = ["low", "high"]

plt.fill_between(pi.index, pi["low"], pi["high"], alpha=0.2, color=["tab:green"])
```

```python
sim_height = sim
```

<!-- #region tags=[] -->
### 4.63
<!-- #endregion -->

```python tags=[]
# we don't have `sim` in "unthinking", so this is just a repeat of 4.59 and 4.60

df = link(
    m43,
    {"mu": lambda df: df["a"] + df["b"] * (df["weight"] - xbar)},
    pd.DataFrame({"weight": weight_seq}),
)
df["height"] = stats.norm.rvs(df["mu"], df["sigma"])

height_pi = df.groupby("weight")["height"].quantile((0.05, 0.94)).unstack()
height_pi.columns = ["lower", "upper"]
```

<!-- #region tags=[] -->
### 4.64
<!-- #endregion -->

```python tags=[]
df = pd.read_csv("data/Howell1.csv", sep=";")
```

<!-- #region tags=[] -->
### 4.65
<!-- #endregion -->

```python tags=[]
weight_s = (df["weight"] - df["weight"].mean()) / df["weight"].std()
weight_s2 = weight_s ** 2

with pm.Model() as model:
    a = pm.Normal("a", mu=178, sigma=20)
    b1 = pm.Lognormal("b1", mu=0, sigma=1)
    b2 = pm.Normal("b2", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", 0, 50)
    height = pm.Normal(
        "height",
        mu=a + b1 * weight_s + b2 * weight_s2,
        sigma=sigma,
        observed=df["height"],
    )
    m45 = quap()
```

<!-- #region tags=[] -->
### 4.66
<!-- #endregion -->

```python tags=[]
precis(m45)
```

<!-- #region tags=[] -->
### 4.67
<!-- #endregion -->

```python tags=[]
weight_seq = np.linspace(-2.2, 2, 30)
pred_df = pd.DataFrame({"weight_s": weight_seq, "weight_s2": weight_seq**2})
df = link(m45, {'mu': lambda df: df['a'] + df['b1'] * df['weight_s'] + df['b2'] * df['weight_s2']}, pred_df)

mu_mean = df.groupby('weight_s')['mu'].mean()
mu_pi = df.groupby("weight_s").quantile([0.05, 0.94])["mu"].unstack()
mu_pi.columns = ["lower", "upper"]

df["sim_height"] = stats.norm.rvs(df["mu"], df["sigma"])
height_pi = df.groupby('weight_s').quantile([0.05, 0.94])["sim_height"].unstack()
height_pi.columns = ["lower", "upper"]
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
ax = dens(samples, xlabel="y", title="prior predictive distribution")
```

<!-- #region tags=[] -->
### 4M2.
<!-- #endregion -->

```python tags=[]
fake_data = stats.norm.rvs(size=500)

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.Exponential("sigma", lam=1)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=fake_data)
    result = quap()
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
df = pd.read_csv("data/Howell1.csv", sep=";")
df2 = df[df["age"] >= 18]

x = np.asarray(df2["weight"])

with pm.Model() as model:
    a = pm.Normal("a", mu=178, sigma=20)
    b = pm.Lognormal("b", mu=0, sigma=1)
    sigma = pm.Uniform("sigma", 0, 50)
    height = pm.Normal("height", mu=a + b * x, sigma=sigma, observed=df2["height"])
    m_uncentered = quap()


uncentered_samples = extract_samples(m_uncentered, n=10_000)
m_uncentered.cov
```

```python tags=[]
m43_samples = extract_samples(m43, n=10_000)

weight_seq = np.linspace(35, 70, 46)
a = np.asarray(m43_samples["a"])
b = np.asarray(m43_samples["b"])

xbar = df2["weight"].mean()

m43_best_fit = (a + b * (weight_seq - xbar).reshape(-1, 1)).mean(axis=1)


a = np.asarray(uncentered_samples["a"])
b = np.asarray(uncentered_samples["b"])
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
