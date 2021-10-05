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

# Chapter 2: Small Worlds and Large Worlds

```python
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

## Code


### 2.1

```python
ways = np.array([0, 3, 8, 9, 0])
ways / sum(ways)
```

### 2.2

```python
stats.binom.pmf(6, 9, 0.5)
```

### 2.3

```python
# define grid
p_grid = np.linspace(0, 1, 20)

# define prior
prior = np.repeat(1, 20)

# compute likelihood at each value in grid
likelihood = stats.binom.pmf(6, 9, p_grid)

# compute product of likelihood and prior
unstd_posterior = likelihood * prior

# standardize the posterior, so it sums to 1
posterior = unstd_posterior / sum(unstd_posterior)
```

### 2.4

```python
ax = plot(
    x=p_grid,
    y=posterior,
    marker="o",
    xlabel="probability of water",
    ylabel="posterior probability",
)
```

```python
def plot_posterior(w, l, num_points, prior_fn, ax=None, title=None):
    p_grid = np.linspace(0, 1, num_points)
    prior = np.vectorize(prior_fn)(p_grid)

    likelihood = stats.binom.pmf(w, w + l, p_grid)
    unstd_posterior = likelihood * prior
    posterior = unstd_posterior / sum(unstd_posterior)

    plot(
        x=p_grid,
        y=posterior,
        marker="o",
        ax=ax,
        xlabel="probability of water",
        ylabel="posterior probability",
        title=title,
    )


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

uninformative_prior = lambda x: 1
plot_posterior(6, 3, 5, uninformative_prior, ax=axes[0], title="5 points")
plot_posterior(6, 3, 20, uninformative_prior, ax=axes[1], title="20 points")
plot_posterior(6, 3, 100, uninformative_prior, axes[2], "100 points")

fig.tight_layout()
```

<!-- #region tags=[] -->
### 2.5
<!-- #endregion -->

```python tags=[]
abundant_water_prior = lambda x: 0 if x < 0.5 else 1
laplace_prior = lambda x: np.exp(-5 * np.abs(x - 0.5))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_posterior(6, 3, 50, abundant_water_prior, axes[0], "p > 0.5 prior")
plot_posterior(6, 3, 50, laplace_prior, axes[1], "Laplace prior")
```

<!-- #region tags=[] -->
### 2.6
<!-- #endregion -->

```python tags=[]
with pm.Model() as model:
    p = pm.Uniform("p", 0, 1)
    w = pm.Binomial("w", n=9, p=p, observed=6)
    quap_result = quap()
precis(quap_result)
```

### 2.7

```python
ax = plt.gca()
ax.set(xlabel="probability of water", ylabel="density")

# analytical calculation
w = 6
l = 3
curve(lambda x: stats.beta.pdf(x, w + 1, l + 1), start=0, end=1, ax=ax, label="exact solution")

# quadratic approximation
curve(lambda x: stats.norm.pdf(x, 0.67, 0.16), start=0, end=1, ax=ax, label="quadratic approximation")

legend = plt.legend(loc="upper left")
```

### 2.8

```python
n_samples = 1000
p = np.repeat(np.nan, n_samples)
p[0] = 0.5
w = 6
l = 3

for i in range(1, n_samples):
    p_new = np.abs(stats.norm.rvs(p[i - 1], 0.1, random_state=random_state))
    if p_new > 1:
        p_new = 2 - p_new
    q0 = stats.binom.pmf(w, w + l, p[i - 1])
    q1 = stats.binom.pmf(w, w + l, p_new)
    p[i] = p_new if stats.uniform.rvs(random_state=random_state) < q1 / q0 else p[i - 1]
```

### 2.9

```python
ax = plt.gca()
dens(p, ax=ax, xlabel="probability of water", ylabel="density", label="MCMC")
curve(
    lambda x: stats.beta.pdf(x, w + 1, l + 1), start=0, end=1, ax=ax, color="tab:orange", label="exact solution"
)
legend = plt.legend(loc="upper left")
```

## Practice


### 2E1.

(2) and (4). (2) is immediate, while (4) follows from the definition of conditional probability.


### 2E2.

(3), the probability that it is Monday, given that it is raining.


### 2E3.

(1) and (4). (1) is immediate and (4) follows from applying Bayes' theorem.


### 2E4.

To say that the probability of water is 0.7 is to say that 70% of the globe is covered with water from the perspective of our observer with limited knowledge.


### 2M1.

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

plot_posterior(3, 0, 40, uninformative_prior, axes[0], "W, W, W")
plot_posterior(3, 1, 40, uninformative_prior, axes[1], "W, W, W, L")
plot_posterior(5, 2, 40, uninformative_prior, axes[2], "L, W, W, L, W, W, W")

fig.tight_layout()
```

### 2M2.

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

plot_posterior(3, 0, 40, abundant_water_prior, axes[0], "W, W, W")
plot_posterior(3, 1, 40, abundant_water_prior, axes[1], "W, W, W, L")
plot_posterior(5, 2, 40, abundant_water_prior, axes[2], "L, W, W, L, W, W, W")

fig.tight_layout()
```

### 2M3.

$$
P(\text{earth}|\text{land})
= \frac{P(\text{land}|\text{earth})P(\text{earth})}{P(\text{land})}
= \frac{0.3 \cdot 0.5}{0.5 \cdot 0.3 + 0.5 \cdot 1} = 0.23
$$


### 2M4.

Since a black side is showing up, it's not possible we have drawn the W/W card. If we have drawn the B/W card, there is only one way we could be showing black, whereas if we have drawn the B/B card, there are two.

Therefore $P(\text{B/B}|{B}) = 2/3$.


### 2M5.

Again, since a black side is showing up, we could not have drawn the W/W card. If we have drawn the B/W card, there is only one way we could be showing black, whereas there are 4 ways we could have drawn a B/B card, since there are two cards with two black sides each.

Therefore $P(\text{B/B}|\text{B}) = 4/5$.

<!-- #region -->
### 2M6.


Since a black side is showing, we could not have drawn the W/W card. There are two ways we have drawn the B/W card, and for each there is only one way we could be showing a black side, whereas there is only one way we could have drawn a B/B card, but two ways to be showing a black side, since there are two black sides. 

Therefore 
$P(\text{B/B}|\text{B}) = 1/2$.
<!-- #endregion -->

### 2M7.

Since a black side is showing, we could not have originally drawn the W/W card.

If we have first drawn the B/W card, we could have done so only one way, with the black side up, and the next card we drew must have been the W/W card, which we can draw two ways (either side up).

If instead we have drawn the B/B card, we could have done so two ways (either side up), and the next card we drew could either have been B/W (one way, white side up) or W/W (two ways, either side up).

Therefore $P(\text{B/B}|\text{first card B, second card W}) = 6/8 = 3/4$.


### 2H1.
$$P(\text{twins}) = P(\text{twins}|\text{A})P(\text{A}) + P(\text{twins}|\text{B})P(\text{B}) = 0.1 \cdot 0.5 + 0.2 \cdot 0.5 = 0.15$$

$$P(\text{A}|\text{twins}) = \frac{P(\text{twins}|\text{A})P(\text{A})}{P(\text{twins})} = \frac{0.1 \cdot 0.5}{0.15} = \frac{1}{3}$$

$$P(\text{B}|\text{twins}) = 1 - P(\text{A}|\text{twins}) = \frac{2}{3} $$

$$P(\text{twins again}|\text{twins}) = P(\text{twins}|\text{A})P(\text{A}|\text{twins}) + P(\text{twins}|\text{B})P(\text{B}|\text{twins}) = \frac{1}{30} + \frac{4}{30} = \frac{1}{6}$$


### 2H2.

We computed this as part of the previous problem, $P(\text{A}|\text{twins}) = 1/3$.


### 2H3.

$$ P(\text{A}|\text{twins first, single}) = \frac{P(\text{single}|\text{A})P(\text{A}|\text{twins first})}{P(\text{single}|\text{twins first})} = \frac{9}{10} \frac{1}{3} \frac{6}{5} = \frac{9}{25} $$


### 2H4.

$$P(\text{A}|\text{A positive}) = \frac{P(\text{A positive}|\text{A})P(\text{A})}{P(\text{A positive})} = \frac{\frac{8}{10} \frac{1}{2}}{\frac{8}{10} \frac{1}{2} + \frac{7}{20} \frac{1}{2}} = \frac{16}{23}$$

$$P(\text{A}|\text{A positive, birth data}) = \frac{P(\text{A positive}|\text{A})P(\text{A}|\text{birth data})}{P(\text{A positive}|\text{birth data})} = \frac{\frac{4}{5} \frac{9}{25}}{\frac{4}{5} \frac{9}{25} + \frac{7}{20} \frac{16}{25}} = \frac{9}{16}$$
