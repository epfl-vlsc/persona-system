<!-- This file is machine generated: DO NOT EDIT! -->

# Statistical distributions (contrib)
[TOC]

Classes representing statistical distributions.  Ops for working with them.

## Classes for statistical distributions.

Classes that represent batches of statistical distributions.  Each class is
initialized with parameters that define the distributions.

### Univariate (scalar) distributions

- - -

### `class tf.contrib.distributions.Gaussian` {#Gaussian}

The scalar Gaussian distribution with mean and stddev parameters mu, sigma.

#### Mathematical details

The PDF of this distribution is:

```f(x) = sqrt(1/(2*pi*sigma^2)) exp(-(x-mu)^2/(2*sigma^2))```

#### Examples

Examples of initialization of one or a batch of distributions.

```python
# Define a single scalar Gaussian distribution.
dist = tf.contrib.Gaussian(mu=0, sigma=3)

# Evaluate the cdf at 1, returning a scalar.
dist.cdf(1)

# Define a batch of two scalar valued Gaussians.
# The first has mean 1 and standard deviation 11, the second 2 and 22.
dist = tf.contrib.distributions.Gaussian(mu=[1, 2.], sigma=[11, 22.])

# Evaluate the pdf of the first distribution on 0, and the second on 1.5,
# returning a length two tensor.
dist.pdf([0, 1.5])

# Get 3 samples, returning a 3 x 2 tensor.
dist.sample(3)
```

Arguments are broadcast when possible.

```python
# Define a batch of two scalar valued Gaussians.
# Both have mean 1, but different standard deviations.
dist = tf.contrib.distributions.Gaussian(mu=1, sigma=[11, 22.])

# Evaluate the pdf of both distributions on the same point, 3.0,
# returning a length 2 tensor.
dist.pdf(3.0)
```
- - -

#### `tf.contrib.distributions.Gaussian.__init__(mu, sigma, name=None)` {#Gaussian.__init__}

Construct Gaussian distributions with mean and stddev `mu` and `sigma`.

The parameters `mu` and `sigma` must be shaped in a way that supports
broadcasting (e.g. `mu + sigma` is a valid operation).

##### Args:


*  <b>`mu`</b>: `float` or `double` tensor, the means of the distribution(s).
*  <b>`sigma`</b>: `float` or `double` tensor, the stddevs of the distribution(s).
    sigma must contain only positive values.
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`TypeError`</b>: if mu and sigma are different dtypes.


- - -

#### `tf.contrib.distributions.Gaussian.cdf(x, name=None)` {#Gaussian.cdf}

CDF of observations in `x` under these Gaussian distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: tensor of dtype `dtype`, the CDFs of `x`.


- - -

#### `tf.contrib.distributions.Gaussian.dtype` {#Gaussian.dtype}




- - -

#### `tf.contrib.distributions.Gaussian.entropy(name=None)` {#Gaussian.entropy}

The entropy of Gaussian distribution(s).

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropy.


- - -

#### `tf.contrib.distributions.Gaussian.log_cdf(x, name=None)` {#Gaussian.log_cdf}

Log CDF of observations `x` under these Gaussian distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_cdf`</b>: tensor of dtype `dtype`, the log-CDFs of `x`.


- - -

#### `tf.contrib.distributions.Gaussian.log_pdf(x, name=None)` {#Gaussian.log_pdf}

Log pdf of observations in `x` under these Gaussian distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_pdf`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.


- - -

#### `tf.contrib.distributions.Gaussian.mean` {#Gaussian.mean}




- - -

#### `tf.contrib.distributions.Gaussian.mu` {#Gaussian.mu}




- - -

#### `tf.contrib.distributions.Gaussian.pdf(x, name=None)` {#Gaussian.pdf}

The PDF of observations in `x` under these Gaussian distribution(s).

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`pdf`</b>: tensor of dtype `dtype`, the pdf values of `x`.


- - -

#### `tf.contrib.distributions.Gaussian.sample(n, seed=None, name=None)` {#Gaussian.sample}

Sample `n` observations from the Gaussian Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the hyperparameters.


- - -

#### `tf.contrib.distributions.Gaussian.sigma` {#Gaussian.sigma}






### Multivariate distributions

- - -

### `class tf.contrib.distributions.MultivariateNormal` {#MultivariateNormal}

The Multivariate Normal distribution on `R^k`.

The distribution has mean and covariance parameters mu (1-D), sigma (2-D),
or alternatively mean `mu` and factored covariance (cholesky decomposed
`sigma`) called `sigma_chol`.

#### Mathematical details

The PDF of this distribution is:

```
f(x) = (2*pi)^(-k/2) |det(sigma)|^(-1/2) exp(-1/2*(x-mu)^*.sigma^{-1}.(x-mu))
```

where `.` denotes the inner product on `R^k` and `^*` denotes transpose.

Alternatively, if `sigma` is positive definite, it can be represented in terms
of its lower triangular cholesky factorization

```sigma = sigma_chol . sigma_chol^*```

and the pdf above allows simpler computation:

```
|det(sigma)| = reduce_prod(diag(sigma_chol))^2
x_whitened = sigma^{-1/2} . (x - mu) = tri_solve(sigma_chol, x - mu)
(x-mu)^* .sigma^{-1} . (x-mu) = x_whitened^* . x_whitened
```

where `tri_solve()` solves a triangular system of equations.

#### Examples

A single multi-variate Gaussian distribution is defined by a vector of means
of length `k`, and a covariance matrix of shape `k x k`.

Extra leading dimensions, if provided, allow for batches.

```python
# Initialize a single 3-variate Gaussian with diagonal covariance.
mu = [1, 2, 3]
sigma = [[1, 0, 0], [0, 3, 0], [0, 0, 2]]
dist = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)

# Evaluate this on an observation in R^3, returning a scalar.
dist.pdf([-1, 0, 1])

# Initialize a batch of two 3-variate Gaussians.
mu = [[1, 2, 3], [11, 22, 33]]
sigma = ...  # shape 2 x 3 x 3
dist = tf.contrib.distributions.MultivariateNormal(mu=mu, sigma=sigma)

# Evaluate this on a two observations, each in R^3, returning a length two
# tensor.
x = [[-1, 0, 1], [-11, 0, 11]]  # Shape 2 x 3.
dist.pdf(x)
```
- - -

#### `tf.contrib.distributions.MultivariateNormal.__init__(mu, sigma=None, sigma_chol=None, name=None)` {#MultivariateNormal.__init__}

Multivariate Normal distributions on `R^k`.

User must provide means `mu`, which are tensors of rank `N+1` (`N >= 0`)
with the last dimension having length `k`.

User must provide exactly one of `sigma` (the covariance matrices) or
`sigma_chol` (the cholesky decompositions of the covariance matrices).
`sigma` or `sigma_chol` must be of rank `N+2`.  The last two dimensions
must both have length `k`.  The first `N` dimensions correspond to batch
indices.

If `sigma_chol` is not provided, the batch cholesky factorization of `sigma`
is calculated for you.

The shapes of `mu` and `sigma` must match for the first `N` dimensions.

Regardless of which parameter is provided, the covariance matrices must all
be **positive definite** (an error is raised if one of them is not).

##### Args:


*  <b>`mu`</b>: (N+1)-D.  `float` or `double` tensor, the means of the distributions.
*  <b>`sigma`</b>: (N+2)-D.  (optional) `float` or `double` tensor, the covariances
    of the distribution(s).  The first `N+1` dimensions must match
    those of `mu`.  Must be batch-positive-definite.
*  <b>`sigma_chol`</b>: (N+2)-D.  (optional) `float` or `double` tensor, a
    lower-triangular factorization of `sigma`
    (`sigma = sigma_chol . sigma_chol^*`).  The first `N+1` dimensions
    must match those of `mu`.  The tensor itself need not be batch
    lower triangular: we ignore the upper triangular part.  However,
    the batch diagonals must be positive (i.e., sigma_chol must be
    batch-positive-definite).
*  <b>`name`</b>: The name to give Ops created by the initializer.

##### Raises:


*  <b>`ValueError`</b>: if neither sigma nor sigma_chol is provided.
*  <b>`TypeError`</b>: if mu and sigma (resp. sigma_chol) are different dtypes.


- - -

#### `tf.contrib.distributions.MultivariateNormal.dtype` {#MultivariateNormal.dtype}




- - -

#### `tf.contrib.distributions.MultivariateNormal.entropy(name=None)` {#MultivariateNormal.entropy}

The entropies of these Multivariate Normals.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`entropy`</b>: tensor of dtype `dtype`, the entropies.


- - -

#### `tf.contrib.distributions.MultivariateNormal.log_pdf(x, name=None)` {#MultivariateNormal.log_pdf}

Log pdf of observations `x` given these Multivariate Normals.

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_pdf`</b>: tensor of dtype `dtype`, the log-PDFs of `x`.


- - -

#### `tf.contrib.distributions.MultivariateNormal.mean` {#MultivariateNormal.mean}




- - -

#### `tf.contrib.distributions.MultivariateNormal.mu` {#MultivariateNormal.mu}




- - -

#### `tf.contrib.distributions.MultivariateNormal.pdf(x, name=None)` {#MultivariateNormal.pdf}

The PDF of observations `x` under these Multivariate Normals.

##### Args:


*  <b>`x`</b>: tensor of dtype `dtype`, must be broadcastable with `mu` and `sigma`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`pdf`</b>: tensor of dtype `dtype`, the pdf values of `x`.


- - -

#### `tf.contrib.distributions.MultivariateNormal.sample(n, seed=None, name=None)` {#MultivariateNormal.sample}

Sample `n` observations from the Multivariate Normal Distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the hyperparameters.


- - -

#### `tf.contrib.distributions.MultivariateNormal.sigma` {#MultivariateNormal.sigma}




- - -

#### `tf.contrib.distributions.MultivariateNormal.sigma_det` {#MultivariateNormal.sigma_det}





- - -

### `class tf.contrib.distributions.DirichletMultinomial` {#DirichletMultinomial}

DirichletMultinomial mixture distribution.

This distribution is parameterized by a vector `alpha` of concentration
parameters for `k` classes.

#### Mathematical details

The Dirichlet Multinomial is a distribution over k-class count data, meaning
for each k-tuple of non-negative integer `counts = [c_1,...,c_k]`, we have a
probability of these draws being made from the distribution.  The distribution
has hyperparameters `alpha = (alpha_1,...,alpha_k)`, and probability mass
function (pmf):

```pmf(counts) = C! / (c_1!...c_k!) * Beta(alpha + c) / Beta(alpha)```

where above `C = sum_j c_j`, `N!` is `N` factorial, and
`Beta(x) = prod_j Gamma(x_j) / Gamma(sum_j x_j)` is the multivariate beta
function.

This is a mixture distribution in that `N` samples can be produced by:
  1. Choose class probabilities `p = (p_1,...,p_k) ~ Dir(alpha)`
  2. Draw integers `m = (m_1,...,m_k) ~ Multinomial(p, N)`

This class provides methods to create indexed batches of Dirichlet
Multinomial distributions.  If the provided `alpha` is rank 2 or higher, for
every fixed set of leading dimensions, the last dimension represents one
single Dirichlet Multinomial distribution.  When calling distribution
functions (e.g. `dist.pdf(counts)`), `alpha` and `counts` are broadcast to the
same shape (if possible).  In all cases, the last dimension of alpha/counts
represents single Dirichlet Multinomial distributions.

#### Examples

```python
alpha = [1, 2, 3]
dist = DirichletMultinomial(alpha)
```

Creates a 3-class distribution, with the 3rd class is most likely to be drawn.
The distribution functions can be evaluated on counts.

```python
# counts same shape as alpha.
counts = [0, 2, 0]
dist.pdf(counts)  # Shape []

# alpha will be broadcast to [[1, 2, 3], [1, 2, 3]] to match counts.
counts = [[11, 22, 33], [44, 55, 66]]
dist.pdf(counts)  # Shape [2]

# alpha will be broadcast to shape [5, 7, 3] to match counts.
counts = [[...]]  # Shape [5, 7, 3]
dist.pdf(counts)  # Shape [5, 7]
```

Creates a 2-batch of 3-class distributions.

```python
alpha = [[1, 2, 3], [4, 5, 6]]  # Shape [2, 3]
dist = DirichletMultinomial(alpha)

# counts will be broadcast to [[11, 22, 33], [11, 22, 33]] to match alpha.
counts = [11, 22, 33]
dist.pdf(counts)  # Shape [2]
```
- - -

#### `tf.contrib.distributions.DirichletMultinomial.__init__(alpha)` {#DirichletMultinomial.__init__}

Initialize a batch of DirichletMultinomial distributions.

##### Args:


*  <b>`alpha`</b>: Shape `[N1,..., Nn, k]` positive `float` or `double` tensor with
    `n >= 0`.  Defines this as a batch of `N1 x ... x Nn` different `k`
    class Dirichlet multinomial distributions.


*  <b>`Examples`</b>: 

```python
# Define 1-batch of 2-class Dirichlet multinomial distribution,
# also known as a beta-binomial.
dist = DirichletMultinomial([1.1, 2.0])

# Define a 2-batch of 3-class distributions.
dist = DirichletMultinomial([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
```


- - -

#### `tf.contrib.distributions.DirichletMultinomial.alpha` {#DirichletMultinomial.alpha}

Parameters defining this distribution.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.cdf(x)` {#DirichletMultinomial.cdf}




- - -

#### `tf.contrib.distributions.DirichletMultinomial.dtype` {#DirichletMultinomial.dtype}




- - -

#### `tf.contrib.distributions.DirichletMultinomial.log_cdf(x)` {#DirichletMultinomial.log_cdf}




- - -

#### `tf.contrib.distributions.DirichletMultinomial.log_pmf(counts, name=None)` {#DirichletMultinomial.log_pmf}

`Log(P[counts])`, computed for every batch member.

For each batch of counts `[c_1,...,c_k]`, `P[counts]` is the probability
that after sampling `sum_j c_j` draws from this Dirichlet Multinomial
distribution, the number of draws falling in class `j` is `c_j`.  Note that
different sequences of draws can result in the same counts, thus the
probability includes a combinatorial coefficient.

##### Args:


*  <b>`counts`</b>: Non-negative `float`, `double`, or `int` tensor whose shape can
    be broadcast with `self.alpha`.  For fixed leading dimensions, the last
    dimension represents counts for the corresponding Dirichlet Multinomial
    distribution in `self.alpha`.
*  <b>`name`</b>: Name to give this Op, defaults to "log_pmf".

##### Returns:

  Log probabilities for each record, shape `[N1,...,Nn]`.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.mean` {#DirichletMultinomial.mean}

Class means for every batch member.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.num_classes` {#DirichletMultinomial.num_classes}

Tensor providing number of classes in each batch member.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.pmf(counts, name=None)` {#DirichletMultinomial.pmf}

`P[counts]`, computed for every batch member.

For each batch of counts `[c_1,...,c_k]`, `P[counts]` is the probability
that after sampling `sum_j c_j` draws from this Dirichlet Multinomial
distribution, the number of draws falling in class `j` is `c_j`.  Note that
different sequences of draws can result in the same counts, thus the
probability includes a combinatorial coefficient.

##### Args:


*  <b>`counts`</b>: Non-negative `float`, `double`, or `int` tensor whose shape can
    be broadcast with `self.alpha`.  For fixed leading dimensions, the last
    dimension represents counts for the corresponding Dirichlet Multinomial
    distribution in `self.alpha`.
*  <b>`name`</b>: Name to give this Op, defaults to "pmf".

##### Returns:

  Probabilities for each record, shape `[N1,...,Nn]`.




## Posterior inference with conjugate priors.

Functions that transform conjugate prior/likelihood pairs to distributions
representing the posterior or posterior predictive.

### Gaussian likelihood with conjugate prior.

- - -

### `tf.contrib.distributions.gaussian_conjugates_known_sigma_posterior(prior, sigma, s, n)` {#gaussian_conjugates_known_sigma_posterior}

Posterior Gaussian distribution with conjugate prior on the mean.

This model assumes that `n` observations (with sum `s`) come from a
Gaussian with unknown mean `mu` (described by the Gaussian `prior`)
and known variance `sigma^2`.  The "known sigma posterior" is
the distribution of the unknown `mu`.

Accepts a prior Gaussian distribution object, having parameters
`mu0` and `sigma0`, as well as known `sigma` values of the predictive
distribution(s) (also assumed Gaussian),
and statistical estimates `s` (the sum(s) of the observations) and
`n` (the number(s) of observations).

Returns a posterior (also Gaussian) distribution object, with parameters
`(mu', sigma'^2)`, where:

```
mu ~ N(mu', sigma'^2)
sigma'^2 = 1/(1/sigma0^2 + n/sigma^2),
mu' = (mu0/sigma0^2 + s/sigma^2) * sigma'^2.
```

Distribution parameters from `prior`, as well as `sigma`, `s`, and `n`.
will broadcast in the case of multidimensional sets of parameters.

##### Args:


*  <b>`prior`</b>: `Gaussian` object of type `dtype`:
    the prior distribution having parameters `(mu0, sigma0)`.
*  <b>`sigma`</b>: tensor of type `dtype`, taking values `sigma > 0`.
    The known stddev parameter(s).
*  <b>`s`</b>: Tensor of type `dtype`.  The sum(s) of observations.
*  <b>`n`</b>: Tensor of type `int`.  The number(s) of observations.

##### Returns:

  A new Gaussian posterior distribution object for the unknown observation
  mean `mu`.

##### Raises:


*  <b>`TypeError`</b>: if dtype of `s` does not match `dtype`, or `prior` is not a
    Gaussian object.


- - -

### `tf.contrib.distributions.gaussian_congugates_known_sigma_predictive(prior, sigma, s, n)` {#gaussian_congugates_known_sigma_predictive}

Posterior predictive Gaussian distribution w. conjugate prior on the mean.

This model assumes that `n` observations (with sum `s`) come from a
Gaussian with unknown mean `mu` (described by the Gaussian `prior`)
and known variance `sigma^2`.  The "known sigma predictive"
is the distribution of new observations, conditioned on the existing
observations and our prior.

Accepts a prior Gaussian distribution object, having parameters
`mu0` and `sigma0`, as well as known `sigma` values of the predictive
distribution(s) (also assumed Gaussian),
and statistical estimates `s` (the sum(s) of the observations) and
`n` (the number(s) of observations).

Calculates the Gaussian distribution(s) `p(x | sigma^2)`:

```
  p(x | sigma^2) = int N(x | mu, sigma^2) N(mu | prior.mu, prior.sigma^2) dmu
                 = N(x | prior.mu, 1/(sigma^2 + prior.sigma^2))
```

Returns the predictive posterior distribution object, with parameters
`(mu', sigma'^2)`, where:

```
sigma_n^2 = 1/(1/sigma0^2 + n/sigma^2),
mu' = (mu0/sigma0^2 + s/sigma^2) * sigma_n^2.
sigma'^2 = sigma_n^2 + sigma^2,
```

Distribution parameters from `prior`, as well as `sigma`, `s`, and `n`.
will broadcast in the case of multidimensional sets of parameters.

##### Args:


*  <b>`prior`</b>: `Gaussian` object of type `dtype`:
    the prior distribution having parameters `(mu0, sigma0)`.
*  <b>`sigma`</b>: tensor of type `dtype`, taking values `sigma > 0`.
    The known stddev parameter(s).
*  <b>`s`</b>: Tensor of type `dtype`.  The sum(s) of observations.
*  <b>`n`</b>: Tensor of type `int`.  The number(s) of observations.

##### Returns:

  A new Gaussian predictive distribution object.

##### Raises:


*  <b>`TypeError`</b>: if dtype of `s` does not match `dtype`, or `prior` is not a
    Gaussian object.


