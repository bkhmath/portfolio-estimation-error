# Where the Marchenko–Pastur law comes from

This note derives the law the repository depends on at the level of rigour of a
sketch: the steps are the real ones, but the concentration arguments are asserted
rather than proved. The thing worth noticing is that the two facts the code uses,
the density formula and the location of the bulk edges, are not separate inputs.
Both fall out of a single quadratic equation, and so does the reason inversion is
the dangerous operation.

Implementation: [`src/estimation_error/rmt.py`](../src/estimation_error/rmt.py).
Tests: [`tests/test_rmt.py`](../tests/test_rmt.py).

## Setting

Let $X$ be $N \times T$ with i.i.d. entries of mean zero and unit variance, and let

$$S = \frac{1}{T} X X^{\top}$$

be the sample covariance matrix, so that the true covariance is $I_N$. Write
$\lambda_1, \dots, \lambda_N$ for the eigenvalues of $S$ and

$$F_N(x) = \frac{1}{N} \#\{ i : \lambda_i \le x \}$$

for the empirical spectral distribution. We are interested in the limit
$N, T \to \infty$ with $N/T \to q \in (0, \infty)$. This is the regime where the
number of assets and the number of observations are *comparable*, which is where a
portfolio manager actually sits and where classical statistics gives no answer.

If the eigenvalues behaved themselves, $F_N$ would converge to a point mass at 1,
since that is the spectrum of the truth. It does not.

## Why the Stieltjes transform

Define, for $z \in \mathbb{C} \setminus \mathbb{R}$,

$$m_N(z) = \int \frac{dF_N(\lambda)}{\lambda - z} = \frac{1}{N} \operatorname{tr}(S - zI)^{-1} .$$

Three properties make it the right object.

1. **It determines the measure.** The density is recovered on the real axis by the
   Stieltjes–Perron inversion formula, $p(x) = \frac{1}{\pi}\lim_{\varepsilon \downarrow 0} \operatorname{Im}\, m(x + i\varepsilon)$.
2. **It is analytic** off the real axis, and normalised: $m(z) \sim -1/z$ as
   $|z| \to \infty$, because for large $|z|$ the integrand is $\approx -1/z$
   uniformly and $F_N$ is a probability measure. This normalisation is what will
   pick the branch of a square root later, so it is not a throwaway remark.
3. **It linearises rank-one updates**, via the Sherman–Morrison identity. Adding
   one observation to a sample covariance *is* a rank-one update, so the
   recursion in $T$ becomes tractable.

The third is the one that does the work here.

## The self-consistent equation

Write $G(z) = (S - zI)^{-1}$ for the resolvent and let $x_1, \dots, x_T \in \mathbb{R}^N$
be the columns of $X$, so $S = \frac{1}{T}\sum_t x_t x_t^{\top}$.

Start from the trivial identity $\frac{1}{N}\operatorname{tr}\big[(S - zI)G\big] = 1$, that is

$$\frac{1}{N} \operatorname{tr}(S G) - z\, m_N(z) = 1. \tag{1}$$

Everything reduces to evaluating the first term. Expanding $S$,

$$\frac{1}{N}\operatorname{tr}(SG) = \frac{1}{N}\sum_{t=1}^{T} \frac{1}{T}\, x_t^{\top} G\, x_t .$$

The obstruction is that $x_t$ appears inside $G$, so the two are not independent.
Remove it: let $S_{(t)} = S - \frac{1}{T}x_t x_t^{\top}$ be the sample covariance
computed without observation $t$, and $G_{(t)} = (S_{(t)} - zI)^{-1}$. Sherman–Morrison gives

$$x_t^{\top} G\, x_t = \frac{x_t^{\top} G_{(t)}\, x_t}{1 + \frac{1}{T} x_t^{\top} G_{(t)}\, x_t} .$$

Now $x_t$ *is* independent of $G_{(t)}$, and $\mathbb{E}[x_t x_t^{\top}] = I$, so the
quadratic form concentrates on its trace:

$$\frac{1}{T} x_t^{\top} G_{(t)}\, x_t \;\approx\; \frac{1}{T}\operatorname{tr} G_{(t)} \;\approx\; \frac{N}{T}\cdot\frac{1}{N}\operatorname{tr} G \;=\; q\, m_N(z).$$

(The first step is the concentration of a quadratic form about its mean; the second
is the fact that a rank-one perturbation moves a normalised trace by $O(1/N)$. These
are the two places where a real proof does its work.)

Substituting, each of the $T$ terms contributes $\dfrac{q\,m}{1 + q\,m}$, and since
$T/N = 1/q$,

$$\frac{1}{N}\operatorname{tr}(SG) \;\longrightarrow\; \frac{T}{N}\cdot\frac{q\,m}{1 + q\,m} \;=\; \frac{m}{1 + q\,m}.$$

Putting this into (1):

$$\frac{m}{1 + qm} - zm = 1 \qquad\Longleftrightarrow\qquad \boxed{\,q z\, m^2 + (z + q - 1)\, m + 1 = 0\,} \tag{2}$$

A quadratic, so the rest is algebra.

## Reading off the answer

Solving (2),

$$m(z) = \frac{-(z + q - 1) \pm \sqrt{(z+q-1)^2 - 4qz}}{2qz},$$

and the normalisation $m(z) \sim -1/z$ at infinity forces the $+$ branch with the
square root behaving like $+z$ for large $|z|$. (Expanding: the numerator tends to
$-2q$, so $m \to -2q/(2qz) = -1/z$ as required. The other root behaves like
$-1/(qz)\cdot z = $ a constant, which is not the transform of a probability measure.)

**The edges.** The discriminant is

$$D(z) = (z + q - 1)^2 - 4qz = z^2 - 2(1+q) z + (1-q)^2,$$

whose roots are

$$z = (1 + q) \pm 2\sqrt{q} = \big(1 \pm \sqrt{q}\big)^2 =: \lambda_{\pm}.$$

So $D(z) = (z - \lambda_-)(z - \lambda_+)$. The support of the limiting measure is
where $m$ fails to be real, that is where the square root is imaginary, and that is
$[\lambda_-, \lambda_+]$. So the bulk edges are the branch points of the quadratic,
not an additional modelling assumption. That is also why `mp_edges` can be tested
by checking that each edge annihilates the discriminant, instead of comparing it
against the formula it implements.

**The density.** On $x \in (\lambda_-, \lambda_+)$ we have $D(x) < 0$, so
$\sqrt{D(x)} = i\sqrt{-D(x)}$ and

$$p(x) = \frac{1}{\pi}\operatorname{Im} m(x + i0^+) = \frac{\sqrt{-D(x)}}{2\pi q x} = \frac{\sqrt{(\lambda_+ - x)(x - \lambda_-)}}{2 \pi q x}.$$

That is `mp_density`. The inverse square-root blow-up at each edge is the standard
soft-edge behaviour; the substitution $x = 1 + q - 2\sqrt{q}\cos\theta$ used in
`mp_cdf` removes it, because $(\lambda_+ - x)(x - \lambda_-) = 4q\sin^2\theta$ and
$dx = 2\sqrt q \sin\theta\, d\theta$, leaving the bounded integrand
$\frac{2}{\pi}\sin^2\theta / x(\theta)$.

**The atom.** If $q > 1$ then $N > T$ and $S$ has rank at most $T$, so at least
$N - T$ eigenvalues are exactly zero: the law carries an atom of mass
$1 - 1/q$ at the origin, and the density above integrates to $1/q$ rather than 1.

## Why this is a statement about inversion

Set $z \to 0$ in (2). The quadratic term drops and $(q-1)m + 1 = 0$, so

$$m(0) = \frac{1}{1 - q} \qquad (q < 1).$$

But $m(0) = \int \lambda^{-1} dF(\lambda) = \frac{1}{N}\operatorname{tr} S^{-1}$. The
true covariance is the identity, for which the same quantity is exactly 1. So

$$\frac{1}{N}\operatorname{tr} S^{-1} \;\longrightarrow\; \frac{1}{1-q}\cdot \frac{1}{N}\operatorname{tr}\Sigma^{-1},$$

the estimated precision matrix is inflated by $1/(1-q)$, and the inflation diverges
as $N$ approaches $T$. That is the whole problem in one line. The sample covariance
is an unbiased estimator of $\Sigma$, but its inverse is a badly biased estimator
of $\Sigma^{-1}$, since inversion is convex and Jensen applies. The
minimum-variance portfolio $w \propto S^{-1}\mathbf{1}$ is built entirely out of
that inflated object.

The factor $1/(1-q)$ is the asymptotic version of the exact finite-sample result
$\mathbb{E}[\text{realised}] / \sigma^2_{\min} = (T-2)/(T-N-1)$ that
[`backtest.realised_variance_ratio`](../src/estimation_error/backtest.py) implements
and that `tests/test_backtest.py` checks against Monte Carlo.

## Two caveats the code has to respect

**Eigenvalues are not an i.i.d. sample.** They repel: the probability of two
eigenvalues being close is suppressed, so the empirical distribution is far more
*regular* than $N$ i.i.d. draws from $p$ would be. Empirically the
Kolmogorov–Smirnov distance between $F_N$ and $F$ falls like $N^{-1}$ rather than
the $N^{-1/2}$ of an i.i.d. sample, which is what
`test_convergence_rate_is_the_rigidity_rate_not_the_iid_rate` asserts. In practice
this means a KS *statistic* is a perfectly good distance while a KS *p-value* is
meaningless here, so the tests use absolute thresholds.

**The edge is soft.** The largest eigenvalue does not sit exactly at $\lambda_+$;
it fluctuates around it on the scale $N^{-2/3}$ with a Tracy–Widom limiting law.
So a handful of eigenvalues leaking past the edge at finite $N$ is expected, not
evidence of signal, which is why `test_pure_noise_has_no_signal_eigenvalues`
tolerates a small count rather than demanding zero.

## What gets used downstream

| Result | Used by |
|---|---|
| Edges $(1\pm\sqrt q)^2$ | `clipped_covariance`, as the threshold separating noise from signal |
| Density $p(x)$ | validation against simulated spectra; the spectrum figures |
| $\frac1N\operatorname{tr}S^{-1}\to\frac{1}{1-q}$ | why the project is about inversion at all |
| Condition number $\left(\frac{1+\sqrt q}{1-\sqrt q}\right)^2$ | the conditioning figure |
| The transform $m(z)$ itself | `nonlinear_shrinkage`, which estimates it from data by kernel smoothing rather than assuming the population spectrum is the identity |

## References

- Marchenko, V. A. and Pastur, L. A. (1967). Distribution of eigenvalues for some
  sets of random matrices. *Mat. Sb.*
- Bai, Z. and Silverstein, J. (2010). *Spectral Analysis of Large Dimensional
  Random Matrices.* The leave-one-out argument above, done properly.
- Laloux, L., Cizeau, P., Bouchaud, J.-P. and Potters, M. (1999). Noise dressing
  of financial correlation matrices. *Phys. Rev. Lett.* The eigenvalue-clipping idea.
- Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional
  covariance matrices. *J. Multivariate Anal.*
- Ledoit, O. and Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional
  covariance matrices. *Ann. Statist.*
- Jagannathan, R. and Ma, T. (2003). Risk reduction in large portfolios: why imposing
  the wrong constraints helps. *J. Finance.*
- DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal versus naive
  diversification. *Rev. Financ. Stud.*
