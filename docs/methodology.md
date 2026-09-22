# Methodology and validation notes

## Timeline

For each month-end signal date `t`:

1. Use adjusted prices observed through `t` and macro observations available after the configured
   publication lag.
2. Classify the QMI state and compute cross-sectional factor scores.
3. Update factor efficacy using only previously completed one-month forecasts. Allocate a 60%
   momentum core and distribute the 40% satellite using positive trailing information
   coefficients, partly conditioned on earlier occurrences of the current QMI regime.
4. Estimate the covariance matrix from returns through `t`.
5. Solve for target weights subject to stock, sector, net and gross constraints.
6. Apply the target after the next trading close; the next day's return is the first earned return.
7. Deduct turnover costs on target changes and accrue short-borrow cost daily.

This sequence deliberately sacrifices one trading day rather than assume an untradeable
month-end close formed with that same close.

## QMI construction

The QMI is an equal-weight composite of expanding z-scores:

```text
QMI = mean(
  business confidence,
  - three-month change in unemployment,
  - absolute deviation of inflation from 2%,
  - three-month change in the ECB deposit rate
)
```

At least three components must be available. The two-month release lag is configurable. A
rule-based level/direction classifier is used because it is interpretable and stable. It conditions
the walk-forward factor-efficacy estimate; it does not impose fixed factor labels learned from the
full sample. A four-component Gaussian mixture model is exported only as a diagnostic. The
Ornstein-Uhlenbeck half-life is likewise descriptive and does not set the rebalance frequency.

## Optimization

For score vector `s`, covariance matrix `Sigma`, current weights `w0` and candidate weights `w`,
the objective is:

```text
minimize  -s'w + lambda * w'Sigma w + gamma * ||w - w0||_1
```

The implementation uses SLSQP with explicit bounds and nonlinear absolute-exposure constraints.
Only securities in the score tails can receive positions. This makes the portfolio's long and
short decisions auditable rather than allowing the optimizer to reverse a weak signal.

## Validation

The automated suite verifies:

- exact mapping of QMI quadrants to the four named regimes;
- positive OU half-life on a known mean-reverting process;
- stock, sector, net and gross limits on optimized weights;
- invariance of walk-forward factor weights to current and future outcomes;
- invariance of historical factor scores when only future prices are changed.

The CI workflow runs Ruff and pytest on Python 3.12 for every push and pull request.
