# Mathematical Model

## Measurement

```text
m(t) = g(t) + r(t) + n(t)
```

After ground balance:

```text
x(t) = max(0, m(t) - alpha * g_hat(t))
```

## PI Decay

```text
r(t) = A exp(-t/tau) + A_f exp(-t/tau_f) + A_c exp(-t/tau_c)
```

## Feature Vector

The model uses a 38-dimensional feature vector containing signal amplitude, phase proxy, tau estimate, ferrous index, conductivity response, soil parameters, 8 waveform samples, decay area, slope, early/late ratio, SNR, ground balance error, and pulse parameters.

## Classifier

Each class is modelled with Gaussian feature distributions:

```text
log P(c|x) ∝ log P(c) + sum_j log N(x_j; mu_cj, sigma_cj^2)
```

## Verified Waveform Memory

Weighted distance:

```text
d(x, x_i) = sqrt(sum_j w_j (z_j - z_ij)^2 / sum_j w_j)
```

Gaussian kernel similarity:

```text
K(x, x_i) = exp(-d(x, x_i)^2 / (2 sigma_K^2))
```

Final fusion:

```text
P_final(c|x) = (1 - lambda_m) p_c + lambda_m q_c
```
