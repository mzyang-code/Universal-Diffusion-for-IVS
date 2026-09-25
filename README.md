# Universal Diffusion Models for Implied Volatility Surfaces

Code for *Universal Diffusion Models for Implied Volatility Surfaces: Learning
Shared Dynamics Across Stocks* — [arXiv:2609.22893](https://arxiv.org/abs/2609.22893).

A single universal conditional diffusion model (DDPM) is trained on a pooled cross-section of
stocks to jointly generate the next-day return and implied-volatility surface
increment, and is compared against a pooled VolGAN benchmark.

## Repository layout

### `configs/`
One json per experiment (`ddpm_mse`, `ddpm_smooth`, `ddpm_arbitrage`, `ddpm_hybrid`,
`volgan`), each with an `*_oos.json` twin for the out-of-sample stocks.

### `ivs/` — shared library
| file | purpose |
|---|---|
| `config.py` | loads every constant from the selected experiment config |
| `data/dataset.py` | builds (condition, target) pairs from the tensor cache |
| `data/normalizer.py` | z-score normalizer fitted on the pooled train split |
| `data/prepare/options_scrubber.py` | quote-quality filters on raw option quotes |
| `data/prepare/utils_rates.py` | risk-free rate and spot-price loaders |
| `data/prepare/select_universe.py` | draws the training universe |
| `data/prepare/bandwidth_search.py` | per-ticker Nadaraya-Watson bandwidth search |
| `data/prepare/tensor_prep.py` | smooths daily quotes onto the fixed (m, τ) grid |
| `evaluate/arbitrage.py` | static-arbitrage penalties and arbitrage violations (Table 2) |
| `evaluate/forecasting.py` | 95% predictive-interval coverage of the return (Table 3) |
| `evaluate/pca.py` | explained-variance ratios of PC1-3 (Table 4) |
| `evaluate/run.py` | evaluates one experiment's samples, writes per-ticker metrics |

### `diffusion/` — conditional DDPM
| file | purpose |
|---|---|
| `losses.py` | MSE plus optional smoothness and static-arbitrage penalties |
| `calibrate_lambda.py` | gradient-matching calibration of the penalty weights |
| `model/denoiser.py` | FiLM-conditioned MLP denoiser |
| `model/process.py` | noise schedule and DDPM sampling |
| `train.py` | training on the pooled cross-section |
| `sample.py` | scenario generation for each test state |

### `volgan/` — pooled VolGAN benchmark （replicated）
| file | purpose |
|---|---|
| `losses.py` | BCE plus log-IV smoothness penalties |
| `model/networks.py` | generator and discriminator |
| `train.py` | training with gradient-matched penalties |
| `sample.py` | scenario generation for each test state |
