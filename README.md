# optiMOFA

**optiMOFA is mofapy2 0.7.5 with exact, much faster updates for the common case of Gaussian
views with one group and no missing values.** It is a fork of
[bioFAM/mofapy2](https://github.com/bioFAM/mofapy2) and a drop-in replacement: same
package name (`mofapy2`), same API, same results.

| 769 samples, 200,000 + 9,512 features, 100 factors (float32) | s / iteration | speedup |
|---|---|---|
| mofapy2 0.7.5, CPU | 76.8 | 1x |
| mofapy2 0.7.5, `gpu_mode` | 24.1 | 3x |
| **optiMOFA, CPU** | **2.45** | **31x** |
| **optiMOFA, `gpu_mode`** | **1.69** | **45x** |

Peak memory on the CPU is slightly lower than stock (7.7 vs 8.2 GB).

## What changes

When a view has one group and no missing values, the noise precision of a feature is the
same for every sample (by construction of the model). It can then be taken out of every sum
over samples, and the W (spike-and-slab), Z and Tau updates, the variance explained used for
dropping factors, and the likelihood term of the ELBO can all be written with the K x K Gram
matrix `E[Z]^T E[Z]` and the D x K product `Y^T E[Z]`. This removes the O(N D K^2) terms and
all N x D temporaries. These are algebraic identities, not approximations: the fits take the
same iterations, drop the same factors and agree with stock mofapy2 to floating-point
rounding. In `gpu_mode`, each view's data matrix is also uploaded to the GPU once per fit
(stock re-uploads it in every update).

Wherever the conditions do not hold (several groups, missing values in a view, stochastic
inference, non-Gaussian likelihoods, W without spike-and-slab, spike-and-slab on Z, GP
nodes), the unchanged stock code runs.

- Derivations, validity conditions, all test and benchmark results:
  [docs/fast_updates.md](docs/fast_updates.md)
- How and why the fork was made: [docs/LOG.md](docs/LOG.md)
- Code: [mofapy2/core/fast_updates.py](mofapy2/core/fast_updates.py) and one early
  branch each in the W, Z, Tau and Y nodes and in `BayesNet.calculate_variance_explained`

## Install

```
pip install git+https://github.com/tristanRW/optiMOFA
```

or, with uv, pinned to a commit in `pyproject.toml`:

```toml
[tool.uv.sources]
mofapy2 = { git = "https://github.com/tristanRW/optiMOFA", rev = "<commit>" }
```

## Use

Nothing changes; the fast updates are on by default. To run the stock updates (for example
to verify a result):

```python
ent.set_train_options(..., fast_updates=False)
# or mofapy2.run.entry_point.mofa(..., fast_updates=False)
```

## Tests

```
pytest tests/test_fast_updates.py
```

The tests fit ten synthetic data sets with different statistical properties (feature-specific
noise, K close to N, pure noise, heavy tails with outliers, correlated and unscaled features,
sample-dependent noise, two groups, missing values, N >> D) with `fast_updates=False` and
`True`, and require identical iterations and factor drops and agreement to 1e-9 in E[W],
E[Z], E[tau], the ELBO and the variance explained. They also check float32, and, if CuPy and a
GPU are present, GPU mode (float64 and float32), that the data are uploaded once, and that
two GPU fits with the same seed are bit-identical.

## License

LGPL-3.0, as upstream. All credit for MOFA+ and mofapy2 goes to its authors (Argelaguet,
Arnol, Bredikhin, Velten et al.); please cite MOFA+ (Argelaguet et al., *Genome Biology*
2020) when using this fork.

---

*Upstream README follows.*

# Multi-Omics Factor Analysis

[![PyPi version](https://img.shields.io/pypi/v/mofapy2)](https://pypi.org/project/mofapy2)

MOFA is a factor analysis model that provides a general framework for the integration of multi-omic data sets in an unsupervised fashion.  
This repository contains `mofapy2` Python library source code.

- For the downstream analysis in Python please check the mofax package: https://github.com/bioFAM/mofax
- For the downstream analysis in R please check the MOFA2 package: https://github.com/bioFAM/MOFA2

Please [visit our website](https://biofam.github.io/MOFA2/) for details, tutorials, and much more.

## Installation

Install the stable version from the Python Package Index:
```
pip install mofapy2
```

Or install the latest development version from the repository:
```
pip install git+https://github.com/bioFAM/mofapy2@dev --force-reinstall --no-deps
```


