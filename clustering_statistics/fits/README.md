# EFT Galaxy Clustering Fits (Sandbox)

Scripts for fitting EFT models to DESI Y1 cutsky power spectrum and bispectrum 
multipoles using [desilike](https://github.com/cosmodesi/desilike) and FOLPSv2.

> **Status**: Work in progress.  These scripts are being validated before
> integration into the main `clustering_statistics` module.

---

## Files

| File | Description |
|---|---|
| `cutsky_data_tools.py` | Data loading utilities (power spectrum and bispectrum) |
| `fit_pk_eft.py` | EFT power spectrum fit — multiple tracers, joint cosmology |
| `fit_pk_bk_eft.py` | EFT joint power spectrum + bispectrum fit — single tracer |

---

## Quick start

### Power spectrum only (`fit_pk_eft.py`)

Fits P0 + P2 multipoles for one or more DESI tracers simultaneously with shared
cosmological parameters and per-tracer nuisance parameters.

```bash
# Test likelihood evaluation
python fit_pk_eft.py --test --tracers LRG2 --region GCcomb

# Run MCMC chain (Cobaya sampler)
python fit_pk_eft.py --run_chains --tracers LRG1 LRG2 LRG3 ELG QSO --region GCcomb

# Plot best-fit theory from saved chain
python fit_pk_eft.py --plot_bestfit --tracers LRG2 --region GCcomb

# Plot posterior contours
python fit_pk_eft.py --plot_chains --tracers LRG2 --region GCcomb
```

All settings have sensible defaults; the most important ones can also be
overridden at the class level in `FitConfig` inside the script.

### Power spectrum + bispectrum (`fit_pk_bk_eft.py`)

Fits P0 + P2 and B000 + B202 simultaneously for a single tracer, using the full
joint covariance matrix.

```bash
# Test likelihood evaluation
python fit_pk_bk_eft.py --test --tracer LRG2 --region SGC

# Run MCMC chain
python fit_pk_bk_eft.py --run_chains --tracer LRG2 --region SGC \
    --k_max_p 0.20 --k_max_b0 0.20 --k_max_b2 0.03

# Plot best-fit
python fit_pk_bk_eft.py --plot_bestfit --tracer LRG2 --region SGC

# Plot posteriors
python fit_pk_bk_eft.py --plot_chains --tracer LRG2 --region SGC
```

---

## Configuration

Both scripts expose a `FitConfig` dataclass with all tunable settings.  Edit
the defaults in the class definition for batch jobs, or override specific
settings via command-line arguments for interactive use.

Key settings:

| Setting | Default | Description |
|---|---|---|
| `prior_basis` | `'physical_prior_doc'` | `'physical_prior_doc'` or `'standard'` bias parametrisation |
| `pt_model` | `'EFT'` | Perturbation theory model (`'EFT'` = FOLPSv2, `'rept_velocileptors'`) |
| `damping` | `'lor'` | FoG damping: `'lor'`, `'exp'`, or `'vdg'` |
| `k_max_p` | `0.201` | Maximum k [h/Mpc] for the power spectrum |
| `k_max_b0` | `0.20` | Maximum k [h/Mpc] for B000 (PS+BK script only) |
| `k_max_b2` | `0.03` | Maximum k [h/Mpc] for B202 (PS+BK script only) |
| `use_emulator` | `True` | Use a 4th-order Taylor emulator (strongly recommended for MCMC) |
| `A_full` | `False` | Use full A-matrix in FOLPS |
| `b3_coev` | `False` | Enforce co-evolution prior b3 = 32/315*(b1-1) |
| `GR_criteria` | `0.03` | Cobaya Gelman–Rubin convergence threshold R-1 |

---

## Data tools (`cutsky_data_tools.py`)

Provides two public functions:

```python
from cutsky_data_tools import build_pk_data_cutsky, build_pk_bk_data_cutsky

# Power spectrum only
dataset = build_pk_data_cutsky(tracer='LRG2', region='SGC', k_max_p=0.20)
# Keys: p0, p2, k_data, k_window, window_matrix, cov_pk

# Joint PS + BK
dataset = build_pk_bk_data_cutsky(
    tracer='LRG2', region='SGC',
    k_max_p=0.20, k_max_b0=0.20, k_max_b2=0.03,
)
# Additional keys: kr_b0, kr_b2, b000, b202, covariance
```

### TODO before running `fit_pk_bk_eft.py`

The bispectrum data file loader (`_load_bispectrum_file`) is not yet wired up
to actual files.  You need to populate the `_BISPECTRUM_DATA_FILES` dictionary
inside that function with the correct paths for each `(tracer, region)` pair.

---

## Dependencies

- `desilike` (with desilike's FOLPSv2 and Velocileptors theories)
- `cosmoprimo`
- `numpy`, `scipy`
- `cobaya` (for Cobaya sampler)
- `emcee` (optional, for emcee sampler)
- `getdist` (for chain plotting)
- JAX (for the FOLPS JAX backend; set `FOLPS_BACKEND=jax`)

---

## Notes

- The **Hartlap correction** to the inverse covariance is expected to be
  applied inside the data tools (either by the pipeline that produced the
  forfit files, or explicitly in `cutsky_data_tools.py`).
- The **Taylor emulator** is built once at the start of each run and takes
  O(minutes); the MCMC then runs with O(ms) per evaluation.
- **Analytic marginalisation** over EFT counterterms (alpha0/2/4) and
  shot-noise parameters (sn0/2) is applied to the PS theory, removing them
  from the MCMC chain and improving convergence.
- Chain files are saved as `.npy` in the path returned by `cfg.chain_name()`.
