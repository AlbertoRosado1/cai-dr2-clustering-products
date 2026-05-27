# Cosmology inference

This directory contains cosmological inference helpers that run downstream of
`clustering_statistics/` measurements and `bao/` BAO fitting/compression.

DESI cosmology fits are run with Cobaya using lightweight `desi-clustering`
helpers: small runtime functions, explicit option dictionaries, injectable
paths, registry-driven likelihood metadata, and thin desipipe job scripts.

## Environment

Use Arnaud's current cosmodesi environment on Perlmutter:

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
cd /global/homes/u/uendert/repos/desi/desi-clustering-cosmo
export PYTHONPATH=$PWD:$PYTHONPATH
```

The module environment includes Cobaya, desipipe, desilike, and the standard
DESI cosmology stack. During development, the environment may also load a
packaged `desi-clustering` module. If Python imports that packaged copy instead
of this checkout, unload it before setting `PYTHONPATH`:

```bash
module unload desi-clustering || true
export PYTHONPATH=$PWD:$PYTHONPATH
```

This is a development safeguard while working from a checkout, not a separate
cosmology-runtime requirement.

## Current scope

BAO likelihoods use native lightweight Cobaya wrapper classes in
`cosmo.cobaya_likelihoods.bao.desi_dr2`. Shared BAO dataset metadata is kept in
`cosmo.bindings.bao`, registered in `cosmo.bindings.registry`, and translated to
Cobaya likelihood dictionaries by `cosmo.bindings.cobaya`. The registry uses a
general `likelihoods` interface so likelihood families can be combined without
changing the runtime helpers.

DESI BAO DR2 mean/covariance text files are read from the DESI collaboration CFS
area by default:

```text
/global/cfs/cdirs/desicollab/science/cpe/y3_bao_cosmo/bao_v1p2/bao/cobaya_data
```

Override this path with `likelihood_path=` or:

```bash
export DESI_CLUSTERING_COSMO_BAO_DATA_PATH=/path/to/cobaya_data
```

Registered likelihoods include DESI BAO DR2, SN, BBN, compressed CMB, full CMB,
and CMB-lensing entries. BBN, SN zmin variants, compressed-CMB priors, and the
Momento/SROLL wrapper are native `desi-clustering` likelihoods under
`cosmo.cobaya_likelihoods`. The Cobaya helper supports background-only
combinations, such as BAO+SN+BBN+compressed-CMB priors, and full-CMB Cobaya
configs for standard external Planck/ACT/lensing likelihoods. Full-CMB
initialization and production runs depend on the external Cobaya likelihood
packages/data available in the active environment. DESI DR1 BAO would require a
separate data mapping.

- likelihoods: `desi-bao-all`, SN names such as `pantheonplus`, BBN names such
  as `schoneberg2024-bbn`, compressed-CMB names such as `CMB-compressed-theta`,
  and full-CMB names such as `planck-NPIPE-highl-CamSpec-TTTEEE`;
- models: `base`, `base_w`, `base_w_wa`;
- theory: `camb`;
- samplers: `evaluate`, `cobaya` MCMC, `iminuit` minimization.

Compressed BAO products from `bao/` can be connected by writing compatible
mean/covariance text files and pointing `likelihood_path` to their directory.

## Direct Python usage

Use `cosmo.cobaya` for Cobaya configuration and runtime helpers.

```python
from cosmo.cobaya import get_cobaya_info, sample_cobaya

info = get_cobaya_info(
    model='base',
    likelihoods='desi-bao-all',
    sampler='evaluate',
    output=False,
)

sample_cobaya(model='base', likelihoods='desi-bao-all', sampler='evaluate', output=False)
```

For BAO-only calls, `dataset=` is accepted as a convenience alias:

```python
info = get_cobaya_info(model='base', dataset='desi-bao-all', sampler='evaluate', output=False)
```

Full-CMB configs can be generated with the `likelihoods=` interface:

```python
info = get_cobaya_info(
    model='base',
    likelihoods='bao-planck-npipe',
    sampler='evaluate',
    output=False,
)
```

To write a Cobaya YAML:

```python
from cosmo.cobaya import get_cobaya_info, write_cobaya_yaml

info = get_cobaya_info(model='base', likelihoods='desi-bao-all', sampler='cobaya')
write_cobaya_yaml(info, 'configs/base_desi-bao-all.yaml')
```

## Desipipe usage

The desipipe entry point is:

```bash
python cosmo/job_scripts/desipipe_cobaya.py \
    --todo evaluate \
    --models base \
    --likelihoods bao-sn-cmb-compressed-theta
```

Named likelihood-combination presets are available. List them with:

```bash
python cosmo/job_scripts/desipipe_cobaya.py --list-likelihood-combinations
```

Common presets include:

```text
bao
bao-sn-pantheonplus
bao-sn-union3
bao-sn-desy5
bao-sn-desdovekie
bao-bbn
bao-cmb-compressed-theta
bao-sn-cmb-compressed-theta
bao-planck-npipe
bao-planck-npipe-lensing
cmb-spa
bao-cmb-spa
bao-sn-desdovekie-cmb-spa
cmb-spa-tauprior
```

Explicit comma-separated likelihood combinations still work. Pass multiple
`--likelihoods` values to create multiple configurations:

```bash
python cosmo/job_scripts/desipipe_cobaya.py \
    --todo evaluate \
    --models base \
    --likelihoods bao \
                  desi-bao-all,pantheonplus \
                  bao-cmb-compressed-theta
```

Inspect and spawn:

```bash
desipipe tasks -q desi_clustering_cobaya
desipipe spawn -q desi_clustering_cobaya --spawn
desipipe queues -q desi_clustering_cobaya
```

For quick interactive debugging without creating a queue:

```bash
python cosmo/job_scripts/desipipe_cobaya.py \
    --interactive \
    --todo evaluate \
    --models base \
    --likelihoods bao-cmb-compressed-theta \
    --test
```

To export Cobaya YAML files for review without launching jobs:

```bash
python cosmo/job_scripts/desipipe_cobaya.py \
    --todo export \
    --models base base_w base_w_wa \
    --likelihoods bao bao-bbn bao-cmb-compressed-theta bao-cmb-spa \
    --config-dir configs/cobaya
```

This writes files such as:

```text
configs/cobaya/base/bao.yaml
configs/cobaya/base/bao-bbn.yaml
configs/cobaya/base/bao-cmb-compressed-theta.yaml
configs/cobaya/base/bao-cmb-spa.yaml
```


## Validation status

Generated Cobaya info dictionaries can be compared against the Y3 reference with:

```bash
python cosmo/job_scripts/compare_y3_cobaya_configs.py
```

The comparison canonicalizes native `desi-clustering` likelihood class paths and
checks canonical likelihoods, sampled parameters, and CAMB `extra_args`. The
default comparison set covers BAO-only, BAO+SN, BAO+BBN, BAO+compressed-CMB,
BAO+Planck NPIPE, BAO+SN+Planck NPIPE, and BAO+Planck NPIPE+lensing. For these
tested `base` configurations, sampled parameters and CAMB `extra_args` match the
reference comparison, except where runtime-complete native configs intentionally
provide parameters required by native likelihood initialization. The comparison
utility imports the Y3 repository only as a reference; runtime likelihood
configuration is native to this package.
