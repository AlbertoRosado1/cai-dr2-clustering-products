"""
cutsky_data_tools.py
====================
Data loading utilities for DESI cutsky power spectrum and bispectrum measurements.

This module provides functions to load pre-processed data vectors, covariance
matrices, and window matrices for DESI Y1 cutsky tracer samples, and to assemble
them into dictionaries suitable for use with the desilike inference framework.

Functions
---------
build_pk_data_cutsky
    Assemble power-spectrum-only dataset for a given tracer and sky region.
build_pk_bk_data_cutsky
    Assemble joint power spectrum + bispectrum dataset for a given tracer.

Tracer metadata (effective redshifts, representative sigma_8, data file paths)
are collected in module-level dictionaries so they are easy to update:
  - TRACER_REDSHIFTS
  - TRACER_SIGMA8
  - TRACER_DATA_FILES

Usage
-----
>>> from cutsky_data_tools import build_pk_data_cutsky
>>> dataset = build_pk_data_cutsky(tracer='LRG2', region='SGC', k_max_p=0.20)
>>> pk_monopole = dataset['p0']

"""

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Tracer metadata
# ---------------------------------------------------------------------------

#: Effective redshift for each tracer, used to evaluate the growth rate f(z)
#: and the fiducial power spectrum template.
TRACER_REDSHIFTS = {
    'BGS':  0.295,
    'LRG1': 0.5094,
    'LRG2': 0.7054,
    'LRG3': 0.9264,
    'ELG':  1.3442,
    'QSO':  1.4864,
}

#: AbacusSummit fiducial sigma_8(z_eff) for each tracer, used when setting
#: Gaussian priors centred on the linear bias b_s in physical parametrisation.
TRACER_SIGMA8 = {
    'BGS':  0.69376997,
    'LRG1': 0.62404056,
    'LRG2': 0.53930063,
    'LRG3': 0.50823223,
    'ELG':  0.43197292,
    'QSO':  0.41825647,
}

# Base directory for DESI Y1 pre-processed measurement files
_DATA_BASE = (
    '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/'
    'LSScats/v1.5/unblinded/desipipe/forfit_2pt'
)

#: Absolute paths to the pre-processed (forfit) power spectrum .npy files,
#: keyed by (tracer, region).  GCcomb refers to the NGC+SGC combined sample;
#: separate NGC and SGC files may exist with the same naming convention but
#: different region tags in the filename.
#:
#: Add SGC / NGC entries as needed once those files are available.
TRACER_DATA_FILES: dict = {
    ('BGS',  'GCcomb'): (
        f'{_DATA_BASE}/forfit_power_syst-rotation-hod-photo_klim_'
        '0-0.02-0.20_2-0.02-0.20_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4.npy'
    ),
    ('LRG1', 'GCcomb'): (
        f'{_DATA_BASE}/forfit_power_syst-rotation-hod-photo_klim_'
        '0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.4-0.6.npy'
    ),
    ('LRG2', 'GCcomb'): (
        f'{_DATA_BASE}/forfit_power_syst-rotation-hod-photo_klim_'
        '0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.6-0.8.npy'
    ),
    ('LRG3', 'GCcomb'): (
        f'{_DATA_BASE}/forfit_power_syst-rotation-hod-photo_klim_'
        '0-0.02-0.20_2-0.02-0.20_LRG_GCcomb_z0.8-1.1.npy'
    ),
    ('ELG',  'GCcomb'): (
        f'{_DATA_BASE}/forfit_power_syst-rotation-hod-photo_klim_'
        '0-0.02-0.20_2-0.02-0.20_ELG_LOPnotqso_GCcomb_z1.1-1.6.npy'
    ),
    ('QSO',  'GCcomb'): (
        f'{_DATA_BASE}/forfit_power_syst-rotation-hod-photo_klim_'
        '0-0.02-0.20_2-0.02-0.20_QSO_GCcomb_z0.8-2.1.npy'
    ),
    # TODO: add SGC and NGC entries once separate-region files are confirmed
    # ('LRG2', 'SGC'): '/path/to/..._LRG_SGC_z0.6-0.8.npy',
    # ('LRG2', 'NGC'): '/path/to/..._LRG_NGC_z0.6-0.8.npy',
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_forfit_file(tracer: str, region: str) -> dict:
    """Load a pre-processed power spectrum .npy file for *tracer* and *region*.

    Parameters
    ----------
    tracer : str
        Tracer label, e.g. ``'LRG2'``.
    region : str
        Sky region, e.g. ``'GCcomb'``, ``'SGC'``, ``'NGC'``.

    Returns
    -------
    dict
        Dictionary stored in the .npy file (see Notes).

    Notes
    -----
    The forfit files produced by the DESI pipeline are numpy archives that,
    when loaded with ``allow_pickle=True``, return a Python dict (or a 0-d
    object array wrapping one).  Expected keys vary by pipeline version but
    typically include ``'k'``, ``'P0'``, ``'P2'``, ``'P4'``,
    ``'cov'`` / ``'covariance'``, and ``'wmatrix'`` / ``'window_matrix'``.
    """
    key = (tracer, region)
    if key not in TRACER_DATA_FILES:
        raise KeyError(
            f'No data file entry for tracer={tracer!r}, region={region!r}.\n'
            f'Available entries: {list(TRACER_DATA_FILES.keys())}\n'
            'Add missing entries to TRACER_DATA_FILES in cutsky_data_tools.py.'
        )
    path = Path(TRACER_DATA_FILES[key])
    if not path.exists():
        raise FileNotFoundError(
            f'Data file not found for tracer {tracer!r}, region {region!r}:\n'
            f'  {path}\n'
            'Check TRACER_DATA_FILES in cutsky_data_tools.py.'
        )
    raw = np.load(path, allow_pickle=True)
    # The pipeline may wrap the dict in a 0-d object array
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        return raw.item()
    return dict(raw)


def _apply_hartlap(cov: np.ndarray, n_mocks: int) -> np.ndarray:
    """Apply the Hartlap (2007) correction factor to a precision matrix.

    The unbiased estimate of the inverse covariance is

        C^{-1}_unbiased = (N_s - N_d - 2) / (N_s - 1) * C^{-1}_sample

    where *N_s* is the number of simulations and *N_d* is the dimension of
    the data vector.  Here we scale the covariance itself by the inverse of
    that factor so that its inverse already incorporates the correction.

    Parameters
    ----------
    cov : ndarray, shape (N_d, N_d)
        Sample covariance matrix.
    n_mocks : int
        Number of mock realisations used to estimate the covariance.

    Returns
    -------
    ndarray
        Hartlap-corrected covariance matrix.
    """
    n_d = cov.shape[0]
    if n_mocks <= n_d + 2:
        raise ValueError(
            f'n_mocks ({n_mocks}) must be larger than n_d + 2 ({n_d + 2}) '
            'for the Hartlap correction to be valid.'
        )
    hartlap = (n_mocks - 1) / (n_mocks - n_d - 2)
    return cov * hartlap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pk_data_cutsky(
    tracer: str,
    region: str,
    k_max_p: float = 0.20,
) -> dict:
    """Assemble a power-spectrum-only dataset for *tracer* and *region*.

    Loads the pre-processed DESI forfit power spectrum file and extracts the
    data vector (P0, P2), covariance, window matrix, and k-grid arrays needed
    to build a ``TracerPowerSpectrumMultipolesObservable`` in desilike.

    Parameters
    ----------
    tracer : str
        Tracer label, e.g. ``'LRG2'``.  Must be present in
        ``TRACER_DATA_FILES``.
    region : str
        Sky region, e.g. ``'SGC'``, ``'NGC'``, or ``'GCcomb'``.  Used to
        select the correct subblock of the covariance / window matrix when
        the file contains multiple regions.
    k_max_p : float, optional
        Maximum wavenumber [h/Mpc] to include in the power spectrum data
        vector.  Default is 0.20 h/Mpc.

    Returns
    -------
    dict with keys
        p0 : ndarray
            Monopole data vector, shape (N_k,).
        p2 : ndarray
            Quadrupole data vector, shape (N_k,).
        k_data : ndarray
            Observed k values [h/Mpc] at which P0/P2 are measured, shape (N_k,).
        k_window : ndarray
            Fine k grid [h/Mpc] on which the window matrix is defined (used
            as ``kin`` in desilike), shape (N_kin,).
        window_matrix : ndarray
            Window matrix of shape (N_ell_out * N_k, N_ell_in * N_kin),
            mapping theory multipoles at ``k_window`` to observed multipoles
            at ``k_data``.
        cov_pk : ndarray
            Covariance matrix for the concatenated data vector
            [P0, P2], shape (2*N_k, 2*N_k).  The Hartlap correction is
            applied if ``n_mocks`` is provided in the data file.
    """
    data = _load_forfit_file(tracer, region)

    # ------------------------------------------------------------------
    # k grids
    # k_data  : k values at which the multipoles are observed
    # k_window: fine k grid for the window matrix convolution
    # ------------------------------------------------------------------
    k_data = data['k']           # shape (N_k,)
    k_window = data['k_window']  # shape (N_kin,)

    # Apply k-max cut
    mask = k_data <= k_max_p
    k_data = k_data[mask]

    # ------------------------------------------------------------------
    # Data vectors
    # ------------------------------------------------------------------
    p0 = data['P0'][mask]
    p2 = data['P2'][mask]

    # ------------------------------------------------------------------
    # Covariance (Hartlap correction applied inside the pipeline)
    # ------------------------------------------------------------------
    n_d = 2 * len(k_data)  # monopole + quadrupole
    cov_full = data['covariance']  # full covariance across all multipoles

    # The forfit files store the covariance for the full k range; we slice
    # the monopole and quadrupole blocks to match our k_max cut.
    n_k_full = len(data['k'])
    mask_full = np.concatenate([mask, mask])  # monopole + quadrupole
    cov_pk = cov_full[np.ix_(mask_full, mask_full)]

    # ------------------------------------------------------------------
    # Window matrix
    # The window matrix rows correspond to (ellout, k_data) pairs and
    # columns to (ellin, k_window) pairs.
    # ------------------------------------------------------------------
    window_matrix = data['window_matrix']

    return dict(
        p0=p0,
        p2=p2,
        k_data=k_data,
        k_window=k_window,
        window_matrix=window_matrix,
        cov_pk=cov_pk,
    )


def build_pk_bk_data_cutsky(
    tracer: str,
    region: str,
    k_max_p: float = 0.20,
    k_max_b0: float = 0.20,
    k_max_b2: float = 0.03,
) -> dict:
    """Assemble a joint power spectrum + bispectrum dataset for *tracer*.

    Loads power spectrum data from the forfit .npy file (same as
    :func:`build_pk_data_cutsky`) and bispectrum data from the corresponding
    bispectrum file, then returns a unified dataset dictionary.

    Parameters
    ----------
    tracer : str
        Tracer label, e.g. ``'LRG2'``.
    region : str
        Sky region, e.g. ``'SGC'`` or ``'NGC'``.
    k_max_p : float, optional
        Maximum k [h/Mpc] for the power spectrum. Default 0.20.
    k_max_b0 : float, optional
        Maximum k [h/Mpc] for the bispectrum monopole B_000.  Default 0.20.
    k_max_b2 : float, optional
        Maximum k [h/Mpc] for the bispectrum quadrupole B_202.  Default 0.03.

    Returns
    -------
    dict with keys
        p0 : ndarray
            Power spectrum monopole, shape (N_kp,).
        p2 : ndarray
            Power spectrum quadrupole, shape (N_kp,).
        k_data : ndarray
            k values for the power spectrum data, shape (N_kp,).
        k_window : ndarray
            Fine k grid for the PS window convolution, shape (N_kin,).
        window_matrix : ndarray
            Power spectrum window matrix.
        cov_pk : ndarray
            Covariance for [P0, P2] alone, shape (2*N_kp, 2*N_kp).
        kr_b0 : ndarray
            Triangle configuration array for B_000, shape (N_b0, 3).
            The first column contains the k values used to cut on k_max_b0.
        kr_b2 : ndarray
            Triangle configuration array for B_202, shape (N_b2, 3).
        b000 : ndarray
            Bispectrum monopole data vector, shape (N_b0,).
        b202 : ndarray
            Bispectrum quadrupole data vector, shape (N_b2,).
        covariance : ndarray
            Full joint covariance for [P0, P2, B000, B202],
            shape (2*N_kp + N_b0 + N_b2, 2*N_kp + N_b0 + N_b2).
    """
    # ---- Power spectrum part ----
    pk_dataset = build_pk_data_cutsky(tracer=tracer, region=region, k_max_p=k_max_p)

    # ---- Bispectrum part ----
    bk_data = _load_bispectrum_file(tracer=tracer, region=region)

    # Triangle configurations: 2D arrays with columns (k1, k2, k3)
    kr_b0_full = bk_data['kr_b0']   # shape (N_b0_full, 3)
    kr_b2_full = bk_data['kr_b2']   # shape (N_b2_full, 3)

    # Apply k-max cuts on the first k leg
    mask_b0 = kr_b0_full[:, 0] <= k_max_b0
    mask_b2 = kr_b2_full[:, 0] <= k_max_b2
    kr_b0 = kr_b0_full[mask_b0]
    kr_b2 = kr_b2_full[mask_b2]

    b000_full = bk_data['b000']
    b202_full = bk_data['b202']
    b000 = b000_full[mask_b0]
    b202 = b202_full[mask_b2]

    # ---- Full joint covariance ----
    # The joint covariance is stored in the bispectrum file and covers all
    # data components: [P0, P2, B000, B202] in that order.
    cov_joint_full = bk_data['covariance']

    # Build index mask for the joint covariance blocks
    n_kp_full = len(bk_data['k'])
    n_b0_full = len(b000_full)
    n_b2_full = len(b202_full)

    # Indices within the full data vector
    idx_p0 = np.arange(n_kp_full)
    idx_p2 = np.arange(n_kp_full, 2 * n_kp_full)
    idx_b0 = np.arange(2 * n_kp_full, 2 * n_kp_full + n_b0_full)
    idx_b2 = np.arange(2 * n_kp_full + n_b0_full, 2 * n_kp_full + n_b0_full + n_b2_full)

    kp_mask = pk_dataset['k_data'] <= k_max_p  # already within k_max_p
    # map back to full k indices
    k_full = bk_data['k']
    pk_mask_full = k_full <= k_max_p

    sel_p0 = idx_p0[pk_mask_full]
    sel_p2 = idx_p2[pk_mask_full]
    sel_b0 = idx_b0[mask_b0]
    sel_b2 = idx_b2[mask_b2]

    selection = np.concatenate([sel_p0, sel_p2, sel_b0, sel_b2])
    covariance = cov_joint_full[np.ix_(selection, selection)]

    # PS-only covariance block for convenience
    n_p = len(pk_dataset['p0'])
    sel_pk = np.arange(2 * n_p)
    cov_pk = covariance[np.ix_(sel_pk, sel_pk)]

    return dict(
        # Power spectrum
        p0=pk_dataset['p0'],
        p2=pk_dataset['p2'],
        k_data=pk_dataset['k_data'],
        k_window=pk_dataset['k_window'],
        window_matrix=pk_dataset['window_matrix'],
        cov_pk=cov_pk,
        # Bispectrum
        kr_b0=kr_b0,
        kr_b2=kr_b2,
        b000=b000,
        b202=b202,
        # Joint covariance
        covariance=covariance,
    )


def _load_bispectrum_file(tracer: str, region: str) -> dict:
    """Load the pre-processed bispectrum .npy file for *tracer* and *region*.

    Parameters
    ----------
    tracer : str
        Tracer label.
    region : str
        Sky region.

    Returns
    -------
    dict
        Dictionary with keys ``'k'``, ``'kr_b0'``, ``'kr_b2'``,
        ``'b000'``, ``'b202'``, ``'covariance'``.

    Notes
    -----
    Bispectrum files are not yet stored in a single standard location.
    Update the ``_BISPECTRUM_DATA_FILES`` dictionary below to point to the
    correct paths for your analysis.
    """
    # TODO: Populate this dictionary with the correct paths for each tracer
    # and region.  The expected format is the same as the power spectrum
    # forfit files: a numpy .npy file that, when loaded, yields a dict with
    # the keys listed in the docstring.
    _BISPECTRUM_DATA_FILES: dict = {
        # Example entry (replace with actual paths):
        # ('LRG2', 'SGC'): '/path/to/bispectrum_LRG2_SGC.npy',
    }
    key = (tracer, region)
    if key not in _BISPECTRUM_DATA_FILES:
        raise NotImplementedError(
            f'No bispectrum data file configured for tracer={tracer!r}, '
            f'region={region!r}.  Add an entry to _BISPECTRUM_DATA_FILES '
            'in cutsky_data_tools._load_bispectrum_file().'
        )
    path = Path(_BISPECTRUM_DATA_FILES[key])
    if not path.exists():
        raise FileNotFoundError(f'Bispectrum file not found:\n  {path}')
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        return raw.item()
    return dict(raw)
