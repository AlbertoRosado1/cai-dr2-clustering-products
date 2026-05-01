# desi-clustering

Collection of scripts to produce the DESI DR2 clustering measurements from the data / mocks catalogs to the parameter inferences.

## 📦 Installation

You can install the latest version directly from the GitHub repository:

```bash
pip install git+https://github.com/cosmodesi/desi-clustering.git
```

Alternatively, if you plan to contribute or modify the code, install in editable (development) mode:

```bash
git clone https://github.com/cosmodesi/desi-clustering.git
cd desi-clustering
pip install -e .
```

Importantly, if you are working at NERSC with the `cosmodesi` conda environment, you need to unload the `desi-clustering` loaded with `cosmodesi`:
```` bash
module unload desi-clustering
# add packages locally installed to the PYTHONPATH (supposing you are working with python3.12 which is the python version in cosmodesi)
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages/:$PYTHONPATH
```

## Overview

The package follows a simple structure:

* **`clustering_statistics`**: measurement of clustering statistics (power spectrum, correlation function, bispectrum, etc.),
common to Full Shape, BAO and PNG Key Projects
* **`full_shape`**: full-shape fits
* **`bao`**: BAO fits
* **`local_png`**: local primordial non-Gaussianity

## Environment

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main  # source the environment
# You may want to have it in the jupyter-kernel for plots
${COSMODESIMODULES}/install_jupyter_kernel.sh main  # this to be done once
```
You may already have the above kernel (corresponding to the standard GQC environment) installed.
In this case, you can delete it:
```bash
rm -rf $HOME/.local/share/jupyter/kernels/cosmodesi-main
```
and rerun:
```bash
${COSMODESIMODULES}/install_jupyter_kernel.sh main
```
Note that you may need to restart (close and reopen) your notebooks for the changes to propagate.


----
### How to install `cosmodesi-main-dev` jupyter enviornment

(i.e. the standard `cosmodesi-main` env but with a developement-mode use for the `desi-clustering` module)

(1) Install `desi-clustering` with
```
git clone https://github.com/cosmodesi/desi-clustering.git
cd desi-clustering
pip install -e .
```

(2) Create a file `$HOME/.local/share/jupyter/kernels/cosmodesi-main-dev/cosmodesi-dev-kernel.sh` with the contents
```
#!/bin/bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desi-clustering
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
exec python -m ipykernel_launcher -f "$1"
```
and make it executable.

(3) Create a file `$HOME/.local/share/jupyter/kernels/cosmodesi-main-dev/kernel.json` with the contents
```
{
 "language": "python",
 "argv": [
  "$HOME/.local/share/jupyter/kernels/cosmodesi-main-dev/cosmodesi-dev-kernel.sh",
  "{connection_file}"
 ],
 "display_name": "cosmodesi-main-dev"
}
```
Restart the notebook and you should see `cosmodesi-main-dev` appear in the kernel options. You can verify that the `desi-clustering` module is being loaded from the cloned repo with
```
import clustering_statistics; print(clustering_statistics.__file__)
```
