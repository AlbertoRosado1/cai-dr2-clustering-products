# desi-clustering

Collection of scripts to produce the DESI DR2 clustering measurements from the data / mocks catalogs to the parameter inferences.


## Overview

The package follows a simple structure:

* **`clustering_statistics`**: measurement of clustering statistics (power spectrum, correlation function, bispectrum, etc.),
common to Full Shape, BAO and PNG Key Projects
* **`full_shape`**: full-shape fits
* **`bao`**: BAO fits
* **`local_png`**: local primordial non-Gaussianity

Tutorial notebooks (this a work in progress...):

* **`nb/`**: contains an example of how to load the catalogs in particular with expand option.
* **`clustering_statistics/tutorials`**: contains reading and clustering measurements example.


## Environment

If you are on NERSC and in DESI (if not or if you want to modify the code see Installation), you may access this code (and all the necessary dependencies) by loading `cosmodesi` conda environment:
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


## 📦 Installation

### Standard installation:

You can install the latest version directly from the GitHub repository:

```bash
pip install git+https://github.com/cosmodesi/desi-clustering.git
```

### Development mode:

Alternatively, if you plan to contribute or modify the code, install in editable (development) mode:

```bash
git clone https://github.com/cosmodesi/desi-clustering.git
cd desi-clustering
pip install -e .
```

Importantly, if you are working at NERSC with the `cosmodesi` conda environment, you need to unload the `desi-clustering` loaded with `cosmodesi`:
```bash
module unload desi-clustering
# add packages locally installed to the PYTHONPATH (supposing you are working with python3.12 which is the python version in cosmodesi)
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages/:$PYTHONPATH
```

### How to install developement-mode jupyter kernel:

`cosmodesi-main-dev` is the standard `cosmodesi-main` env but with a developement-mode use for the `desilike` and `desi-clustering` modules.

(1) Install `desi-clustering` in developer mode with
```bash
git clone https://github.com/cosmodesi/desi-clustering.git
cd desi-clustering
git checkout edmond-dev
pip install --user -e .
```

(2) Install `desilike` in developer mode with 
```bash
git clone https://github.com/cosmodesi/desilike.git
cd desilike
git checkout dr2-dev
pip install --user -e .
```

(3) Setup the jupyter kernel by running:
```sh
KERNEL_DIR="$HOME/.local/share/jupyter/kernels/cosmodesi-main-dev"
mkdir -p "$KERNEL_DIR"

# Write the launcher script.
cat > "$KERNEL_DIR/cosmodesi-dev-kernel.sh" <<'EOF'
#!/bin/bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desilike
module unload desi-clustering
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH
exec python -m ipykernel_launcher -f "$1"
EOF

chmod u+x "$KERNEL_DIR/cosmodesi-dev-kernel.sh"

# Write kernel.json
cat > "$KERNEL_DIR/kernel.json" <<'EOF'
{
 "language": "python",
 "argv": [
  "{resource_dir}/cosmodesi-dev-kernel.sh",
  "{connection_file}"
 ],
 "display_name": "cosmodesi-main-dev"
}
EOF
```
Restart the notebook and you should see `cosmodesi-main-dev` appear in the kernel options. You can verify that the `desilike` and `desi-clustering` modules are being loaded from the cloned repos with
```python
import desilike; print(desilike.__file__)
import clustering_statistics; print(clustering_statistics.__file__)
```
