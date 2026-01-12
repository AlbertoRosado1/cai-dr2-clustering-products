import os
import functools
from pathlib import Path

import tools
from compute_fiducial_stats import compute_fiducial_stats_from_options


def check_boxsize(stats=['mesh2_spectrum']):
    meas_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    boxsizes = {'LRG': [5000., 6000., 7000., 8000., 9000., 10000.],
                'ELG_LOP': [6000., 7000., 8000., 9000., 10000.],
                'QSO': [7000., 8000., 9000., 10000.]}
    for tracer in boxsizes:
        for zrange in tools.propose_fiducial('zranges', tracer):
             for region in ['NGC', 'SGC']:
                for boxsize in boxsizes[tracer]:
                    catalog_args = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region)
                    cellsize = 7.8
                    extra = f'boxsize{boxsize:0.f}_cellsize{cellsize:.1f}'
                    compute_fiducial_stats_from_options(stats, catalog=catalog_args, mattrs={'boxsize': boxsize, 'cellsize': cellsize},
                                                        get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir, extra=extra))

def check_nran(stats=['mesh2_spectrum']):
    meas_dir = Path(Path(os.getenv('SCRATCH')) / 'clustering-measurements-checks')
    nrans = {'LRG': [8, 9, 11, 18],
            'ELG_LOP': [11, 13, 16, 18],
            'QSO': [8, 9, 11, 18]}
    for tracer in nrans:
        for zrange in tools.propose_fiducial('zranges', tracer):
             for region in ['NGC', 'SGC']:
                for nran in nran[tracer]:
                    catalog_args = dict(version='holi-v1-altmtl', tracer=tracer, zrange=zrange, region=region, nran=nran)
                    extra = f'nran{nran:d}'
                    compute_fiducial_stats_from_options(stats, catalog=catalog_args,
                                                        get_measurement_fn=functools.partial(tools.get_measurement_fn, meas_dir=meas_dir, extra=extra))


if __name__ == '__main__':

    check_boxsize()
    check_nran()