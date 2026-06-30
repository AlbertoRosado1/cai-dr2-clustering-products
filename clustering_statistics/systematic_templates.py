
import numpy as np

import lsstypes as types


def include_systematic_templates(window: types.WindowMatrix, templates: dict, effects: tuple | list=('auw',)):
    """Include systematic templates in the window matrix."""
    observable = window.observable
    windows, theories, effects = [], [], []
    for effect in effects:
        if effect == 'auw':
            diff = templates['auw'].match(observable).value() - templates['raw'].match(observable).value()
            windows.append(diff[:, None])
            theories.append(types.ObservableLeaf(value=0., sigma=0.2))
            types.append(effect)
        elif effect == 'photo':
            diff_amr = templates['mock_wsys'].match(observable).value() - templates['data_nowsys'].match(observable).value()
            diff_wsys = templates['data_wsys'].match(observable).value() - templates['data_nowsys'].match(observable).value()
            diff_wsys -= diff_amr
            windows.append(diff_amr[:, None])
            theories.append(types.ObservableLeaf(value=1., sigma=0.01))
            types.append('amr')
            windows.append(diff_wsys[:, None])
            theories.append(types.ObservableLeaf(value=0., sigma=0.3))
            types.append('sys')
    theory = window.theory
    theory = types.ObservableTree([theory] + theories, types=['theory'] + types)
    window = window.clone(value=np.concatenate([window.value] + windows, axis=1), theory=theory)
    return window