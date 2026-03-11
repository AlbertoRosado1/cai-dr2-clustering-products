from full_shape.tools import generate_likelihood_options_helper, str_from_likelihood_options, get_likelihood


def test_str():
    likelihood_options = generate_likelihood_options_helper(tracer='LRG2')
    for level in [None, 1, 2, 3]:
        s = str_from_likelihood_options(likelihood_options, level=level)
        if level is None:
            assert s == 'LRG2-S2+LRG2-S3'
        elif level == 1:
            assert s == 'LRG2-S2-ell0-0.02-0.20-0.005-ell2-0.02-0.20-0.005-th-folpsD+LRG2-S3-ell000-0.02-0.12-0.005-ell202-0.02-0.08-0.005-th-folpsD+cov-holi-v1-altmtl'
    s = str_from_likelihood_options(likelihood_options, level={'select': 1})
    assert s == 'LRG2-S2-ell0-0.02-0.20-0.005-ell2-0.02-0.20-0.005+LRG2-S3-ell000-0.02-0.12-0.005-ell202-0.02-0.08-0.005'

    likelihood_options = generate_likelihood_options_helper(tracer='LRG3xELG1')
    for level in [None, 1, 2, 3]:
        s = str_from_likelihood_options(likelihood_options, level=level)
        if level is None:
            assert s == 'LRG3xELG1-S2+LRG3xELG1-S3'
        elif level == 1:
            assert s == 'LRG3xELG1-S2-ell0-0.02-0.20-0.005-ell2-0.02-0.20-0.005-th-folpsD+LRG3xELG1-S3-ell000-0.02-0.12-0.005-ell202-0.02-0.08-0.005-th-folpsD+cov-holi-v1-altmtl'
    s = str_from_likelihood_options(likelihood_options, level={'select': 1})
    assert s == 'LRG3xELG1-S2-ell0-0.02-0.20-0.005-ell2-0.02-0.20-0.005+LRG3xELG1-S3-ell000-0.02-0.12-0.005-ell202-0.02-0.08-0.005'


def test_likelihood():
    likelihood_options = generate_likelihood_options_helper(tracer='LRG2')
    likelihood = get_likelihood([likelihood_options], cosmo=None, fiducial=None, cache_dir='./_tests/')
    likelihood()


if __name__ == '__main__':

    test_str()