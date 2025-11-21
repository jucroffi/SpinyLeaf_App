""" Minimal folder config."""
minimal = {
    'APERTURE': True,
    'APERTURE-GROUP': False,
    'SCENE': True,
    'GRID': False,
    'VIEW': False,
    'INTERIOR-APERTURE-GROUP': False,
    'BSDF': False,
    'IES': False,
    'DYNAMIC-SCENE': False,
    'INDOOR-DYNAMIC-SCENE': False,
    'RECEIVER': False
}

""" Config for model with aperture groups."""
aperture_groups = {
    'APERTURE': True,
    'APERTURE-GROUP': True,
    'SCENE': True,
    'GRID': False,
    'VIEW': False,
    'INTERIOR-APERTURE-GROUP': True,
    'BSDF': False,
    'IES': False,
    'DYNAMIC-SCENE': False,
    'INDOOR-DYNAMIC-SCENE': False,
    'RECEIVER': True
}

"""Full folder with all subfolders."""
full = {k: True for k in minimal}
