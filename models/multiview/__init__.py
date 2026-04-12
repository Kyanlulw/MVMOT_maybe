# ------------------------------------------------------------------------
# Multi-View MOTR package exports
# ------------------------------------------------------------------------

from importlib import import_module


def _load_module():
    return import_module('.multiview', __name__)


def build(*args, **kwargs):
    return _load_module().build(*args, **kwargs)


build_multiview_motr = build


def __getattr__(name):
    if name == 'MultiviewMOTR':
        return _load_module().MultiviewMOTR
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = ['MultiviewMOTR', 'build', 'build_multiview_motr']
