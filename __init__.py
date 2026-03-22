import lichtfeld as lf
from .panel import SAM3MaskPanel

_classes = [SAM3MaskPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
