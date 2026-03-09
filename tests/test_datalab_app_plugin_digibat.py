from datalab_app_plugin_digibat import __version__
from datalab_app_plugin_digibat.blocks import XPSBlock


def test_version():
    assert __version__


def test_xps_block():
    block = XPSBlock
    assert block.version == __version__
