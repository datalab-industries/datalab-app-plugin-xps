from datalab_app_plugin_xps import XPSBlock, __version__


def test_version():
    assert __version__


def test_xps_block():
    block = XPSBlock
    assert block.version == __version__
