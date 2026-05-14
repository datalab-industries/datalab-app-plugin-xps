from datalab_app_plugin_xps import XPSBlock, __version__

import pytest
from pathlib import Path

@pytest.fixture
def example_files(tmp_path):
    data_path = Path(__file__).parent / "data" / "In-718"
    return list(data_path.glob("*.VGD"))

def test_version():
    assert __version__


def test_xps_block(example_files):
    block = XPSBlock
    assert block.version == __version__

    block = XPSBlock(item_id="xps-block")
    block.plot_xps(filenames=example_files)
    plot = block.data["bokeh_plot_data"]
    assert plot is not None
