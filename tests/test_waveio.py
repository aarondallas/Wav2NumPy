import pytest
import numpy
from os import path
from waveio import waveio


@pytest.fixture(scope="module")
def fixture_dir():
    return path.join(path.dirname(__file__), 'fixtures')


# noinspection PyShadowingNames
def test_read_stereo(fixture_dir):
    filename = path.join(fixture_dir, 'stereo_phone_recording.wav')
    result = waveio.read_nparray_stereo_channel(filename, 0)

    assert isinstance(result, numpy.ndarray)
    # assert result.shape == () TODO
    # assert waveio.find_noise_start(result) == x TODO


# noinspection PyShadowingNames
def test_read_mono(fixture_dir):
    filename = path.join(fixture_dir, 'mono_phone_recording.wav')
    result = waveio.read_nparray_mono(filename)

    assert isinstance(result, numpy.ndarray)
    # assert result.shape == () TODO
    # assert waveio.find_window_end(result, x, x) == x TODO
