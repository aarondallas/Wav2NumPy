######
Wav2NumPy
######

Read Wave files directly into NumPy arrays

Synopsis
========

.. code:: python

    from wav2numpy import wav2numpy

    # Read the first (left) channel of a stereo file
    np_array = wav2numpy.read_nparray_stereo_channel('stereo_file.wav', 0)

    # Read a mono file
    np_array = wav2numpy.read_nparray_mono('mono_file.wav')

