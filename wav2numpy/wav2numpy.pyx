cimport cython
from libc.stdlib cimport calloc, free, abs
from libc.stdint cimport int16_t, int32_t, uint32_t
from libc.stdio cimport *
from libc.string cimport strerror
from libc.errno cimport errno
import numpy as np
cimport numpy as cnp

DEF AUDIO_OFFSET = 44  # Byte where audio starts, i.e. end of WAV header data
DEF READ_BUF_SIZE = 32768 / 2  # 32k L1 cache / sizeof int16
DEF AUDIO_THRESHOLD = (8000 / 1000) * 5  # 5 times the height of a sample
DEF ULAW_BITRATE = 8000
DEF WINDOW_SIZE = 10  # Seconds


# noinspection PyPep8Naming,PyClassicStyleClass
cdef class _finalizer:
    cdef void *_data

    def __dealloc__(self):
        if self._data is not NULL:
            free(self._data)


cdef void set_base(cnp.ndarray arr, void *carr):
    cdef _finalizer f = _finalizer()

    f._data = <void*>carr
    cnp.set_array_base(arr, f)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int16_t* read_mono_wave(char *filename, long *audio_size) except NULL:
    # TODO confirm that file is WAVE and has 1 channel
    cdef size_t int16_size = sizeof(int16_t)

    cdef FILE* fh = fopen(filename, 'rb')
    if fh == NULL:
        fprintf(stderr, "Can't open file %s for reading\n", filename)
        return NULL

    # Get size of new audio data structure
    cdef int fh_result = 0
    fh_result = fseek(fh, 0, SEEK_END)
    if fh_result != 0:
        fprintf(stderr, "Seek to end of file %s failed: %s\n", filename, strerror(errno))
        return NULL

    cdef uint32_t file_size = ftell(fh)

    # Return to start of audio data
    fh_result = 0
    fh_result = fseek(fh, AUDIO_OFFSET, SEEK_SET) # offset of audio data
    if fh_result != 0:
        fprintf(stderr, "Seek to start of audio in file %s failed: %s\n", filename, strerror(errno))
        return NULL

    # in Cython, array syntax is used to deref pointers
    # Size of audio minus header / size of data elements
    audio_size[0] = (file_size - AUDIO_OFFSET) / int16_size

    cdef int16_t* audio = <int16_t*>calloc(audio_size[0], int16_size)
    if audio == NULL:
        fprintf(stderr, "Can't allocate audio array of size %ld\n", audio_size[0])
        return NULL

    cdef:
        int16_t* array_ptr = audio
        uint32_t items_read = 0
        int items_to_read

    # Move array pointer through file
    while items_read < audio_size[0]:
        if items_read + READ_BUF_SIZE > audio_size[0]:
            items_to_read = audio_size[0] - items_read
        else:
            items_to_read = READ_BUF_SIZE

        items_read += fread(array_ptr + items_read, int16_size, items_to_read, fh)

        if items_read < items_to_read:
            fprintf(stderr, "File read error: requested %d, got %d", items_to_read, items_read)
            free(audio)
            return NULL

    fh_result = 0
    fh_result = fclose(fh)
    if fh_result != 0:
        fprintf(stderr, "Can't close file %s: %s\n", filename, strerror(errno))
        free(audio)
        return NULL

    return audio


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int16_t* read_stereo_channel(char *filename, int channel, long *audio_size) except NULL:
    # TODO confirm that the file is WAVE and has 2 channels
    cdef FILE* fh = fopen(filename, 'rb')
    if fh == NULL:
        fprintf(stderr, "Can't open file %s for reading\n", filename)
        return NULL

    # Get size of new audio data structure
    cdef int fh_result = 0
    fh_result = fseek(fh, 0, SEEK_END)
    if fh_result != 0:
        fprintf(stderr, "Seek to end of file %s failed: %s\n", filename, strerror(errno))
        return NULL

    cdef uint32_t file_size = ftell(fh)

    # Return to start of audio data
    fh_result = 0
    fh_result = fseek(fh, AUDIO_OFFSET, SEEK_SET) # offset of audio data
    if fh_result != 0:
        fprintf(stderr, "Seek to start of audio in file %s failed: %s\n", filename, strerror(errno))
        return NULL


    # in Cython, array syntax is used to deref pointers
    # Size of audio minus header / 2 (only getting one channel) / size of data elements
    audio_size[0] = ((file_size - AUDIO_OFFSET) / 2) / sizeof(int16_t)

    cdef int16_t* audio = <int16_t*>calloc(audio_size[0], sizeof(int16_t))
    if audio == NULL:
        fprintf(stderr, "Can't allocate audio array of size %ld\n", audio_size[0])
        return NULL

    cdef:
        # Create an array of the requested size
        int i = 0
        int j = 0
        int16_t frame[READ_BUF_SIZE]
        size_t bytes_read

    while True:
        bytes_read = fread(&frame, sizeof(int16_t), READ_BUF_SIZE, fh)
        if bytes_read == 0:
            break

        # Step through every other frame
        for j in range(channel, bytes_read, 2):
            if i > audio_size[0]:
                fprintf(stderr, "Attempt to read past end of allocated audio array\n", NULL)
                free(audio)
                return NULL

            audio[i] = frame[j]
            i += 1

    fh_result = 0
    fh_result = fclose(fh)
    if fh_result != 0:
        fprintf(stderr, "Can't close file %s: %s\n", filename, strerror(errno))
        free(audio)
        return NULL

    return audio


def read_nparray_stereo_channel(filename, int channel):
    """
    Read a single channel of a stereo WAV file into a Numpy Array
    """
    cdef long audio_size = 0
    cdef int16_t *audio_a = read_stereo_channel(filename.encode('utf-8'), channel, &audio_size)

    if audio_size == 0:
        free(audio_a)
        raise ValueError("No audio in given channel of given file")

    cdef int16_t[::1] audio_mv = <int16_t[:audio_size]>audio_a
    cdef cnp.ndarray audio = np.asarray(audio_mv)
    set_base(audio, audio_a)

    return audio


def read_nparray_mono(filename):
    """
    Read a mono WAV file into a Numpy array
    """
    cdef long audio_size = 0
    cdef int16_t *audio_a = read_mono_wave(filename.encode('utf-8'), &audio_size)

    if audio_size == 0:
        free(audio_a)
        raise ValueError("No audio in given file")

    cdef int16_t[::1] audio_mv = <int16_t[:audio_size]>audio_a
    cdef cnp.ndarray audio = np.asarray(audio_mv)
    set_base(audio, audio_a)

    return audio


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef int32_t find_noise_start(int16_t[:] audio):
    """
    Returns the index of the Numpy array where the noise seems to start
    (i.e. is above the hardcoded threshold) or -1 if unable to find any noise

    :param audio: `numpy.ndarray`
    """
    cdef uint32_t i
    for i in range(0, audio.shape[0]-3, 3):
        if (audio[i] + audio[i+1] + audio[i+2]) / 3 >= AUDIO_THRESHOLD:
            return abs(i)

    # Failed to find start of noise
    return -1


cdef inline uint32_t sec2sample(uint32_t sec, uint32_t fs):
    return sec * fs * 2


cpdef uint32_t find_window_end(int16_t[:] audio, uint32_t window_start, uint32_t window_size):
    """
    Given an array of audio data, a window start index, and a window size, return the index
    where the audio window ends

    :param audio: `numpy.ndarray`
    :param int window_start: index where the window starts
    :param int window_size: size of the window in seconds
    """
    cdef uint32_t samples = sec2sample(window_size, ULAW_BITRATE)

    if (samples + window_start) > audio.shape[0]:
        return audio.shape[0]  # Return end-of-file
    else:
        return samples + window_start

