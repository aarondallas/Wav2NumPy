from setuptools import setup
from Cython.Build import cythonize

# Read requirements
with open('requirements.txt', 'r') as fh:
    reqs = [str(x).strip() for x in fh.readlines()]

# Read version string
with open('wav2numpy/_version.py', 'r') as fh:
    for line in fh:
        if line.startswith('__version__'):
            exec(line)

# noinspection PyUnresolvedReferences
setup(
    name="wav2numpy",
    version=__version__,
    author='Aaron Dallas',
    description='NumPy Array Wave utilities',
    url='https://github.atl.pdrop.net/research/WaveIO',
    packages=['wav2numpy'],
    install_requires=reqs,
    ext_modules=cythonize('wav2numpy/wav2numpy.pyx'),
)

