from setuptools import setup, find_packages

from ftrosa import __version__

setup(
    name='ftrosa',
    version=__version__,

    url='https://github.com/jo-cho/ftrosa',
    author='Cheonghyo Cho',
    author_email='jhcho1016@gmail.com',

    py_modules=['ftrosa'],

    install_requires=['librosa>=0.9.2','numpy','pandas','matplotlib']
)
