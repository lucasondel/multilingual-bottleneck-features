from setuptools import setup, find_packages

setup(
    author='Brno University of Technology',
    description='Multilingual Bottleneck Features extractor',
    name='mbn-extractor',
    author_email='iondel@fit.vutbr.cz',
    license='Creative Commons',
    install_requires=[
        'numpy',
        'scipy',
        'numexpr'
    ],
    version=1.0,
    packages=['mbnextractor'],
    scripts=['mbnextractor/audio2bottleneck'],
    data_files=[
        ('mbnextractor/models', ['mbnextractor/models/FisherEnglish_FBANK_HL500_SBN80_PhnStates120.npz']),
        ('mbnextractor/models', ['mbnextractor/models/FisherEnglish_SBN80_triphones2423.npz']),
        ('mbnextractor/models', ['mbnextractor/models/Babel-ML17_SBN80_PhnStates3096.npz']),
    ]
)

