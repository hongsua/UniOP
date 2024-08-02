from setuptools import setup, find_packages

VERSION = '0.1.0' 

setup(
    name='UniOP',
    version=VERSION,
    description='Genomic and Metagenomic sequences operon prediction',
    long_description=open('README.md').read(),
    url='https://github.com/hongsua/UniOP',
    author='Hong Su',
    author_email='hong.su@mpinat.mpg.de',
    py_modules=['UniOP.py'],
    install_requires=[
        'argparse',
        'pandas',
        'numpy',
        'datetime',
        'scikit-learn',
    ],

    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.11',
    entry_points={
        'console_scripts':[
            'uniop=UniOP:main',
        ],
    },
)


