from distutils.core import setup
setup(
  name = 'PcmPy',
  packages = ['PcmPy'],
  version = '0.9.1',
  license='MIT',
  description = 'Modeling of multivariate activity patterns',
  author = 'Jörn Diedrichsen',
  author_email = 'joern.diedrichsen@googlemail.com',
  url = 'https://github.com/DiedrichsenLab/PCMPy',
  download_url = 'https://github.com/DiedrichsenLab/PcmPy/archive/refs/tags/v0.9.1.tar.gz',
  keywords = ['statistics', 'imaging analysis', 'multivariate'],
  install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'matplotlib',
          'seaborn'],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9'
  ],
  python_requires='>=3.6'
)