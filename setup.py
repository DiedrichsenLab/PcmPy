from distutils.core import setup
setup(
  name = 'PCMPy',        
  packages = ['PCMPy'],  
  version = '0.2',      
  license='MIT', 
  description = 'Modeling of multivariate activity patterns',
  author = 'Jörn Diedrichsen',
  author_email = 'joern.dierichsen@googlemail.com',      
  url = 'https://github.com/DiedrichsenLab/PCMPy',  
  download_url = 'https://github.com/DiedrichsenLab/PCMPy/archive/v_01.tar.gz', 
  keywords = ['statistics', 'imaging analysis', 'multivariate'],
  install_requires=[            # I get to this in a second
          'validators',
          'beautifulsoup4',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)