
from distutils.core import setup

setup(name='blob_types',
      version='0.1',
      author='Steffen Kampmann',
      author_email='steffen.kampmann@smail.h-brs.de',
      url='https://github.com/abbgrade/blob_types',
      packages=['blob_types'],
      requires=['pyopencl (>= 2013.1)', 'numpy', 'pytools']
)
