
from distutils.core import setup

from blob_types import __version__, __author__, __email__, __license__

setup(name='blob_types',
      version=__version__,
      author=__author__,
      author_email=__email__,
      license=__license__,
      url='https://github.com/abbgrade/blob_types',
      packages=['blob_types'],
      requires=['pyopencl (>= 2013.1)', 'numpy', 'pytools']
)
