from distutils.core import setup
import cmasf

setup(
    name='cmasf',
    version=cmasf.__version__,
    packages=['cmasf'],
    url='',
    license='',
    author='G. Golyshev',
    install_requires=['pandas', 'sqlalchemy', 'numpy', 'scipy'],
    author_email='g.golyshev@forecast.ru',
    description='CMASF functions and classes'
)
