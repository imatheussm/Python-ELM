from distutils.core import setup

setup(
    name='elm',
    version='0.4.1',
    description='Extreme Learning Machine',
    author='masaponto',
    author_email='masaponto@gmail.com',
    url='masaponto.github.io',
    install_requires=['scikit-learn>=0.18.1', 'numpy'],
    packages=['elm'],
    package_dir={'': 'src'},
)
