import codecs
import pathlib
import re

from setuptools import find_packages, setup


PACKAGE_PATH = pathlib.Path(__file__).parent


def read(*paths):
    file = str(PACKAGE_PATH.joinpath(*paths).resolve())
    with codecs.open(file) as f:
        return f.read()


def find_version(*paths):
    version_file = read(*paths)
    version_pattern = re.compile(r"^__version__ = ['\"]([^'\"]*)['\"]", re.M)
    version_match = version_pattern.search(version_file)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError('Unable to find version string.')


with codecs.open('requirements.txt') as f:
    requirements = f.read().splitlines()

DESCRIPTION = 'Python analytical scripts that will overcome ' \
              'paralysis in your data analysis.'
LONG_DESCRIPTION = read('README.rst')
EXTRAS_REQUIREMENTS = {
    "all": ["selenium>=3.141.0", "seaborn>=0.9.0", "statsmodels>=0.9.0"],
    "browser": ["selenium>=3.141.0"],
    "plot": ["seaborn>=0.9.0"],
    "vif": ["statsmodels>=0.9.0"]
}

setup(
    name='paralytics',
    version=find_version('paralytics', '__init__.py'),
    author='Mateusz Zakrzewski, Łukasz Bala',
    author_email="paralytics@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url='https://mrtovsky.github.io/Paralytics/',
    install_requires=requirements,
    extras_require=EXTRAS_REQUIREMENTS,
    license='MIT',
    packages=find_packages('.'),
    zip_safe=True
)
