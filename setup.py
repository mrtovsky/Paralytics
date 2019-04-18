import codecs
import pathlib
import re

from setuptools import find_packages, setup


parent_path = pathlib.Path(__file__).parent


def read(*files):
    file_path = str(parent_path.joinpath(*files).resolve())
    with codecs.open(file_path) as f:
        return f.read()


def find_version(*files):
    version_file = read(*files)
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
long_description = read('README.rst')

setup(
    name='paralytics',
    version=find_version('paralytics', '__init__.py'),
    author='Mateusz Zakrzewski',
    author_email="paralytics@gmail.com",
    description=DESCRIPTION,
    long_description=long_description,
    url='https://mrtovsky.github.io/Paralytics/',
    install_requires=requirements,
    license='MIT',
    packages=find_packages('.'),
    zip_safe=True
)
