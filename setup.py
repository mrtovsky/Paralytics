from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='paralytics',
    version='0.2',
    author='Mateusz Zakrzewski',
    description='Python Analytical Scripts.',
    long_description=open('README.md').read(),
    url='https://github.com/mrtovsky/Paralytics',
    install_requires=requirements,
    license='MIT',
    packages=find_packages('.'),
    zip_safe=True
)
