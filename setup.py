from setuptools import setup


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='paralytics',
    version='0.1',
    author='Mateusz Zakrzewski',
    description='Python Analytical Scripts.',
    long_description=open('README.md').read(),
    url='https://github.com/mrtovsky/Paralytics',
    install_requires=requirements,
    license='MIT',
    packages=['paralytics'],
    zip_safe=False
)
