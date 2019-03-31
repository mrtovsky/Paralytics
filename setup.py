from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='paralytics',
    version='0.2.2',
    author='Mateusz Zakrzewski',
    author_email="paralytics@gmail.com",
    description='Python Analytical Scripts.',
    long_description=open('README.rst').read(),
    url='https://mrtovsky.github.io/Paralytics/',
    install_requires=requirements,
    license='MIT',
    packages=find_packages('.'),
    zip_safe=True
)
