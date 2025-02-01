from setuptools import setup, find_packages

setup(
    name='projects',
    version='0.0.0',
    packages=find_packages(include=['projects', 'projects.*'])
)