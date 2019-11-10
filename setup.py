# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Sklearn_PyTorch',
    version='0.1.0',
    description='Sklearn_Pytorch package to use Sklearn model within Pytorch',
    long_description=readme,
    author='Valentin Figué',
    author_email='valentin.figue@polytechnique.edu',
    url='https://github.com/ValentinFigue/Sklearn_PyTorch',
    license=license,
    install_requires=requirements,
    packages=find_packages(exclude=('tests', 'docs'))
)

