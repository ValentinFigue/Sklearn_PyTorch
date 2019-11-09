# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='source',
    version='0.1.0',
    description='Sample package for Python-Guide.org',
    long_description=readme,
    author='Valentin Figué',
    author_email='valentin.figue@polytechnique.edu',
    url='https://github.com/ValentinFigue/Sklearn-PyTorch',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

