#!/usr/bin/env python

import os

from setuptools import setup, find_packages

setup(name='fslks',
      version='0.0.1-SNAPSHOT',
      author='Travis R. Goodwin, Max Savery',
      author_email='travis.goodwin@nih.gov',
      description='Implementation of Few-short Learning with the Kitchen Sink for Consumer Health Answer Generation',
      license='MIT',
      keywords='tensorflow deep-learning machine-learning question-answering few-shot-learning',
      long_description=os.open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
      install_requires=open(os.path.join(os.path.dirname(__file__), 'requirements.txt')).read(),
      packages=find_packages()
      )
