from setuptools import setup, find_packages
import sys

def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()

setup(
    name='across',
    version='1.0',
    description='A framework to convert between different sensor modalities.',
    author='Wadhah Zai El Amri, Malte Kuhlmann, Nicolás Navarro-Guerrero',
    author_email='wadhah.zai@l3s.de',
    packages=find_packages(),
    extras_require={
        'pipeline': read_requirements('requirements/pipeline_requirements.txt'),
    },
    python_requires='>=3.7',
)