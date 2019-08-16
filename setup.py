from setuptools import setup, find_packages
import sys, os.path

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = "0.1.1"

setup(name='fast_rl',
      version=VERSION,
      description='Fastai for computer vision and tabular learning has been amazing. One would wish that this would '
                  'be the same for RL. The purpose of this repo is to have a framework that is as easy as possible to '
                  'start, but also designed for testing new agents. ',
      url='https://github.com/josiahls/fast-reinforcement-learning',
      author='Josiah Laivins',
      author_email='jokellum@northstate.net',
      long_description=long_description,
      long_description_content_type="text/markdown",
      license='',
      packages=find_packages(),
      zip_safe=False,
      install_requires=['numpy', 'tqdm', 'pillow', 'pandas', 'fastai', 'gym[box2d, atari]', 'jupyter', 'namedlist',
                        'pytest-asyncio', 'pytest'],
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
)
