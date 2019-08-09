from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fast_rl'))

VERSION = 0.1

setup_py_dir = os.path.dirname(os.path.realpath(__file__))

setup(name='fast_rl',
      version=VERSION,
      description='Fastai for computer vision and tabular learning has been amazing. One would wish that this would '
                  'be the same for RL. The purpose of this repo is to have a framework that is as easy as possible to '
                  'start, but also designed for testing new agents. ',
      url='https://github.com/josiahls/fast-reinforcement-learning',
      author='Josiah Laivins',
      author_email='jlaivins@uncc.edu',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('fast_rl')],
      zip_safe=False,
      install_requires=['numpy', 'tqdm', 'pillow', 'pandas', 'fastai', 'gym[box2d, atari]', 'jupyter', 'namedlist',
                        'pytest-asyncio', 'pytest'],
      )
