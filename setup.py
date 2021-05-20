import setuptools

setuptools.setup(
  name='noneq_opt',
  version='0.1',
  url='https://github.com/mc2engel/noneq_opt',
  license='MIT',
  packages=setuptools.find_packages(),
  install_requires=[
    # Remove restriction once TFP is fixed.
    "jax<=0.2.11",
    "jax-md",
    "matplotlib",
    "numpy",
    "pandas",
    "pytest",
    "scipy",
    "seaborn",
    "distrax"
  ]
)
