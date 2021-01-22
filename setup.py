import setuptools

setuptools.setup(
  name='noneq_opt',
  version='0.1',
  url='https://github.com/mc2engel/noneq_opt',
  license='MIT',
  packages=setuptools.find_packages(),
  install_requires=[
  "jax",
  "jax-md",
  "matplotlib",
  "numpy",
  "pandas",
  "pytest",
  "scipy",
  "seaborn",
  "tensorflow_probability",
  ]
)
