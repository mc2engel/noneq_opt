import setuptools

setuptools.setup(
  name='noneq_opt',
  version='0.1',
  url='https://github.com/mc2engel/noneq_opt',
  license='MIT',
  packages=setuptools.find_packages(),
  install_requires=["numpy", "scipy", "pandas", "matplotlib", "seaborn", "tensorflow_probability", "jax"]
)
