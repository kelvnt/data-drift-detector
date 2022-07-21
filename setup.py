import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data-drift-detector",
    version="0.0.11",
    author="Kelvin Tay",
    author_email="btkelvin@gmail.com",
    description="Compare differences between 2 datasets to identify data drift",
    url="https://github.com/kelvnt/data-drift-detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license="GPLv3",
    install_requires=[
        "category-encoders>=2.2.2",
        "matplotlib>=3.4.3",
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.24.1",
        "scipy>=1.5.4",
        "seaborn>=0.11.2"
    ],
    python_requires=">=3.6.0"
)
