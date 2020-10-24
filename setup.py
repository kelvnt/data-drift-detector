import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data-drift-detector",
    version="0.0.2",
    author="Kelvin Tay",
    author_email="btkelvin@gmail.com",
    description="Compare differences between 2 datasets to identify data drift",
    url="https://github.com/kelvnt/data-drift-detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license="MIT license",
    install_requires=[
        "category-encoders>=2.2.2",
        "matplotlib>=3.3.0",
        "numpy>=1.19.0",
        "pandas==1.0.0",
        "scikit-learn>=0.23.0",
        "scipy>=1.5.2",
        "seaborn>=0.11.0"
    ],
    python_requires=">=3.6.0"
)
