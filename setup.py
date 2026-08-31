from setuptools import setup, find_packages

setup(
    name="datadna",
    version="1.0.0",
    author="Mehrdad Ourang",
    author_email="mehrdad87ourangg@gmail.com",
    description="Synthetic data quality evaluation with DNA fingerprint signatures",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mehrdad87org/datadna",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "click>=8.1",
        "pandas>=2.0",
        "numpy>=1.24",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "plotly>=5.18",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "rich>=13.0",
    ],
    entry_points={
        "console_scripts": [
            "datadna=datadna.cli:main",
        ],
    },
)
