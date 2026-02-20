from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="scagenet",
    version="1.0.0",
    description="scAgeNet: Aging program profiler for single-cell RNA-seq",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mikal El Hajjar",
    url="https://github.com/mikalelh/scAgeNet",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "scagenet": ["data/*"],
    },
    python_requires=">=3.8",
    install_requires=[
        "scanpy>=1.9",
        "torch>=1.12",
        "numpy>=1.21",
        "pandas>=1.4",
        "matplotlib>=3.5",
        "seaborn>=0.12",
        "scipy>=1.9",
        "scikit-learn>=1.1",
        "anndata>=0.8",
        "tqdm",
    ],
    extras_require={
        "go": ["gprofiler-official>=1.0"],
    },
    entry_points={
        "console_scripts": [
            "scagenet=scagenet.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
)
