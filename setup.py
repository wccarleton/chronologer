from setuptools import setup, find_packages

setup(
    name="chronologer",
    version="0.1.0",
    description="Bayesian Radiocarbon Calibration, Time-Series Modeling, and other inference involving data with chronological uncertainty",
    author="W. Christopher Carleton",
    author_email="ccarleton@protonmail.com",
    url="https://github.com/wccarleton/chronologer",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pymc",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
