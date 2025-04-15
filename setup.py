from setuptools import setup, find_packages

setup(
    name="moon-whales",
    version="0.1.0",
    description="Blockchain analysis and trading system",
    author="DutchTheNomad",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "ccxt",
        # Add other dependencies as needed
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
