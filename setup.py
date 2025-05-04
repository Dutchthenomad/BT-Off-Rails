from setuptools import setup, find_packages

setup(
    name="bt-off-rails",
    version="0.1.0",
    description="Cryptocurrency backtesting framework with RSI divergence strategies",
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
