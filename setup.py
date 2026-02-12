"""
Setup configuration for Brain-to-Text Mamba Decoder package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="brain-to-text-mamba-decoder",
    version="1.0.0",
    author="Brain-to-Text Team",
    description="Production-ready neural signal to text decoder with Mamba and GRU architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/brain-to-text-mamba-decoder",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "editdistance>=0.8.1",
    ],
    extras_require={
        "mamba": [
            "mamba-ssm>=1.1.0",
            "causal-conv1d>=1.2.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "kaggle": [
            "kagglehub>=0.1.0",
        ],
        "huggingface": [
            "huggingface-hub>=0.16.0",
            "transformers>=4.30.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipython>=8.10.0",
            "matplotlib>=3.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "benchmark-btt=scripts.benchmark_latency:run_all_benchmarks",
        ],
    },
)
