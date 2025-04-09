from setuptools import setup, find_packages

setup(
    name="token-turing-machine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.1",
        "jaxlib>=0.4.1",
        "flax>=0.6.3",
        "optax>=0.1.4",
        "numpy>=1.22.0",
        "pytest>=7.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Implementation of Token Turing Machine",
    keywords="machine learning, neural networks, transformers, memory",
    url="https://github.com/yourusername/token-turing-machine",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
