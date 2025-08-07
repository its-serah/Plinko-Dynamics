"""
Setup script for Plinko Dynamics package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="plinko-dynamics",
    version="2.0.0",
    author="Serah Rashidi",
    author_email="serah@example.com",
    description="Advanced Quantum Galton Board Simulation with AI-Enhanced Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/its-serah/Plinko-Dynamics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="quantum computing, galton board, plinko, simulation, AI, neural ODE",
    project_urls={
        "Bug Reports": "https://github.com/its-serah/Plinko-Dynamics/issues",
        "Source": "https://github.com/its-serah/Plinko-Dynamics",
        "Documentation": "https://github.com/its-serah/Plinko-Dynamics/blob/main/README.md",
    },
)
