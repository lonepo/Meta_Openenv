"""Setup configuration for the circuitsynth package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="circuitsynth",
    version="1.0.0",
    author="CircuitSynth Authors",
    description=(
        "OpenEnv-compliant RL environment for electronic circuit synthesis "
        "using ngspice SPICE simulation"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    install_requires=install_requires,
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
        "dtw": ["fastdtw>=0.3.4"],
        "sb3": ["stable-baselines3>=2.0"],
    },
    entry_points={
        "console_scripts": [
            "circuitsynth-baseline=scripts.baseline_inference:main",
            "circuitsynth-eval=scripts.evaluate:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    package_data={"circuitsynth": ["*.yaml"]},
)
