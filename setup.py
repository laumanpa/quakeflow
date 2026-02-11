from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quakeflow",
    version="1.0.0",
    author="QuakeFlow Team",
    author_email="example@email.com",
    description="A comprehensive template matching and magnitude estimation pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quakeflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "quakeflow=quakeflow.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "quakeflow": ["examples/*.yaml"],
    },
)