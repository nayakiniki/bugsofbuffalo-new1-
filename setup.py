from setuptools import setup, find_packages

setup(
    name="bugs-of-buffalo",
    version="1.0.0",
    description="AI-powered Indian cattle and buffalo breed identification system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Bugs of Buffalo Team",
    author_email="team@bugsofbuffalo.com",
    url="https://github.com/nayakiniki/bugs-of-buffalo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Agriculture",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.22.0",
        "tensorflow>=2.10.0",
        "Pillow>=9.3.0",
        "requests>=2.28.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "opencv-python>=4.6.0",
        "python-multipart>=0.0.5",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "ml": [
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "seaborn>=0.11.0",
            "tqdm>=4.62.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bugs-train=ml-model.train:main",
        ],
    },
    include_package_data=True,
    package_data={
        "streamlit-app": ["assets/example_images/*"],
    },
)
