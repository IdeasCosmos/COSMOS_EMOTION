from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cosmos-emotion",
    version="2.0.0",
    author="COSMOS EMOTION Team",
    author_email="sjpupro@gmail.com",
    description="차세대 감정 분석 시스템 - 음악 이론 + 양방향 전파 + 5채널 공명",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IdeasCosmos/COSMOS_EMOTION",
    packages=find_packages(),
    py_modules=[
        "api_server",
        "bidirectional_propagation",
        "dataset_integration",
        "integrated_cosmos_system",
        "morpheme_intensity_system",
        "quick_start_example",
        "resonance_system",
        "visualization_comparison",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "konlpy": ["konlpy>=0.6.0"],
        "neural": ["torch>=2.0.0", "transformers>=4.30.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=5.0.0"],
    },
)
