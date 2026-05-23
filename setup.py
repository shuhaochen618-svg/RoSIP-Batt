from setuptools import setup, find_packages

setup(
    name="RoSIP-Batt",
    version="1.0.0",
    description="RoSIP-Batt: Rotary SOH-Injected Prior Battery Transformer for Joint SOH and RUL Prediction",
    long_description=open("README.md", encoding="utf-8").read() if open("README.md", encoding="utf-8") else "",
    long_description_content_type="text/markdown",
    author="Shuhao Chen",
    author_email="2023333541008@mails.zstu.edu.cn",
    url="https://github.com/shuhaochen618-svg/RoSIP-Batt",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.3",
        "numpy>=1.24",
        "scipy>=1.11",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "PyYAML>=6.0",
        "tqdm>=4.65",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
