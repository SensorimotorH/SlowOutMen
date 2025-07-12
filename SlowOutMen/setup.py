from setuptools import setup, find_packages
import os

setup(
    name="SlowOutMen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="SlouOutMen",
    author_email="zy2307508@buaa.edu.cn",
    description="SlowOutMen",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/jackypromaxplus/SlowOutMen",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)