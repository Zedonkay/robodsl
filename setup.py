from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="robodsl",
    version="0.1.0",
    description="A DSL for GPU-accelerated robotics applications with ROS2 and CUDA",
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    author="Ishayu Shikhare",
    author_email="ishikhar@andrew.cmu.edu",
    url="https://github.com/Zedonkay/robodsl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "jinja2>=3.0.0",
    ],
    entry_points={
        'console_scripts': [
            'robodsl = robodsl.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Code Generators',
        'Topic :: Scientific/Engineering',
    ],
)
