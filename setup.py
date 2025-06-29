from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="robodsl",
    version="0.1.0",
    description="A DSL for GPU-accelerated robotics applications with ROS2 and CUDA",
    long_description=read('README.md') if os.path.exists('README.md') else "A DSL for GPU-accelerated robotics applications with ROS2 and CUDA",
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
        "lark>=1.1.0",
        "onnxruntime>=1.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        'dev': [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        'cuda': [
            "onnxruntime>=1.15.0",
        ],
        'tensorrt': [
            "tensorrt>=8.6.0",
            "pycuda>=2022.2.0",
        ],
        'docs': [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        'all': [
            "robodsl[dev,cuda,tensorrt,docs]",
        ],
    },
    entry_points={
        'console_scripts': [
            'robodsl = robodsl.cli.cli:main',
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
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Code Generators',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    keywords="robotics, dsl, cuda, gpu, ros2, onnx, tensorrt, code-generation",
    project_urls={
        'Homepage': 'https://github.com/Zedonkay/robodsl',
        'Documentation': 'https://robodsl.readthedocs.io',
        'Repository': 'https://github.com/Zedonkay/robodsl.git',
        'Bug Tracker': 'https://github.com/Zedonkay/robodsl/issues',
    },
)
