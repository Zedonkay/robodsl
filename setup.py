from setuptools import setup, find_packages

setup(
    name="robodsl",
    version="0.1.0",
    description="A DSL for GPU-accelerated robotics applications with ROS2 and CUDA",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
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
)
