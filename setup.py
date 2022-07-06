import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepwatermap",
    version="0.0.2",
    author="Example Author",
    author_email="author@example.com",
    description="deepwatermap",
    long_description=long_description,
    long_description_content_type="text/markdown",        
    entry_points={
        "console_scripts": [
            "dwm_inference=deepwatermap.inference:main"
        ]
    },
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
)
