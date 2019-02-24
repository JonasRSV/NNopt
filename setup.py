import setuptools

setuptools.setup(
    name="NNOpt",
    version="0.0.1",
    author="Jonas & Johan",
    author_email="jonas@valfridsson.net",
    description="Utils for warehouse simulator",
    url="https://github.com/kex2019/Utilities",
    packages=["nnopt"],
    install_requires=["numpy==1.14.2", "tensorflow==1.12", "tensorflow_probability"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
