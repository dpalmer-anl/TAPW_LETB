from setuptools import setup, find_packages

setup(
        name='mtbmtbg',
        version='0.0.2',
        author="Wangqian Miao and Chu Li and Ding Pan and Xi Dai",
        author_email="dpalmer3@illinois.edu",
        description="TAPW method for twisted bilayer graphene",
        long_description="TAPW for twisted bilayer graphene",
        url="https://github.com/zybbigpy/TAPW",
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        python_requires=">=3.6, <4",
        install_requires = ["numpy", "scipy", "pandas", "h5py", "ase", "pythtb"],
)