from setuptools import setup, find_packages

setup(
    name='jw_ice',
    version='0.1',
    description='Ice opacity calculator for JWST',
    author="Klaus Pontoppidan",
    author_email="pontoppi@stsci.edu",
    packages={"jw_ice","OCs"},
    package_data={
        "OCs" : ["*.asc","*.txt","*.NK","*.lnk"], 
        "jw_ice" : ["*.json"]},
#    data_files=[('', ['config.json']),],
    include_package_data = True,
    install_requires=["PyMieScatt"],
    )

    
