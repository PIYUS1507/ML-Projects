from setuptools import find_packages,setup
from typing import List

E_DOT='-e .'

def get_requirement(file_path:str)->List[str]:
    '''
    this function will retuen the list of requirement
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("/n","") for req in requirements]
        if E_DOT in requirements:
            requirements.remove(E_DOT)
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Piyush',
    author_email='piyushrajurkar205@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirement.txt')
)