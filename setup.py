from setuptools import find_packages,setup

from typing import List

HYPE_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return list of requirements
    '''

    

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPE_E_DOT in requirements:
            requirements.remove(HYPE_E_DOT)

    return requirements



setup(

    name = 'Monitoring_Portal',
    version = '0.0.1',
    author = 'Sanket',
    author_email= 'Sanketthakare.st5153@gmail.com',
    packages= find_packages(),
    #install_requires = ['pandas','numpy','seaborn']
    install_requires = get_requirements('requirements.txt'),
)

