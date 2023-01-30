from setuptools import find_packages, setup
setup(
    name='graph-transformer',
    long_description='This is the implementation of Graph Transformer (https://www.ijcai.org/proceedings/2021/0214.pdf)',
    packages=find_packages(include=['graph-transformer']),
    version='0.1.0',
    description='Graph Transformer Model',
    author='Willy Fitra Hendria',
    author_email = 'willyfitrahendria@gmail.com',
    url = 'https://github.com/willyfh/graph-transformer',
    keywords = ['deep learning','transformers','graph neural networks'],
    license='MIT',
    install_requires=['torch>=1.6', 'numpy>=1.8'],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)