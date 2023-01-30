from setuptools import find_packages, setup
setup(
    name='graph-transformer',
    description='This is the implementation of Graph Transformer (https://www.ijcai.org/proceedings/2021/0214.pdf)',
    packages=find_packages(include=['graph-transformer']),
    version='0.1',
    author='Willy Fitra Hendria',
    author_email = 'willyfitrahendria@gmail.com',
    url = 'https://github.com/willyfh/graph-transformer',
    download_url = 'https://github.com/willyfh/graph-transformer/archive/refs/tags/v0.1.tar.gz',
    keywords = ['deep learning','transformers','graph neural networks'],
    license='MIT',
    install_requires=['torch>=1.6', 'numpy>=1.8'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
