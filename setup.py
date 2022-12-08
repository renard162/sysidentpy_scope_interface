# -*- coding: utf-8 -*-
import setuptools

with open(file='./requirements.txt', mode='r', encoding='utf-8') as file:
    required = file.read().splitlines()

setuptools.setup(
    name="sysidentpy_scope",
    version="0.1.0",
    license='MIT',
    author="Samuel Carlos Pessoa Oliveira",
    author_email="samuelcpoliveira@gmail.com",
    description="Scope de Identificação de sistemas em modelos NARX polinomiais para RaspberryPi 3 B+",
    long_description="Scope de Identificação de sistemas em modelos NARX polinomiais para RaspberryPi 3 B+",
    long_description_content_type="text/markdown",
    url="https://github.com/renard162/TCC_engenharia_eletrica",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    python_requires='>=3.6'
)
