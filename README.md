# Applications of Physics-Informed Neural Networks (PINNs)

## Introduction

This repository contains material for the workshop on physics-informed neural network (PINNs) applications.

PINNs are deep learning models recently proposed as an alternative method for solving direct or inverse problems involving mathematical models of a physical problem [(Raissi *etal.* 2019)](https://www.sciencedirect.com/science/article/pii/S0021999118307125), [(Karniadakis *etal.* 2021)](https://www.nature.com/articles/s42254-021-00314-5). They rely on the fundamental universal approximation theorem, which shows that, under certain architectures, artificial neural networks have the ability to accurately approximate different nonlinear functions (or operators) [(Hornik, 1991)](https://www.sciencedirect.com/science/article/pii/089360809190009T?via%3Dihub),[(Barron, 1993)](https://ieeexplore.ieee.org/document/256500),[(Villota 2019)](https://investigacion.unirioja.es/documentos/5fbf7e47299952682503c2fa/). In addition, the incorporation of automatic differentiation [(Baydin *etal.*, 2018)](https://arxiv.org/abs/1502.05767), outlines PINNs as an innovative option for the solution of complex physical models without the need of big amount of data. 


## Organizers

 - [Rodrigo Salas, Dr. Eng.](https://sites.google.com/uv.cl/rodrigo-salas)
 - [David Ortiz, Ph. D.](https://github.com/dortiz5)

If you have questions, comments or recommendations, you can send us an email 📧 at any of these addresses:

Rodrigo Salas [rodrigo.salas@uv.cl](mailto:rodrigo.salas@uv.cl)

David Ortiz [david.ortiz.puerta@gmail.com](mailto:david.ortiz.puerta@gmail.com) | [dortiz5@uc.cl](mailto:dortiz5@uc.cl)


## Schedule
The workshop will be held on Friday 24 May 2024, between 14 and 18hrs (CLT). 

| Time          | Activity |
| ------------- | --------- |
| 14:00 – 15:15 | Welcome and introduction to applications|
| 15:15 – 16:00 | Computational activity 1: ANN vs. PINNs |
| 16:00 – 16:15 | Coffee Break |
| 16:15 – 17:15 | Computational activity 2: Forward applications |
| 17:15 – 17:20 | Small break |
| 17:20 – 17:55 | Computational activity 3: Inverse applications |


## Contents

- [Learning Objectives](#Learning-objectives)
- [Some related material](#Some-related-material)
- [Preparation and prerequisites](#preparation-and-prerequisites)
- [Installation and setup](#installation-and-setup)

## Learning objectives
The key learning objective from this workshop could be simply summarised as:

_Provide basic tools to develop PINNs for solving various physical models using [PyTorch](https://pytorch.org/)._

More specifically we aim to:

 - provide an understanding of the applications of PINNs as presented in the literature,
 - introduce the differences and comparisons between traditional neural networks (NN) and physics-informed neural networks (PINNs),
 - explore the formulation and solution of direct and inverse problems using PINNs in 1D and 2D models, and
 - discuss advanced practices and techniques in developing and optimizing PINNs.


## Some related material
Some interesting videos and material for further studyes

- Previous workshop by [Prof. Ph.D. Francisco Sahli](https://fsahli.github.io/): [Workshop on April](https://fsahli.github.io/PINN-notes/)

- Neural networks: [Interesting video series by 3Blue1Brown about neural networks and machine learning](https://www.3blue1brown.com/topics/neural-networks)

- Automatic differentiation. Here you can find 3 links about automatic differentiation and dual numbers: [link 1](https://thenumb.at/Autodiff/), [link 2](https://blog.demofox.org/2014/12/30/dual-numbers-automatic-differentiation/), [link 3](https://en.wikipedia.org/wiki/Dual_number)

- Autograd

- [Physics Informed Neural Network for Computer Vision and Medical Imaging [1]](https://collab.dvb.bayern/display/TUMdlma/Physics+Informed+Neural+Network+for+Computer+Vision+and+Medical+Imaging). From this last link, we highlight the following table with different theoretical methods and applications:

![Table from [1]](data/figures/methods_and_applications.png)


## Instalation and setup
There are three options for participating in this workshop, with instructions provided below:

 - via a local install
 - on Google Colab
 - on Binder

We recommend the local install approach, especially if you forked the repository, as it is the easiest way to keep a copy of your work and push it back to GitHub.

However, if you encounter issues with the installation process or are unfamiliar with the terminal/installation process, you have the option to run the notebooks on Google Colab or Binder.


### Local Install

Recomendamos usar ``conda`` para instalar los paquetes necesarios para
este tutorial.

Tenga en cuenta también que este tutorial está escrito para Python 3.X.


Cree un entorno conda usando el archivo ``pinn-tutorial.yml`` en la ruta
del repositorio usando

```console
conda env create -f pinn-tutorial.yml
```

Esto creará un entorno conda llamado "pinn-tutorial" con todos los
paquetes requeridos.

Puedes activar el entorno con

```console
conda activate pinn-tutorial
```

## Comprobando la instalación

Después de la instalación puedes comprobar si todo está instalado.

```console
python probar_instalacion.py
```

Para comprobar si todo funciona, ejecute las demostraciones con

```console
python demo.py
```

## Licencia

Todo el código está bajo licencia MIT y el contenido bajo licencia Creative Commons Attribute.

El contenido de este repositorio está bajo licencia bajo la
[Licencia Creative Commons Attribution 4.0](http://choosealicense.com/licenses/cc-by-4.0/),
y el código fuente que acompaña al contenido tiene 
[Licencia MIT](https://opensource.org/licenses/mit-license.php).



### Local Install

### Google Colab

### Binder

## References
