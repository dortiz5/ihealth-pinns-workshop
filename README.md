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
Some interesting videos and material for further studies:

- Previous workshop by [Prof. Ph.D. Francisco Sahli](https://fsahli.github.io/): [Workshop on April](https://fsahli.github.io/PINN-notes/)

- Neural networks: [Interesting video series by 3Blue1Brown about neural networks and machine learning](https://www.3blue1brown.com/topics/neural-networks)

- Automatic differentiation. Here you can find 3 links about automatic differentiation and dual numbers: [link 1](https://thenumb.at/Autodiff/), [link 2](https://blog.demofox.org/2014/12/30/dual-numbers-automatic-differentiation/), [link 3](https://en.wikipedia.org/wiki/Dual_number). Also, here you can find a  [tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#a-gentle-introduction-to-torch-autograd) in 
PyTorch

- [Physics Informed Neural Network for Computer Vision and Medical Imaging [1]](https://collab.dvb.bayern/display/TUMdlma/Physics+Informed+Neural+Network+for+Computer+Vision+and+Medical+Imaging)

- Ben Moseley [personal blog](https://benmoseley.blog/)

## Teaching material

### Slides
ADD SLIDES

### Exercises and solutions
The exercises for this course are located in the [notebooks](notebooks/) directory, provided as partially completed Jupyter notebooks. Worked solutions can be found in the [solutions](solutions/) directory, and are intended for review after the course, in case you missed anything. 


## Preparation and prerequisites
To maximize the benefits of this session, we assume you have a basic understanding in certain areas and that you complete some preparatory work in advance. Below, we outline the expected knowledge and provide resources for further reading if needed.

### Mathematics and machine learning

Basic mathematics in:
 - Basic calculus - video series by [3Blue1Brown](https://www.youtube.com/playlist?list=PL0-GT3co4r2wlh6UHTUeQsrf3mlS2lk6x)
 - Ordinary differential equations (ODE) - video series by [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNPOjrT6KVlfJuKtYTftqH6)
 - Partial diferential equations (PDE) - video by [3Blue1Brown](https://www.youtube.com/watch?v=ly4S0oi3Yz8&ab_channel=3Blue1Brown)
 - Optimization

Artificial Neural Networks:
 - Some basic concepts

### Python
The course will be conducted in Python using PyTorch. While prior knowledge of PyTorch is not required, we assume participants are comfortable with the basics of Python 3. This includes:

 - basic mathematical operations
 - Writing and executing scripts/programs
 - Creating and using functions
 - Understanding the concept of object oriented programming [(OOP)](https://eli5.gg/Object-oriented%20programming)
 - familiarity with the following libraries:
    - [`numpy`](https://numpy.org/)  for mathematical and array operations
    - [`matplotlib`](https://matplotlib.org/) for plotting and visualization
    - [`PyTorch`](https://pytorch.org/) for high level training of ANN 
 - Understanding the concept of a [jupyter notebook](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/index.html)

Also, for the course, we recommend you to have the following:

- A text editor, such as [vim/neovim](https://neovim.io/), [gedit](https://gedit.en.softonic.com/), [vscode](https://code.visualstudio.com/), or [sublimetext](https://www.sublimetext.com/), [pycharm](https://www.jetbrains.com/pycharm/) to open and edit python code files.
- A terminal emulator, like [GNOME Terminal](https://help.gnome.org/users/gnome-terminal/stable/), [wezterm](https://wezfurlong.org/wezterm/index.html), [Windows Terminal (for Windows)](https://learn.microsoft.com/en-us/windows/terminal/), or [iTerm (for macOS)](https://iterm2.com/).
- We encourage you to install [miniconda](https://docs.anaconda.com/free/miniconda/index.html).


### git and GitHub
You will be expected to know how to
- clone or fork a repository,
- commit, and push.

The [workshop from the 2022 ICCS Summer School](https://www.youtube.com/watch?v=ZrwzK4CnJ3Q) 
should provide the necessary knowledge.

## Instalation and setup
There are three options for participating in this workshop, with instructions provided below:

 - via a local install
 - on [Google Colab](#google-colab)
 - on [binder](#binder)

We highly recommend the local install approach. However, if you encounter issues with the installation process or are unfamiliar with the terminal/installation process, you have the option to run the notebooks on [Google Colab](#google-colab) or [binder](#binder).


### Local Install
we will now explain how to perform the local installation using `conda`

#### 1. Clone or fork the repository
Navigate to the directory you want to install this repository on your system and clone via https by running:
```
git clone https://github.com/dortiz5/ihealth-pinns-workshop.git
```
This will create a directory `ihealth-pinns-workshop/` with the contents of this repository.

Please note that if you have a GitHub account and wish to save your work, we recommend [forking the repository](https://github.com/dortiz5/ihealth-pinns-workshop/fork) and cloning your fork, enabling you to push your changes and progress back to your fork for future reference.


#### 2. Installing miniconda
Installing conda is easy and it run in *Windows, macOS and Linux*. You just have to follow the [instructions](https://docs.anaconda.com/free/miniconda/miniconda-install/) on the website. **Make sure you test your installation!**

#### 3. Creating a conda environment
**Make sure you have conda installed**. This project has been package with a [`pinn-ihealth-tutorial.yml`](pinn-ihealth-tutorial.yml) to create and install the `python3` environment. 

In windows you have open the *Anaconda Prompt*, in macOS the *terminal application* and in Linux, a *terminal* window. Then, type the following code:

```
conda env create -f pinn-ihealth-tutorial.yml
```

This will create a `conda` enviroment named `pinn-ihealth-tutorial`. To activate it you just need to run:

```
conda activate pinn-ihealth-tutorial
```
to deactivate, you just need to run 

```
conda deactivate
```

#### 4. Run the notebook

From the current directory, launch the jupyter notebook server:
```
jupyter notebook
```
This command should then point you to the right location within your browser to use the notebook, typically [http://localhost:8888/](http://localhost:8888/).



### Google Colab

Using Colab is advantageous as it provides access to GPU resources. To open the notebooks in Google Colab, click the links below for each exercise:

* [Exercise 01](https://colab.research.google.com/github/Cambridge-ICCS/ml-training-material/blob/colab/exercises/01_penguin_classification.ipynb) - [Worked Solution 01](https://colab.research.google.com/github/Cambridge-ICCS/ml-training-material/blob/colab/worked-solutions/01_penguin_classification_solutions.ipynb)
* [Exercise 02](https://colab.research.google.com/github/Cambridge-ICCS/ml-training-material/blob/colab/exercises/02_penguin_regression.ipynb) - [Worked Solution 02](https://colab.research.google.com/github/Cambridge-ICCS/ml-training-material/blob/colab/worked-solutions/02_penguin_regression_solutions.ipynb)
* [Exercise 03](https://colab.research.google.com/github/Cambridge-ICCS/ml-training-material/blob/colab/exercises/03_mnist_classification.ipynb) - [Worked Solution 03](https://colab.research.google.com/github/Cambridge-ICCS/ml-training-material/blob/colab/worked-solutions/03_mnist_classification_solutions.ipynb)
* [Exercise 04](https://colab.research.google.com/github/Cambridge-ICCS/ml-training-material/blob/colab/exercises/04_ellipse_regression.ipynb) - [Worked Solution 04](https://colab.research.google.com/github/Cambridge-ICCS/ml-training-material/blob/colab/worked-solutions/04_ellipse_regression_solutions.ipynb)

_Notes:_
* _Running in Google Colab requires a Google account._
* _If you exit a Colab session, your work will be lost, so make sure to save any work you want to keep._

### Binder

If a local installation is not feasible and you prefer not to sign up for a Google account, you can launch the repository on [Binder](https://mybinder.org/v2/gh/Cambridge-ICCS/ml-training-material/main).

_Notes:_
* _If you exit a Binder session, your work will be lost, so make sure to save any work you want to keep._
* _Due to limited resources on Binder, running the training for exercises 3 and 4 may be challenging._



## License
All code and content of this repository is licensed under the [MIT License](https://opensource.org/licenses/mit-license.php).

## Acknowledgements
This repository was created based on the template of the [Institute of Computing for Climate Science](https://github.com/Cambridge-ICCS/ml-training-material) and the [harminic oscillator pinn workshop](https://github.com/benmoseley/harmonic-oscillator-pinn-workshop) by Ben Moseley.