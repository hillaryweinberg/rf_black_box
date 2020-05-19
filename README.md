# EmptyProject
An empty project for CSPB2270 final projects
Project has following folder tree

.  
├── CMakeLists.txt  
├── CMakeLists.txt.in  
├── README.md  
├── app  
│   └── main.cpp  
├── build  
├── code  
│   ├── randomforest.cpp    
│   └── randomforest.h  
└── tests  
    └── test_randomforest.cpp  

CMakeLists.txt      : Ignore this file  
CMakeLists.txt.in   : Ignore this file  
README.md           : Readme file  
app\                : Application folder  
main.cpp            : Application main file for your personal tests. you can use this executable to debug your own classes.  
build\              : Build folder to build the project. your executables are gonna be here eventually.  
code\               : all your code should be in this folder.   
randomforest.cpp            : data_class and rf class source file  
randomforest.h              : data_class and rf class header file
tests\              : Tests folder  
test_randomforest.cpp     : Tests implemented for you / your personal tests  

## Scroll to the bottom of this README file for my project proposal description

## Where to Start
Open a terminal window in Jupytherhub(recommended) or your personal linux/mac machine (no windows platforms). First pull the this repository by
```console
jovyan@jupyter-yourcuid:~$ git clone https://github.com/hillaryweinberg/rf_black_box.git
```
### If you want to use vscode environment
Then open the VScode app through JupyterHub and open Graphs folder from vscode.

Now your environment is set up, change Graphs.cpp and Graphs.h files and eventually press CTRL+SHIFT+B to compile your code.

Open a terminal window in vscode and go into ''build'' folder and run tests by
```console
jovyan@jupyter-yourcuid:~$ ./run_tests
```
debugger is also set up for you, go to debug tab on the left column and select **Run Tests** in drop-down menu and press the green play button to run the debugger. if you have any implementation in **app/main.cpp** you can also debug that code by first choosing **Run App** in drop-down menu and pressing green play button.

### If you want to use linux terminal for comilation
Make sure you have the dependecies installed (ckeck dependencies in this document)
go into your project folder, then build folder
```console
jovyan@jupyter-yourcuid:~$ cd Graphs
jovyan@jupyter-yourcuid:~$ cd build
```
run cmake to create make file for your project
```console
jovyan@jupyter-yourcuid:~$ cmake ..
```
run make to compile your code
```console
jovyan@jupyter-yourcuid:~$ make
```
once done, you can run tests by 
```console
jovyan@jupyter-yourcuid:~$ ./run_tests
```
app executable is also in this folder, you can run it by
```console
jovyan@jupyter-yourcuid:~$ ./run_app
```
you can debug in terminal using gdb

## Dependencies
you need **gcc** and **cmake** installed to be able to compile this code.

if you are using vscode environment in JupyterHub, you just need to make sure you have C/C++ extension installed.

## About This Project
Check randomforest.h file comments for detailed information for each function.

## Project Proposal Description

Project: Random Forest Practice
Name: Hillary Weinberg
Class and Semester: CSPB2270, Spring 2020
Machine learning algorithms often deal with multiple trees, arrays, or lists, and establish patterns by running the training data through algorithms and establishing patterns.  Often the decision tree data is stored as a binary tree to increase prediction speed and work efficiently with un-normalized datasets.  A random forest combines and merges multiple decision trees to get a more accurate and stable prediction.  Decision trees individually can be “weak learners”, but come together to form a “strong learner.”
Overfitting is a common occurrence when building flexible (high capacity) models.  Models that are too flexible end up incorporating noise into the data.  The structure of the model will have a high variance as they will vary based on the data fed to the model.  In contrast, an inflexible model is biased to pre-conceived assumption about the data, and is not adaptable to changing inputs.  Decision trees are especially disposed to overfitting when there are no limits on maximum depth, and can keep growing until there is a single leaf for every observation.  Overfitting can be avoided by using random forests to create random sampling of training observations.  Each tree in the forest learns from a random sample (sometimes known as “bootstrapping”), which are drawn with replacement.  The trees are composed of random subsets of features for splitting nodes calculated by the square root of the features. 
There are several abstract data type methods applied to binary decision tree to create random forests or simplify the decision tree.  The structure of the tree will change according to the feature and threshold.  Real-world problems often have unbalanced trees.  Unlike binary search trees, in a decision tree you can delete a node to balance the tree and minimize the number of decision steps.  This creates a stronger learner and can bring the “Gini impurity” down to null.    
There are many machine learning algorithms available as Python libraries, such as scikit-learn and Tensorflow.  Often these algorithms construct trees and forests once you provide cleaned data.  For this project I will explore building a random forest that is usually hidden behind a black box.  To do this I will build decision trees, calculate the current Gini impurity, and create a subset of trees to conduct supervised training on the data by random sampling.  I will not be training the algorithm, but time permitting may run the data through a similar algorithm in Python to see results.

# rf_black_box
