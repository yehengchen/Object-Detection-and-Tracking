# Maks R-CNN

## [labelme](https://github.com/wkentaro/labelme) - Image Polygonal Annotation with Python 

<div align="left">
  <img src="https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/TwoStage/R-CNN/annotation.jpg" width="50%">
</div>

### Requirements

- Ubuntu / macOS / Windows
- Python2 / Python3
- [PyQt4 / PyQt5](http://www.riverbankcomputing.co.uk/software/pyqt/intro) / [PySide2](https://wiki.qt.io/PySide2_GettingStarted)


#### Anaconda

You need install [Anaconda](https://www.continuum.io/downloads), then run below:

```
# python2
conda create --name=labelme python=2.7
source activate labelme
# conda install -c conda-forge pyside2
conda install pyqt
pip install labelme
# if you'd like to use the latest version. run below:
# pip install git+https://github.com/wkentaro/labelme.git

# python3
conda create --name=labelme python=3.6
source activate labelme
# conda install -c conda-forge pyside2
# conda install pyqt
# pip install pyqt5  # pyqt5 can be installed via pip on python3
pip install labelme
# or you can install everything by conda command
# conda install labelme -c conda-forge
```

#### Ubuntu

```bash
# Ubuntu 14.04 / Ubuntu 16.04
# Python2
# sudo apt-get install python-qt4  # PyQt4
sudo apt-get install python-pyqt5  # PyQt5
sudo pip install labelme
# Python3
sudo apt-get install python3-pyqt5  # PyQt5
sudo pip3 install labelme
```

### Windows

Firstly, follow instruction in [Anaconda](#anaconda).

```bash
# Pillow 5 causes dll load error on Windows.
# https://github.com/wkentaro/labelme/pull/174
conda install pillow=4.0.0
```
