Problems encountered buring install and other 

use --no-build-isolation on all pip installs 
1) install torch 2.11 for cuda 13.0
-> pip3 install torch torchvision (on april 29 2026)

I then installed 
cd third_party/MLEntropy/cpp/3rdparty
git clone https://github.com/pybind/pybind11.git

I then had this problem ModuleNotFoundError: No module named '_gridencoder'
cd third_party/gridencoder
pip install -e .