
#
http://www.nyayapati.com/srao/2014/06/how-to-pip-install-python-packages-offline/

## get the prebuilt binaries
+ [Link](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pycuda)


## Download dependencies from machine with internet

```
cd pycuda_whl_py35
pip install --download win_amd64 "pycuda-2016.1.1+cuda7518-cp35-cp35m-win_amd64.whl"
```

## Offline install

```
cd win_amd64
pip install --no-index appdirs-1.4.0-py2.py3-none-any.whl
pip install --no-index pytools-2016.2.1-py2.py3-none-any.whl
pip install --no-index "pycuda-2016.1.1+cuda7518-cp35-cp35m-win_amd64.whl"
pip install --no-index numpy-1.11.0-cp35-none-win_amd64.whl
pip install --no-index pytest-2.9.1-py2.py3-none-any.whl
```

Already satisfied requirements
```
pip install --no-index colorama-0.3.7-py2.py3-none-any.whl
pip install --no-index decorator-4.0.9-py2.py3-none-any.whl
pip install --no-index py-1.4.31-py2.py3-none-any.whl
pip install --no-index six-1.10.0-py2.py3-none-any.whl
```