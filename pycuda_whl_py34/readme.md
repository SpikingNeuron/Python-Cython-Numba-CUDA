
#
http://www.nyayapati.com/srao/2014/06/how-to-pip-install-python-packages-offline/

## get the prebuilt binaries
+ [Link](http://www.lfd.uci.edu/~gohlke/pythonlibs/?cm_mc_uid=87700576828314629030924&cm_mc_sid_50200000=1463871313#pycuda)


## Download dependent files from machine with internet

```
pip install --download win_amd64 "pycuda-2016.1+cuda7518-cp34-cp34m-win_amd64.whl"
```

## Offline install

```
pip install --no-index --find-links=file:/win_amd64 "pycuda-2016.1+cuda7518-cp35-cp35m-win_amd64.whl"
```