# RayTraceDicom
Sub-second pencil beam dose calculation on GPU for adaptive proton therapy

![](doc/watercube.png)

LEGAL NOTICE
------------
The core of the code within this repository is based on the PhD project of Joakim da Silva, developed within the ENTERVISION Marie Curie Initial Training Network.
The core was later refactored and documented by Fernando Hueso-González.

The code is licensed under GPLv3. When (re)using this code, attribution to Joakim da Silva must be granted and relevant papers must be cited:
- https://doi.org/10.1088/0031-9155/60/12/4777
- https://doi.org/10.1016/j.jpdc.2015.07.003

More details:
- https://doi.org/10.17863/CAM.16186

REQUIREMENTS
------------
- CUDA, CUDA-TOOLKIT, CUDA-SAMPLES
- GDCM
- ITK (if custom build, then with `-DITK_USE_SYSTEM_GDCM=ON`, otherwise just `apt-get install libinsighttoolkit5-dev`)
- `git clone https://github.com/ferdymercury/cmake-modules` into `/opt`
- qhelpgenerator (`sudo apt install qhelpgenerator-qt5`)

BUILDING
--------
- `mkdir build`
- `cmake <path/to/src>/RayTraceDicom`, optionally with `-DITK_DIR=/path/to/build`
- `make`

EXAMPLE FILES
-------------
To generate a dummy water phantom CT and RT plan, use:
- `sudo pip3 install scipy pydicom`
- `python3 extern/dicom-interface/rti/test/dicom/generate_water_cube.py --outdir /tmp/watercube/ --institution rbe --machine 1.1`

RUNNING
-------
- `./src/RayTraceDicom --output_directory /tmp/watercube/ --ct_dir /tmp/watercube/ct/ --rtplan /tmp/watercube/rtplan.dcm --beams G000`

NOTES
-----
You might need for old Tesla C2070 commands such as:
- Install patched nvidia-390 driver on Ubuntu 22: https://launchpad.net/%7Edtl131/+archive/ubuntu/nvidiaexp
- Install gcc5 and cuda8: https://askubuntu.com/questions/1442001/cuda-8-and-gcc-5-on-ubuntu-22-04-for-tesla-c2070
- Error with stncpy: https://stackoverflow.com/questions/76531467/nvcc-cuda8-gcc-5-3-no-longer-compiles-with-o1-on-ubuntu-22-04
- Error with float128: https://askubuntu.com/questions/1442001/cuda-8-and-gcc-5-on-ubuntu-22-04-for-tesla-c2070
- `cmake ../ -DCOMPILE_SM20=ON -DCOMPILE_SM35=OFF -DWATER_CUBE_TEST=ON -DCUDA_HOST_COMPILER=/opt/gcc5/gcc -DSEPARATE_COMPILATION=OFF -DCMAKE_CXX_COMPILER=/opt/gcc5/g++ -DCMAKE_C_COMPILER=/opt/gcc5/gcc -DCMAKE_CXX_STANDARD=11`
- This might also be needed depending on the platform or CMake version: `export PATH=/opt/gcc5:$PATH`
- Need to fine-tune QtCreator adding a new custom compiler /opt/cuda-8.0/bin/nvcc and edit .config/clangd/config.yaml file with
```
CompileFlags:
Add:
  [
    '--cuda-path="/opt/cuda-8.0/"',
    --cuda-gpu-arch=sm_20,
    '-L"/opt/cuda-8.0/lib64/"',
    -lcudart,
  ]
```
- See https://github.com/clangd/clangd/issues/858 and https://github.com/clangd/clangd/issues/1815
