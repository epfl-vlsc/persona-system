
**Persona**: A High-Performance Bioinformatics Framework

Persona is an open source framework for executing bioinformatics computations on the Aggregate Genomic Data format. Persona is fast, efficient, and scalable.

This repo contains the source implementation of the Persona dataflow operators, all of which is built on top of TensorFlow. 

[This repo](https://github.com/epfl-vlsc/persona) contains the Persona python layer that allows you to access common functions via a command line interface. You will need both repos to use Persona.

## Build and Install

The Persona system can be built and installed in a similar manner to installing TensorFlow from sources. 
We recommend using a Python virtual environment when installing so as not to conflict with any existing TensorFlow installation.

First you should clone this repo with `--recurse-submodules` and [prepare your environment](https://www.tensorflow.org/install/install_sources#prepare_environment_for_linux). 
Disregard any setup for GPUs.
In addition, you will need to install the following dependencies via your package manager:

* liblttng-ust-dev
* librados-dev
* libboost-system-dev
* libboost-timer-dev
* libsparsehash-dev

e.g. 
```shell
sudo apt-get install liblttng-ust-dev librados-dev libboost-system-dev libboost-timer-dev libsparsehash-dev 
```

The following may need to be installed from source:

* [libngs (and associated python bindings)](https://github.com/ncbi/ngs)
* [ncbi-vdb](https://github.com/ncbi/ncbi-vdb)

Next, configure your environment using:

```shell
cd persona-system
./default-configure.sh
```

This configures TensorFlow with the minimum requirements for Persona.
Next we will create our virtual environment. Persona provides a convenient script. Simply:

```shell
./setup-dev.sh
```

Enter the environment:

```shell
source python-dev/bin/activate
```

Compile the pip package:

```shell
./compile.sh
```

Next, head on over to the [Persona](https://github.com/epfl-vlsc/persona) repo to see how to use the Persona framework. 

