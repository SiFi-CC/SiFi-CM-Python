# Auxiliary scripts 

These scripts are mainly prepared for the 1d simulation/reconstruction, but could be easily extended to 2D case. 

Copying the script you need to your working directory can make life easier ;)

## Set of simulations

Usage
``` bash
bash simbash.sh
```

Prepares a set of simulations with different source positions. If not changed, should be used in the build directory of `G4Simulation` project.
*Please change source parameters according to your needs*

## Set of reconstructions

Usage
``` bash
bash recobash.sh
```


Prepares a set of reconstructions with different source positions. If not changed, should be used in the build directory of `ComptonCamera6` project.
*Please put the paths to your simulations files and system matrix*

## Reconstructed data analysis in Python

Following scripts are in `Python3`.

### Setting-up

* Install Python3

``` bash
sudo apt-get install python3
```

* Create virtual environment [optional - one can use other options, like Anaconda etc.]

``` bash
python3 -m venv <environment_name>
```

* Activate environment

```bash
source <environment_name>/bin/activate
```
 *Can be deactivated by the command `deactivate`*

* Install required libraries (*may take some time*)

```bash
pip3 install matplotlib uproot tqdm scipy pyqt5
```

### Scripts

**analyze_reco_1d.py** can be used for the preparing of figures of the set of reconstruction files

```bash
python3 analyze_reco_1d.py
```

*Please change it according to your data*

**root_aux.py** contains some auxiliary modules which allow to process `.root` files, make reconstruction in Python, calculate MSE and UQI values etc.