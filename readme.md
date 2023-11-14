Continual Optimistic INitialization (COIN)
==========================================

This codebase is based on the [Spinning Up repository](https://github.com/openai/spinningup/tree/master).

## Installation
---------------

Installing Spinning Up and Real-World Reinforcement Learning (RWRL) Suite requires the following steps. The installation has been only tested on Ubuntu 20.04.

### Step 1: Install Anaconda
----------
Follow the instructions [here](https://docs.anaconda.com/free/anaconda/install/linux/).

### Step 2: Install Bazel
----------
Follow the instructions [here](https://bazel.build/install/ubuntu). Version 4.2.0 has been tested to be compatible.

### Step 3: Create virtual environment
----------
```sh
git clone https://github.com/Pi-Star-Lab/spinning-coin.git
cd spinning-coin
git checkout origin/feature/coin_non_stationary
conda env create -f environment.yml
```

### Step 4: Install Libglew
```sh
sudo apt-get update
sudo apt-get -y install libglew-dev
```

### Step 5: Setup mujoco
Download `mujoco200 linux` release [here](https://www.roboti.us/download/mujoco200_linux.zip). A free license key is available for download [here](https://www.roboti.us/file/mjkey.txt).

Create a hidden folder `.mujoco`.
```sh
mkdir /home/<username>/.mujoco` 
```

Extract `mujoco200 linux.zip` to the `.mujoco` folder.
```sh
unzip -o ~/Downloads/mujoco200_linux.zip -d ~/.mujoco/
```

Move `mjkey.txt` to the `.mujoco` folder.
```sh
mv ~/Downloads/mjkey.txt ~/.mujoco/mjkey.txt
```

### Step 6: Install `realworldrl_suite`
Make sure you're in the `spinning-coin` directory. 

```sh
git clone https://github.com/google-research/realworldrl_suite.git
cd realworldrl_suite
conda activate nonstat
pip install realworldrl_suite/
```

>Note: If installation using `pip` fails, install using `conda`.

>Note: `pip` refers to your virtual environment (`nonstat`) pip. You may check this by running `whereis pip` and it should look something like this: `/home/<username>/anaconda3/envs/nonstat/bin/pip`.


### Step 7: Install `spinup`
Make sure you're in the `spinning-coin` directory.
```sh
pip install -e .
```
>Note: `pip` refers to your virtual environment (`nonstat`) pip.


## Running experiments
-------------------

```sh
python -m spinup.run <algo> --env <env_name> --exp_name <log_folder> --epochs <num_epochs> --bonus <b> --bonus_freq <bonus_frequency> --seed <seed>
```

e.g.,

For discrete actions
```sh
python -m spinup.run coin_bdual --env LunarLander-v2 --exp_name coin_bdual_lunarlander --epochs 60  --eps_disp 0.05 --eps_b 0.1 --seed 0
```

For continuous control
```
python -m spinup.run coin_td3 --env LunarLanderContinuous-v2 --exp_name coin_td3_lunarlander --epochs 100 --eps_disp 0.05 --eps_b 0.1 --seed 0
```

To run experiments using `realworldrl_suite`
```
python -m spinup.run coin_td3 --env cartpole --task_name realworld_swingup --exp_name coin_td3_cart_test --epochs 100 --seed 0 --eps_disp 0.0 --eps_b 0.0 --log_freq 1
```