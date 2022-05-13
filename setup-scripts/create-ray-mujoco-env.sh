#!/bin/bash

cd ~
conda create --clone ray-test --name ray-mujoco
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ray-mujoco
cd ~/.mujoco/mujoco-py
pip install -r requirements.txt
pip install -r requirements.dev.txt
pip3 install -e . --no-cache
pip3 install -U 'mujoco-py<2.2,>=2.1'
git clone https://github.com/parachutel/gym.git ~/miniconda3/envs/ray-mujoco/lib/python3.7/site-packages/
python ~/.mujoco/mujoco-py/examples/setting_state.py 

