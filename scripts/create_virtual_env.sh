#!/usr/bin/env bash
# Create virtual environment with alias.
python3 -m venv ~/.env/my_ml
echo "alias activate_my_ml='source $HOME/.env/my_ml/bin/activate'" >> $HOME/.aliases
shopt -s expand_aliases
source $HOME/.aliases
activate_my_ml
# Download requirements.
pip3 install -r requirements.txt
# Neccessary install for my text editor configuration.
pip3 install pynvim
# Set up the Jupyter Notebook kernel.
python3 -m ipykernel install --user --name my_ml --display-name "my_ml"
# Changing directory names as well.
mv module my_ml
mv bin/module bin/my_ml
