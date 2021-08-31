# Alzheimer's disease classification
Data source: http://adni.loni.usc.edu/

This project aims to create a classification for Alzheimer's disease using a 3D deep learning model


## Installation

Create virtual environment

```bash
conda create -n Alzheimer python=3.8
conda activate Alzheimer
```
Install dependencies:

```bash
pip install -r requirements.txt
```

Download and set up data by running

```bash
bash setup_data.sh
```   

## Usage
Run and save model
```python
python train.py
```
Expected output:

```
Epoch 1/1
5/5 - 12s - loss: 0.7031 - acc: 0.5286 - val_loss: 1.1421 - val_acc: 0.5000

<tensorflow.python.keras.callbacks.History at 0x7fea600ecef0>

``` 



## After the model is trained, run the test file:

``` bash
python test.py
```
