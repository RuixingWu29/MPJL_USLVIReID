# MPJL_USLVIReID
Modality-Invariant Progressive Joint Learning for Unsupervised Visible-Infrared Person Re-Identification (MPJL)

## Dataset
Put SYSU-MM01 and RegDB dataset into data/sysu and data/regdb, run prepare_sysu.py and prepare_regdb.py to convert the dataset format.

## Prerequisites
A minimum of 72GB of GPU memory (Batchsize = 128)

## Getting Started
git clone https://github.com/RuixingWu29/MPJL_USLVIReID.git

## install dependencies
- conda create -n MPJLReID python==3.7
- conda activate MPJLReID
- pip install -r requirement.txt

## Training
```
sh MPJL_sysu_train.sh   # for SYSU-MM01
sh MPJL_regdb_train.sh  # for RegDB
```

## Testing
Before testing, you need to modify the path in test_sysu.py and test_regdb.py.
```
python test_sysu.py    # for SYSU-MM01
python test_regdb.py   # for RegDB
```
