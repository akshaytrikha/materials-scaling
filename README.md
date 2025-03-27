### Env setup

1. `pyenv local 3.11.8`
2. `pyenv exec python -m venv materials-scaling-venv`
3. `source materials-scaling-venv/bin/activate`
4. `python3 -m pip install -q -r requirements.txt`
5. `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html`
6. `brew install python-tk`
5. `wandb login`

### Running training
`python train.py --architecture=FCN --datasets all --split_name train --data_fractions 1e-6 1e-5 1e-4 1e-3 1e-2 --val_data_fraction=1e-3  --epochs=500 --batch_size=256 --gradient_clip=0.1 --vis_every=10 --val_every=10 --train_workers=8 --val_workers=8`

### Black formatting setup
1. Open VSCode settings and search for `editor: default formatter`
2. Select `Black Formatter`
2. Select `Format on Save` right below

### Visualizing model's prediction

For one sample
python3 model_prediction_evolution.py results/experiments_20250131_160623.json --split train  --sample-idx 0 

For all samples just simply remove --sample-idx flag

### Running Tensorboard logging

`tensorboard --logdir runs --bind_all`
