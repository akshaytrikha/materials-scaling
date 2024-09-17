### Env setup

1. `pyenv local 3.11.8`
2. `pyenv exec python -m venv materials-scaling-venv`
3. `source materials-scaling-venv/bin/activate`
4. `python3 -m pip install -q -r requirements.txt`
5. `wandb login`

### Running training
`python3 train.py --architecture=FCN --batch_size=64 --num_epochs=5 --lr=0.001 --seq_max_length=512 --dataset_version=small --wandb_log`

### Black formatting setup
1. Open VSCode settings and search for `editor: default formatter`
2. Select `Black Formatter`
2. Select `Format on Save` right below