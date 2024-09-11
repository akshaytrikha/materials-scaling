### Env setup

1. `pyenv local 3.11.8`
2. `pyenv exec python -m venv materials-scaling-venv`
3. `source materials-scaling-venv/bin/activate`
4. `python3 -m pip install -q -r requirements.txt`
5. `wandb login`

### Running training
`python3 train.py --batch_size=64 --num_epochs=5 --lr=0.001 --dataset_size=small --wandb_log`