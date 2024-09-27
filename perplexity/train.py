# External
from datetime import datetime
import torch
import wandb
import matplotlib.pyplot as plt
import pprint
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Internal
from data import setup_tokenizer, load_texts, create_datasets, get_dataloaders
from model import *
from train_utils import train_model
from arg_parser import get_args

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

if __name__ == "__main__":
    # Parse Arguments
    args = get_args()
    
    # Constants
    SEQ_LENGTH = 64
    STRIDE = 1

    # Args
    BATCH_SIZE = args.batch_size
    EPOCHS = args.num_epochs
    LR = args.lr

    tokenizer = setup_tokenizer()

    # Models, Loss
    if args.architecture == "FCN":
        models = MetaFullyConnectedModels(vocab_size=len(tokenizer))
    elif args.architecture == "VanillaTransformer":
        models = MetaVanillaTransformers(vocab_size=len(tokenizer))

    # User Hyperparam Feedback
    pprint.pprint(vars(args))
    print()

    # Setup Dataset
    if args.dataset_version == "small":
        dataset_name = "wikitext-2-v1"
    elif args.dataset_version == "large":
        dataset_name = "wikitext-103-v1"

    train_texts, val_texts = load_texts(dataset_name)
    full_train_dataset, full_val_dataset = create_datasets(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer=tokenizer,
        sequence_length=SEQ_LENGTH,
        stride=STRIDE
    )
    
    # Scaling Experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_name = f"{dataset_name}_{args.architecture}_ts={timestamp}"  # for wandb

    for data_fraction in tqdm(args.data_fractions, desc="Data Iteration"):
        
        train_loader, val_loader = get_dataloaders(
            train_dataset=full_train_dataset,
            val_dataset=full_val_dataset,
            batch_size=BATCH_SIZE,
            train_fraction=data_fraction,
        )

        for model in models:
            model.to(DEVICE)
            print(
                f"\nModel is on device {DEVICE} and has {model.num_params} parameters"
            )
            model_name = f"{args.architecture}_dv={args.dataset_version}_df={data_fraction}_p={model.num_params}"

            if args.wandb_log:
                run = wandb.init(
                    project="wikitext-scaling",
                    name=model_name,
                    group=group_name,
                    config={
                        "learning_rate": LR,
                        "num_epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "data_fraction": f"{int(data_fraction*100)}%",
                    },
                )

            train_loss, val_loss, train_perplexity, val_perplexity = train_model(
                model, DEVICE, train_loader, val_loader, EPOCHS, LR
            )

            print(
                f"Dataset Size: {int(data_fraction*100)}%, Train Perplexity: {train_perplexity}, Val Perplexity: {val_perplexity}\n"
            )

            if args.wandb_log:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_perplexity": train_perplexity,
                        "val_loss": val_loss,
                        "val_perplexity": val_perplexity,
                        "num_params": model.num_params,
                    }
                )
            wandb.finish()
