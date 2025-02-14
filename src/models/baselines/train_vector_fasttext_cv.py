import logging
import datetime
import os
import torch
import itertools
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import KFold
from datasets import EmbeddingDataset
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from tqdm import tqdm
from torch import nn
import argparse
import random
import numpy as np
from sklearn.metrics import r2_score
from src.utils.torch_utils import MLPRegressor
from utils import get_embedding_model

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Seed everything
torch.manual_seed(234)
np.random.seed(234)
random.seed(234)

# Logging setup
def setup_logging(emb_model, data_dir):
    current_time = datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S-{emb_model}")
    log_dir = os.path.join(data_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"log_{current_time}.txt")
    logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(message)s")

def print_log(*args, **kwargs):
    logging.info(*args, **kwargs)
    print(*args, **kwargs)

def step(model, loss_fn, data, labels, num_components, optimizer=None, verbose=False):
    data, labels = data.to(DEVICE), labels.to(DEVICE)
    outputs = model(data)
    vector_dim = labels.shape[-1]

    mu, log_var, pi = torch.split(
        outputs, 
        [num_components * vector_dim, num_components * vector_dim, num_components], 
        dim=-1
    )

    log_var = torch.clamp(log_var, min=-10, max=10)
    var = torch.exp(log_var).clamp(min=1e-6, max=1e6)
    pi = torch.softmax(pi, dim=-1).clamp(min=1e-6, max=1.0) # deal with overflow problem
    mu = mu.view(-1, num_components, vector_dim)
    var = var.view(-1, num_components, vector_dim)
    pi = pi.view(-1, num_components, 1)
    labels = labels.unsqueeze(1)
    if torch.isnan(mu).any() or torch.isnan(var).any() or torch.isnan(pi).any():
        print("🚨 NaN detected in model parameters!")
        print(f"mu NaN count: {torch.isnan(mu).sum()}, var NaN count: {torch.isnan(var).sum()}, pi NaN count: {torch.isnan(pi).sum()}")
        raise ValueError("NaN values detected in Normal distribution parameters!")
    
    dist = Normal(mu, var)
    log_likelihood = dist.log_prob(labels)
    sum_ll = torch.sum(log_likelihood, dim=-1)
    log_likelihoods = torch.logsumexp(sum_ll + torch.log(pi.squeeze() + 1e-6), dim=-1)
    loss = -torch.mean(log_likelihoods)

    if loss_fn is not None:
        additional_loss = loss_fn(mu, labels)
        loss = loss + additional_loss

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

    predicted_mu = torch.sum(mu * pi, dim=1)
    labels_squeezed = labels.squeeze(1)
    mae = F.l1_loss(predicted_mu, labels_squeezed)
    try:
        r2 = r2_score(labels_squeezed.cpu().detach().numpy(), predicted_mu.cpu().detach().numpy())
    except ValueError:
        print("🚨 R2 calculation failed! Returning 0.0")
        r2 = 0.0

    entropy_per_cmp = dist.entropy()  # [batch size, components, features]
    entropy_per_cmp = torch.sum(entropy_per_cmp, dim=-1)  # Sum over features
    weighted_entropy = torch.sum(entropy_per_cmp * (pi.squeeze() + 1e-6), dim=-1) 
    mean_entropy = torch.mean(weighted_entropy)

    return loss.item(), mae.item(), r2, mean_entropy.item()


def train_epoch(model, optimizer, loss_fn, dataloader, num_components, device, print_log_every=50):
    model.train()
    train_loss, train_mae, train_r2, train_entr = 0, 0, 0, 0

    for data, target in tqdm(dataloader, desc="Training", total=len(dataloader)):
        loss, mae, r2, entr = step(model, loss_fn, data, target, num_components, optimizer)
        train_loss += loss
        train_mae += mae
        train_r2 += r2
        train_entr += entr

    return train_loss / len(dataloader), train_mae / len(dataloader), train_r2 / len(dataloader), train_entr/len(dataloader)

def validation_epoch(model, loss_fn, dataloader, num_components, device):
    model.eval()
    validation_loss, validation_mae, validation_r2, validation_entr = 0, 0, 0, 0

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validation", total=len(dataloader)):
            loss, mae, r2, entr = step(model, loss_fn, data, target, num_components)
            validation_loss += loss
            validation_mae += mae
            validation_r2 += r2
            validation_entr += entr

    return validation_loss / len(dataloader), validation_mae / len(dataloader), validation_r2 / len(dataloader), validation_entr / len(dataloader)


def cross_validation(args, dataset, word_model, loss_fn, all_sel_params):
    device = torch.device(args.DEVICE)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = list(range(len(dataset)))

    best_params = None
    best_val_loss = float("inf")
    best_model_state = None
    best_test_metrics = None

        # Select random hyperparameter combination
    # for i in range(2):
    #     print(f"\nRun {i+1} out of {args.NUM_RUNS}")
    #     params = random.choice(all_params)
    for params in all_sel_params:
        print(f"Training with parameters: {params}")
        fold_test_losses = []
        fold_test_entropy = []
        fold_val_losses = []
        patience = 3

        for fold, (train_val_idx, test_idx) in enumerate(kf.split(indices)):
            print(f"\n========== FOLD {fold+1} / 5 ==========")

            # Split dataset into training + validation sets
            train_val_subset = Subset(dataset, train_val_idx)
            test_subset = Subset(dataset, test_idx)

            train_size = int(0.75 * len(train_val_subset))
            val_size = len(train_val_subset) - train_size
            train_subset, val_subset = torch.utils.data.random_split(train_val_subset, [train_size, val_size])

            print(f"Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

            train_dataloader = DataLoader(train_subset, batch_size=args.BATCH_SIZE, shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=args.BATCH_SIZE, shuffle=True)
            # test_dataloader = DataLoader(test_subset, batch_size=args.BATCH_SIZE, shuffle=True)
            test_dataloader = DataLoader(test_subset, batch_size=32, shuffle=True)

            # Initialize model
            model = MLPRegressor(
                num_layers=params["num_layers"],
                input_size=args.INPUT_SIZE,
                hidden_size=params["hidden_units"],
                num_labels=args.NUM_MIX * (args.OUTPUT_SIZE + 1),
                dropout_probability=params["dropout"],
            ).to(DEVICE)

            optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["l2_reg"])
            best_val_fold_loss = float("inf")
            wait = 0  # Early stopping counter
            # Training loop
            for epoch in range(args.EPOCHS):
                print_log(f"Epoch {epoch+1}")
                train_loss, _, _, train_entr = train_epoch(model, optimizer, loss_fn, train_dataloader, args.NUM_MIX, device)
                val_loss, _, _ , val_entr = validation_epoch(model, loss_fn, val_dataloader, args.NUM_MIX, device)
                print_log(f"Train Loss: {train_loss:.6f}, Train Entropy: {train_entr:.6f}, Val Loss: {val_loss:.6f}, Val Entropy: {val_entr:.6f}")

                # early stopping
                if val_loss < best_val_fold_loss:
                    best_val_fold_loss = val_loss
                    best_model_state = model.state_dict()  # Save best model weights
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"🚀 Early stopping triggered at epoch {epoch+1}")
                        break
            fold_val_losses.append(val_loss)

            model.load_state_dict(best_model_state)
            # Evaluate the best model on fixed test set
            test_loss, _, _, test_entr = validation_epoch(model, loss_fn, test_dataloader, args.NUM_MIX, device)
            fold_test_losses.append(test_loss)
            fold_test_entropy.append(test_entr)
            print(f"✅ FOLD {fold+1} Test Loss (Best Model): {test_loss:.6f}, Test Entropy: {test_entr:.6f}")

        mean_val_loss = np.mean(fold_val_losses)

        mean_test_loss = np.mean(fold_test_losses)
        std_test_loss = np.std(fold_test_losses)
        mean_test_entropy = np.mean([x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in fold_test_entropy])
        std_test_entropy = np.std([x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in fold_test_entropy])
        
        print("\nCross-Validation Results:")
        print(f"Test Losses for 5 Folds: {fold_test_losses}")
        print(f"Test Entropy for 5 Folds: {fold_test_entropy}")
        print(f"Mean Test Loss: {mean_test_loss:.6f}, Mean Test Entropy: {mean_test_entropy:.6f}")
        print(f"Standard Deviation of Test Loss: {std_test_loss:.6f}, of Test Entropy: {std_test_entropy:.6f}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_params = params
            best_test_metrics = (mean_test_loss, std_test_loss, fold_test_losses, mean_test_entropy, std_test_entropy, fold_test_entropy)
            print(f"✅ New Best Params Found! Mean Validation Loss: {mean_val_loss:.6f}")
    print("\n🏆 Best Hyperparameter Configuration:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Mean Validation Loss: {best_val_loss:.6f}")

    return best_test_metrics


def main(args):

    HYPERPARAMS = {
        "learning_rate": [0.001],
        "l2_reg": [ 0.001],
        "dropout": [0.2, 0.5],
        "num_layers": [15, 20, 30],
        "hidden_units": [512, 1024],
    }

    all_params = [
        dict(zip(HYPERPARAMS.keys(), values))
        for values in itertools.product(*HYPERPARAMS.values())
    ]
    random.shuffle(all_params)
    selected_params = random.sample(all_params, k=args.NUM_RUNS)

    setup_logging(args.EMB_MODEL, args.DATA_DIR)

    print("Number of Compoments: ", args.NUM_MIX)

    if args.EMB_MODEL == "glove":
        emb_path = args.GLOVE_PATH
    elif args.EMB_MODEL == "fasttext":
        emb_path = args.FASTTEXT_PATH
    else:
        raise ValueError(f"Model {args.EMB_MODEL} not supported.")

    loss_fn = None
    word_model = get_embedding_model(args.EMB_MODEL, emb_path)
    device = torch.device(args.DEVICE)

    train_dataset = EmbeddingDataset(args.DATA_DIR, "train-clean-100", word_model)
    val_dataset = EmbeddingDataset(args.DATA_DIR, "dev-clean", word_model)
    test_dataset = EmbeddingDataset(args.DATA_DIR, "test-clean", word_model)
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=True
    )

    # train_val_dataset = ConcatDataset([train_dataset, val_dataset])
    # print(f"Train validation dataset size: {len(train_val_dataset)}")

    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    print(f"Total dataset size: {len(full_dataset)}")

    (mean_test_loss, 
    std_test_loss, 
    _, 
    mean_test_entropy, 
    std_test_entropy, 
    _) = cross_validation(
        args, full_dataset, word_model, loss_fn, selected_params
        )

    print(f"Final Cross-Validation Completed! Mean Test Loss: {mean_test_loss:.6f}, Std: {std_test_loss:.6f}, Mean Test Entropy: {mean_test_entropy:.6f}, Std: {std_test_entropy:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_RUNS", default=12, type=int)
    parser.add_argument("--INPUT_SIZE", default=300, type=int)
    parser.add_argument("--DEVICE", default=DEVICE, type=str)
    parser.add_argument("--NUM_MIX", default=20, type=int)
    parser.add_argument("--OUTPUT_SIZE", default=2, type=int)
    parser.add_argument("--EPOCHS", default=50, type=int)
    parser.add_argument("--BATCH_SIZE", default=4096, type=int)
    parser.add_argument("--DATA_DIR", default="/home/user/ding/Projects/Prosody/languages/", type=str)
    parser.add_argument("--FASTTEXT_PATH", default="/home/user/ding/Projects/Prosody/models/fastText/cc.en.300.bin", type=str)
    parser.add_argument("--EMB_MODEL", default="fastText", type=str)

    args = parser.parse_args()

    main(args)
