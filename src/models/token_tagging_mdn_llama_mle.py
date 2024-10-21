import os
import inspect
from typing import Any, Dict, List, Tuple
import pickle

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MinMetric, MaxMetric
from torch.distributions import MultivariateNormal
from torch.distributions import Normal
from torch.nn import L1Loss

from transformers import AdamW, AutoModel, get_linear_schedule_with_warmup
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import prepare_model_for_kbit_training

from memory_profiler import profile

from src.utils import utils
from src.utils.torch_utils import (
    masked_loss,
    masked_GNLLL,
    MLPRegressor,
    freeze_pretrained_model,
    print_num_trainable_params,
)
from src.utils.torch_metrics import (
    MaskedMeanAbsoluteError,
    MaskedR2Score,
    MaskedPearsonCorrCoef,
    MeanMetric,
)


class TokenTaggingRegressorLlamaMLE(LightningModule):
    """
    Transformer Model for Token Tagging, i.e. per Token Sequence Regression.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        huggingface_model: str,
        num_labels: int,
        freeze_lm: bool = False,
        train_last_k_layers: int = None,
        optimizer: torch.optim.Optimizer = AdamW,
        scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
        use_mlp: bool = False,
        p_dropout: float = 0.1,
        mlp_hidden_size: int = 512,
        mlp_num_layers: int = 5,
        loss_fn: nn.Module = torch.nn.L1Loss(reduction="none"),
        output_activation: nn.Module = torch.nn.Identity(),
        save_path: str = None,
        llama_path: str = None,
        tokenization_by_letter: bool = False,
        remove_last_layer: int = 0,
        only_one_letter: bool = False,
        num_mixtures: int = 50 # Number of mixtures for Mixture Density Network (MDN)  
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["loss_fn", "output_activation"])
        self.remove_last_layer = remove_last_layer
        self.only_one_letter = only_one_letter

        # Load model and add head
        if llama_path is not None:
            print("Loading LLAMA model.")
            self.model = LlamaForCausalLM.from_pretrained(llama_path)
        elif 'llama' in huggingface_model:
            print("Loading Huggingface model.")
            print('using llama')
            # self.model = AutoModelForCausalLM.from_pretrained(huggingface_model, load_in_4bit=True, device_map="auto", torch_dtype=dtype.float16)
            self.model = AutoModelForCausalLM.from_pretrained(huggingface_model, load_in_4bit=True, device_map="auto")
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16
            # )

            # self.model = AutoModelForCausalLM.from_pretrained(huggingface_model, quantization_config=bnb_config, device_map="auto")
        else:
            raise ValueError(f"Model {huggingface_model} not found.")

        if freeze_lm:
            print("Freezing pretrained model.")
            for param in self.model.parameters():
                param.requires_grad = False

        if train_last_k_layers is not None:
            print(f"Freezing all but last {train_last_k_layers} layers.")
            self.model = freeze_pretrained_model(
                model=self.model, model_name=huggingface_model, k=train_last_k_layers
            )

        self.num_labels = num_labels
        #self.num_parameters = num_labels + int(num_labels * (num_labels + 1) / 2) # for the full covariance matrix
        #self.num_parameters = 2 * num_labels  # for the diagonal covariance matrix
        self.num_mixtures = num_mixtures
        self.num_parameters = num_mixtures * (2 * num_labels + 1)  # for the MDN

        if use_mlp:
            print("Using MLP as head.")
            self.regressor = MLPRegressor(
                num_layers=mlp_num_layers,
                input_size=self.model.config.hidden_size,
                hidden_size=mlp_hidden_size,
                num_labels=num_labels * 2,
                dropout_probability=p_dropout,
            )
        else:
            print("Using linear layer as head.")
            self.regressor = nn.Linear(self.model.config.hidden_size, self.num_parameters)

        self.output_activation = output_activation

        # full==True: true negative log likelihood (with constant)
        self.loss_fn = nn.GaussianNLLLoss(full=True, reduction="none")
        # self.loss_fn = torch.nn.L1Loss(reduction="none")    # stricter

        # Init classifier weights according to initialization rules of model
        self.model._init_weights(self.regressor)

        # Apply dropout rate of model
        dropout_prob = p_dropout  # self.model.config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_loss = MeanMetric()  # already masked in loss function in step
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_mae = MaskedMeanAbsoluteError()
        self.val_mae = MaskedMeanAbsoluteError()
        self.test_mae = MaskedMeanAbsoluteError()

        # self.train_r2 = MaskedR2Score()
        self.val_r2 = MaskedR2Score()
        self.test_r2 = MaskedR2Score()

        self.val_pearson = MaskedPearsonCorrCoef()
        self.test_pearson = MaskedPearsonCorrCoef()

        # for logging best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.val_mae_best = MinMetric()
        self.val_r2_best = MaxMetric()
        self.val_pearson_best = MaxMetric()

        # Collect the forward signature
        params = inspect.signature(self.model.forward).parameters.values()
        params = [
            param.name for param in params if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        self.forward_signature = params

        # create save path dir
        if save_path is not None:
            self.save_path = save_path
            os.makedirs(os.path.join(self.save_path, "predictions"), exist_ok=True)

        # print number of trainable parameters
        print_num_trainable_params(
            self, model_name=f"TokenTaggingRegressorLlamaMLE {huggingface_model}"
        )


    # add by YY 13/10/2024
    def on_load_checkpoint(self, checkpoint):
        """
        Override the default behavior to load the model's state dict with strict=False
        to allow skipping unexpected keys related to quantization.
        """
        print("Loading checkpoint with strict=False to ignore quantization keys.")
        
        # Check if checkpoint contains the model state dict
        if 'state_dict' in checkpoint:
            # Load the model's state dict with strict=False
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Checkpoint does not contain 'state_dict'.")

    def forward(self, batch: Dict[str, torch.tensor], eps=1e-4, verbose=False):
        batch_size, seq_len = batch["input_ids"].shape
        # print('batch["input_ids"]: ', batch["input_ids"])

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[
            -1
        ]  # access the last hidden state

        outputs = outputs.to(torch.float32)
        # print(f'outputs: {outputs}')

        outputs = self.dropout(outputs)
        outputs = self.regressor(outputs)
        if verbose:
            print(f"outputs shape {outputs.shape}")
        # split last dimension into mu, log_var and log_pi
        mu, log_var, log_pi = torch.split(
            outputs,
            [self.num_mixtures * self.num_labels,
             self.num_mixtures * self.num_labels,
             self.num_mixtures],
            dim=-1
        )

        if verbose:
            print(f"mu shape {mu.shape}, var shape {log_var.shape}, pi shape {log_pi.shape} ")
        
        mu = mu.view(batch_size, seq_len, self.num_mixtures, self.num_labels)
        log_var = log_var.view(batch_size, seq_len, self.num_mixtures, self.num_labels)
        log_pi = log_pi.view(batch_size, seq_len, self.num_mixtures)
        
        var = torch.exp(log_var) + eps
        pi = torch.softmax(log_pi, dim=-1)

        # print(f"mu has NaN: {torch.isnan(mu).any()}, var has NaN: {torch.isnan(var).any()}, pi has NaN: {torch.isnan(pi).any()}")

        
        return mu, var, pi

    def on_train_start(self):
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.val_mae.reset()
        self.val_mae_best.reset()
        self.val_r2.reset()
        self.val_r2_best.reset()
        self.val_pearson.reset()
        self.val_pearson_best.reset()

    # @profile
    def step(self, batch: Dict[str, torch.tensor], verbose: bool = False):
        mu, var, pi= self(batch)  # forward pass
        labels = batch["tokenized_labels"]
        loss_mask_init = batch["loss_mask"]  # ignore padded sequence in loss
        input_mask = (batch["input_ids"] != 0).float()  # where input_ids == 0 ## for unk
        loss_mask = loss_mask_init * input_mask

        if verbose:
            print(f"text: {batch['input_text']}")
            print(f"mu {mu}, \nvar {var}, \npi {pi} \nlabels {labels}, \ninit_mask {loss_mask_init} \nmask {loss_mask}")
            # print(f"text: {batch['input_text']}")
            # print(f"tokenized labels: {labels}")
            # print(f"word to tokens: {batch['word_to_tokens']}")
        vector_dim = labels.shape[-1]
        # expand the mask for the vector dimension from (bs, len) to (bs, len, vec_dim)
        vector_loss_mask = loss_mask.unsqueeze(-1).expand(-1, -1, vector_dim)

        if verbose:
            print("vector loss mask shape:", vector_loss_mask.shape)

        use_multivariate_normal = True
        if use_multivariate_normal:
            # Multivariate Gaussian log-likelihood for each mixture component
            log_likelihoods = []
            for i in range(self.num_mixtures):
                # Construct the multivariate normal distribution for each mixture
                dist = MultivariateNormal(mu[:, :, i, :], torch.diag_embed(var[:, :, i, :]))
                # Compute log-likelihood of the true labels for this mixture
                log_likelihood = dist.log_prob(labels)
                log_likelihoods.append(log_likelihood)
            log_likelihoods = torch.stack(log_likelihoods, dim=-1)  # Shape: [batch_size, seq_len, num_mixtures]
        else:
            # 1d Gussian log likelihood for each mixture component
            dist = Normal(mu, var)
            log_likelihoods = dist.log_prob(labels.unsqueeze(2))  # Shape of log_likelihoods: [batch_size, seq_len, num_mixtures, num_labels]
            log_likelihoods = log_likelihoods.sum(-1)  # Sum over label dimensions
        
        # Weight by mixture coefficients
        weighted_log_likelihoods = log_likelihoods + torch.log(pi)
        # print("weighted log likelihoods", weighted_log_likelihoods.shape)
        
        # Log-sum-exp trick to compute total log likelihood
        total_log_likelihood = torch.logsumexp(weighted_log_likelihoods, dim=-1)
        # print("LOG PROB ", total_log_likelihood)
        masked_log_likelihood_loss = total_log_likelihood * loss_mask
        non_zero_mask = (masked_log_likelihood_loss != 0)
        non_zero_ll = torch.masked_select(masked_log_likelihood_loss, non_zero_mask)
        neg_masked_log_likelihood_loss = -torch.mean(non_zero_ll)
            
        # masked_gaussian_nll = masked_GNLLL(
        #     input=mu.mean(2), target=labels, var=var, mask=loss_mask, loss_fn=self.loss_fn
        # )

        return neg_masked_log_likelihood_loss, mu, var, pi, vector_loss_mask

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, train_loss tracks it over the epoch
        (
            neg_masked_log_likelihood_loss,
            mu,
            var,
            pi,
            vector_loss_mask,
        ) = self.step(batch)
        self.train_loss(neg_masked_log_likelihood_loss)
        # self.train_mae(masked_mae)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        # self.log(
        #     "train/mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True
        # )
        return {
            "loss": neg_masked_log_likelihood_loss,
            "predictions": mu,
            "targets": batch["tokenized_labels"],
            "attention_mask": batch["attention_mask"],
        }

    def on_train_epoch_end(self):
        pass
        # torch.cuda.empty_cache()

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        (
            neg_masked_log_likelihood_loss,
            mu,
            var,
            pi,
            vector_loss_mask,
        ) = self.step(batch)
        self.val_loss(neg_masked_log_likelihood_loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return {
            "loss": neg_masked_log_likelihood_loss,
            "preds": mu,
            "targets": batch["tokenized_labels"],
            "attention_mask": batch["attention_mask"],
        }

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()
        self.val_loss_best(loss)
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)
        # mae = self.val_mae.compute()
        # self.val_mae_best(mae)
        # self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)
        # r2 = self.val_r2.compute()
        # self.val_r2_best(r2)
        # self.log("val/r2_best", self.val_r2_best.compute(), prog_bar=True)
        # pearson = self.val_pearson.compute()
        # self.val_pearson_best(pearson)
        # self.log("val/pearson_best", self.val_pearson_best.compute(), prog_bar=True)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, test_loss tracks it over the epoch
        (
            neg_masked_log_likelihood_loss,
            mu,
            var,
            pi,
            vector_loss_mask,
        ) = self.step(batch)
        # self.test_loss(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.test_loss(neg_masked_log_likelihood_loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # TODO: make callback work
        # save predictions
        np.save(
            f"{self.save_path}/predictions/test_input_ids_{batch_idx}.npy",
            batch["input_ids"].cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_attention_mask_{batch_idx}.npy",
            batch["attention_mask"].cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_labels_{batch_idx}.npy",
            batch["tokenized_labels"].cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_preds_mu{batch_idx}.npy",
            mu.cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_preds_var{batch_idx}.npy",
            var.cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_preds_pi{batch_idx}.npy",
            pi.cpu().numpy(),
        )
        np.save(
            f"{self.save_path}/predictions/test_loss_mask_{batch_idx}.npy",
            batch["loss_mask"].cpu().numpy(),
        )
        # pickle input text and original labels and word_to_tokens for later evaluation
        with open(
            f"{self.save_path}/predictions/test_input_text_{batch_idx}.pkl", "wb"
        ) as f:
            pickle.dump(batch["input_text"], f)
        with open(
            f"{self.save_path}/predictions/test_original_labels_{batch_idx}.pkl", "wb"
        ) as f:
            pickle.dump(batch["original_labels"], f)
        with open(
            f"{self.save_path}/predictions/test_word_to_tokens_{batch_idx}.pkl", "wb"
        ) as f:
            pickle.dump(batch["word_to_tokens"], f)

        return {
            "loss": neg_masked_log_likelihood_loss,
            "preds": mu,
            "targets": batch["tokenized_labels"],
            "attention_mask": batch["attention_mask"],
            "loss_mask": batch["loss_mask"],
            "input_ids": batch["input_ids"],
            "input_text": batch["input_text"],
            "original_labels": batch["original_labels"],
        }
    
    def on_test_epoch_end(self):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_loss.reset()
        self.val_loss.reset()
        self.test_loss.reset()

        self.train_mae.reset()
        self.test_mae.reset()
        self.val_mae.reset()

        self.test_r2.reset()
        self.val_r2.reset()

        self.val_pearson.reset()
        self.test_pearson.reset()

    @property
    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches != 0
        ):
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (
            dataset_size // effective_batch_size
        ) * self.trainer.max_epochs

        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
