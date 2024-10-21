import os
import inspect
from typing import Any, Dict, List, Tuple
import pickle

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MinMetric, MaxMetric
from torch.distributions import MultivariateNormal
from torch.nn import L1Loss
from torch.distributions import Cauchy, Normal, Gamma, LogNormal, Laplace
from scipy.stats import gamma

from transformers import AdamW, AutoModel, GPT2Model, AutoConfig, GPT2Config, BertConfig, BertModel, get_linear_schedule_with_warmup
import numpy as np

from memory_profiler import profile

from src.utils import utils
from src.utils.distributions import BimodalGaussianDistribution
from src.utils.torch_utils import (
    masked_loss,
    MLPRegressor,
    freeze_pretrained_model,
    print_num_trainable_params,
)
from src.utils.torch_metrics import (
    MaskedMeanSquaredError,
    MaskedR2Score,
    MaskedPearsonCorrCoef,
    MeanMetric,
)


class TokenTaggingVectorRegressorMLE(LightningModule):
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
        tokenization_by_letter: bool = False,
        remove_last_layer: int = 0,
        only_one_letter: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["loss_fn", "output_activation"])
        self.remove_last_layer = remove_last_layer
        self.only_one_letter = only_one_letter

        # Load model and add head
        if self.remove_last_layer == 0:
            if 'mGPT' in huggingface_model or 'gpt2' in huggingface_model:
                print('using:', huggingface_model)
                self.model = GPT2Model.from_pretrained(huggingface_model)
            elif 'bert' in huggingface_model:
                print('using bert')
                self.model = BertModel.from_pretrained(huggingface_model)
            else:
                raise ValueError(f"Model {huggingface_model} not found.")
        elif self.remove_last_layer>0:
            print("Removing last layer.")
            if 'mGPT' in huggingface_model:
                print('using mGPT without layers')
                config = GPT2Config.from_pretrained(huggingface_model)
                config.n_layer -= remove_last_layer
                self.model = GPT2Model(config=config) #GPT2Model.from_pretrained(huggingface_model, config = config)
                self.model.wpe.weight.data.fill_(1.0)
            elif 'bert' in huggingface_model:
                print('using bert without layers') 
                config = BertConfig.from_pretrained(huggingface_model)
                config.num_hidden_layers -= remove_last_layer
                self.model = BertModel(config=config) #BertModel.from_pretrained(huggingface_model, config = config)
                self.model.embeddings.position_embeddings.weight.data.fill_(1.0)
        if freeze_lm:
            print("Freezing pretrained model.")
            for param in self.model.parameters():
                param.requires_grad = False

        if train_last_k_layers is not None:
            print(f"Freezing all but last {train_last_k_layers} layers.")
            self.model = freeze_pretrained_model(
                model=self.model, model_name=huggingface_model, k=train_last_k_layers
            )

        # we parameterize a normal distribution with vector mu and covariance matrix var
        self.num_labels = num_labels
        #self.num_parameters = num_labels + int(num_labels * (num_labels + 1) / 2) # for the full covariance matrix
        self.num_parameters = 2 * num_labels  # for the diagonal covariance matrix

        if use_mlp:
            print("Using MLP as head.")
            self.regressor = MLPRegressor(
                num_layers=mlp_num_layers,
                input_size=self.model.config.hidden_size,
                hidden_size=mlp_hidden_size,
                num_labels=self.num_parameters,
                dropout_probability=p_dropout,
            )
        else:
            print("Using linear layer as head.")
            self.regressor = nn.Linear(
                self.model.config.hidden_size, self.num_parameters
            )

        self.output_activation = output_activation

        # Init classifier weights according to initialization rules of model
        #if remove_last_layer==0:
        self.model._init_weights(self.regressor)

        # Apply dropout rate of model
        dropout_prob = p_dropout  # self.model.config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        # loss function (assuming single-label multi-class classification)
        self.loss_fn = loss_fn

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_loss = MeanMetric()  # already masked in loss function in step
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_mae = MeanMetric()  # we already compute the masked loss in step
        self.val_mae = MeanMetric()
        self.test_mae = MeanMetric()

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
        # save_path = '/cluster/work/cotterell/gacampa/MIT_prosody/new_predictions'
        if save_path is not None:
            self.save_path = save_path
            os.makedirs(os.path.join(self.save_path, "predictions"), exist_ok=True)
            print(self.save_path)

        # print number of trainable parameters
        print_num_trainable_params(
            self, model_name=f"TokenTaggingRegressorMLE {huggingface_model}"
        )

    def forward(self, batch: Dict[str, torch.tensor], eps=1e-7, verbose=True):
        batch_size, seq_len = batch["input_ids"].shape
        #print(batch_size, seq_len)
        if self.remove_last_layer>0:
            print('not using last layer')
            outputs = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states = True
            ).hidden_states[0]
            #print(outputs.shape)
            '''print(len(self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states = True
            ).hidden_states))'''
            #print('using first hidden state')
        else:
            outputs = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).last_hidden_state
        '''if self.only_one_letter:
            zero_mask = torch.zeros_like(outputs)
            zero_mask[:, 0:1, :] = 1
            outputs = outputs * zero_mask'''
        #print(outputs.shape)
        outputs_dropout = self.dropout(outputs)
        outputs = self.regressor(outputs_dropout)
        #print(outputs.shape)
        mu, var = torch.split(
            outputs,
            [self.num_labels, self.num_parameters - self.num_labels],
            dim=-1,
        )

        # covariance full
        '''mu, var, cov = torch.split(
            outputs,
            [self.num_labels, self.num_labels, (self.num_labels * (self.num_labels-1))//2],
            dim=-1,
        )'''

        var = torch.nn.functional.softplus(var) + eps

        '''mu = torch.nn.functional.softplus(mu) ## gamma
        if self.output_activation is not None: ###gamma
            mu = self.output_activation(mu) ## gamma'''
        
        '''var_matrix = torch.diag_embed(var)
        k = 0
        for i in range(self.num_labels-1):
            for j in range(i+1, self.num_labels):
                var_matrix[:,:,j,i] = cov[:,:,k]
                k += 1
        cov = var_matrix'''

        '''var = torch.nn.functional.softplus(var) + eps
        cov = torch.cat([cov, cov[:,:,self.num_labels:],], dim=-1)
        cov = cov[:, :, [0,4,5,6,10,1,7,8,11,13,2,9,12,14,15,3]]
        # cov = cov.view(cov.shape[0], cov.shape[1], 4, 4)
        cov_shape = cov.shape
        cov = cov.view(-1, 4, 4)
        cov = torch.bmm(cov, cov.transpose(-1, -2))
        cov = cov.view(cov_shape[0], cov_shape[1], 4, 4)
        for i in range(self.num_labels):
            cov[:,:,i,i] = torch.nn.functional.softplus(cov[:,:,i,i]) + eps
        #cov = cov + eps * torch.eye(4).unsqueeze(0).unsqueeze(0)'''
        return mu, var

    # @profile
    def step(self, batch: Dict[str, torch.tensor], verbose: bool = False):
        mu, var = self(batch)  # forward pass
        labels = batch["tokenized_labels"]
        #labels = torch.nn.ReLU()(batch["tokenized_labels"] + 2) + 1e-4 # GAMMA
        #import ipdb; ipdb.set_trace()
        loss_mask = batch["loss_mask"]  # ignore padded sequence in loss
        print(loss_mask)
        # adapt the mask for the vector labels
        vector_dim = labels.shape[-1]
        # expand the mask for the vector dimension from (bs, len) to (bs, len, vec_dim)
        vector_loss_mask = loss_mask.unsqueeze(-1).expand(-1, -1, vector_dim)
        #print("vector loss mask shape")
        #print(vector_loss_mask.shape)

        if verbose:
            # print(f"text: {batch['input_text']}")
            print(f"mu {mu}, \nlabels {labels}, \nmask {vector_loss_mask}")
        
        ### MULTIVARIATE GAUSSIAN DISTRIBUTION
        # cov = torch.diag_embed(var) # for diagonal covariance matrix
        # dist = MultivariateNormal(mu, scale_tril = cov)
        # print(mu.shape, cov.shape)
        # log_likelihood = dist.log_prob(labels)
        # #print(log_likelihood.shape)
        # masked_log_likelihood_loss = log_likelihood * loss_mask
        # #print(masked_log_likelihood_loss.shape)
        
        ### GAUSSIAN DISTRIBUTION
        dist = Normal(mu, var)
        log_likelihood = dist.log_prob(labels)
        masked_log_likelihood = log_likelihood * vector_loss_mask
        masked_log_likelihood_loss = torch.sum(masked_log_likelihood, dim=-1) # or torch.mean?

        ### CAUCHY DISTRIBUTION
        '''dist = Cauchy(mu, var)
        log_likelihood = dist.log_prob(labels)
        masked_log_likelihood = log_likelihood * vector_loss_mask
        #for i in range(4):
        #    print(sum(masked_log_likelihood[0,:,i])/sum(masked_log_likelihood[0,:,i] != 0))
        #print(masked_log_likelihood)
        masked_log_likelihood_loss = torch.sum(masked_log_likelihood, dim=-1) # or torch.mean?'''

        ### LAPLACE DISTRIBUTION
        '''dist = Laplace(mu, var)
        log_likelihood = dist.log_prob(labels)
        masked_log_likelihood = log_likelihood * vector_loss_mask
        masked_log_likelihood_loss = torch.sum(masked_log_likelihood, dim=-1) # or torch.mean?'''

        ### 1 CAUCHY AND 3 LAPLACE DISTRIBUTION
        '''dist1 = Cauchy(mu[:,:,0:1], var[:,:,0:1])
        dist2 = Laplace(mu[:,:,1:], var[:,:,1:])
        log_likelihood1 = dist1.log_prob(labels[:,:,0:1])
        log_likelihood2 = dist2.log_prob(labels[:,:,1:])
        log_likelihood = torch.cat([log_likelihood1, log_likelihood2], dim=-1)
        masked_log_likelihood = log_likelihood * vector_loss_mask
        masked_log_likelihood_loss = torch.sum(masked_log_likelihood, dim=-1) # or torch.mean?'''

        ### GAMMA DISTRIBUTION
        '''dist = Gamma(mu, var)
        labels = labels * vector_loss_mask + 1e-4
        log_likelihood = dist.log_prob(labels)
        pred = dist.mode
        masked_log_likelihood = log_likelihood * vector_loss_mask
        #for i in range(4):
        #    print(sum(masked_log_likelihood[0,:,i])/sum(masked_log_likelihood[0,:,i] != 0))
        #print(masked_log_likelihood)
        masked_log_likelihood_loss = torch.sum(masked_log_likelihood, dim=-1) # or torch.mean?'''

        non_zero_mask = (masked_log_likelihood_loss != 0)
        #print(non_zero_mask.shape)
        non_zero_ll = torch.masked_select(masked_log_likelihood_loss, non_zero_mask)
        #print(non_zero_ll.shape)
        neg_masked_log_likelihood_loss = -torch.mean(non_zero_ll)
        #print(neg_masked_log_likelihood_loss.shape)
        #mu = torch.concat([mu[:,:,0:1] * weight + mu[:,:,1:2] * (1-weight), mu[:,:,2:]], dim=-1)
        #print(mu.shape)
        masked_mae = masked_loss(
            labels=labels,
            predictions=mu, #pred
            mask=vector_loss_mask,
            loss_fn=L1Loss(reduction="none"),
        )
        if verbose:
            print(f"masked neg log_likelihood loss: {neg_masked_log_likelihood_loss}")
            print(f"masked mae: {masked_mae}")

        return neg_masked_log_likelihood_loss, masked_mae, mu, var, vector_loss_mask #mu instead of pred #var or cov

    def on_train_start(self):
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.val_r2.reset()
        self.val_r2_best.reset()

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, train_loss tracks it over the epoch
        (
            neg_masked_log_likelihood_loss,
            masked_mae,
            mu,
            var,
            vector_loss_mask,
        ) = self.step(batch)
        self.train_loss(neg_masked_log_likelihood_loss)
        self.train_mae(masked_mae)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True
        )
        return {
            "loss": neg_masked_log_likelihood_loss,
            "predictions": mu,
            "targets": batch["tokenized_labels"],
            "attention_mask": batch["attention_mask"],
        }

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, val_loss tracks it over the epoch
        # loss, preds = self.step(batch)
        (
            neg_masked_log_likelihood_loss,
            masked_mae,
            mu,
            var,
            vector_loss_mask,
        ) = self.step(batch)
        # self.val_loss(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.val_loss(neg_masked_log_likelihood_loss)
        self.val_mae(masked_mae)
        # self.val_r2(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.val_r2(mu, batch["tokenized_labels"], vector_loss_mask)
        # self.val_pearson(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.val_pearson(mu, batch["tokenized_labels"], vector_loss_mask)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/r2", self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/pearson",
            self.val_pearson,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
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
        mae = self.val_mae.compute()
        self.val_mae_best(mae)
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)
        r2 = self.val_r2.compute()
        self.val_r2_best(r2)
        self.log("val/r2_best", self.val_r2_best.compute(), prog_bar=True)
        pearson = self.val_pearson.compute()
        self.val_pearson_best(pearson)
        self.log("val/pearson_best", self.val_pearson_best.compute(), prog_bar=True)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        # loss is batch loss, test_loss tracks it over the epoch
        (
            neg_masked_log_likelihood_loss,
            masked_mae,
            mu,
            var,
            vector_loss_mask,
        ) = self.step(batch)
        # self.test_loss(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.test_loss(neg_masked_log_likelihood_loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.test_mae(masked_mae)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)
        # self.test_r2(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.test_r2(mu, batch["tokenized_labels"], vector_loss_mask)
        self.log("test/r2", self.test_r2, on_step=False, on_epoch=True, prog_bar=True)
        # self.test_pearson(preds, batch["tokenized_labels"], batch["loss_mask"])
        self.test_pearson(mu, batch["tokenized_labels"], vector_loss_mask)
        self.log(
            "test/pearson",
            self.test_pearson,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
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
        self.test_loss.reset()
        self.val_loss.reset()
        self.train_mae.reset()
        self.test_mae.reset()
        self.val_mae.reset()
        self.train_r2.reset()
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
