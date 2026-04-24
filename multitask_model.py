"""Compatibility classes for DeepLC v4 multitask checkpoints.

These classes mirror the symbols referenced by the packaged
`multitask_model.pt` checkpoint so that PyTorch can deserialize the model and
the resulting model objects can also be pickled by MuMDIA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from deeplc._architecture import (
    ConvBlock,
    GlobalFeatureBranch,
    LeakyReLUSaturation,
    OneHotBranch,
)


class BatchedHeads(nn.Module):
    def __init__(
        self,
        input_size: int = 128,
        n_datasets: int = 1020,
        head_hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.n_datasets = n_datasets
        self.head_hidden_size = head_hidden_size
        self.layer1 = nn.Linear(input_size, n_datasets * head_hidden_size)
        self.w2 = nn.Parameter(torch.empty(n_datasets, head_hidden_size))
        self.b2 = nn.Parameter(torch.empty(n_datasets))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.layer1(x)
        n_datasets, head_hidden_size = self.w2.shape
        hidden = hidden.view(-1, n_datasets, head_hidden_size)
        return (hidden * self.w2.unsqueeze(0)).sum(dim=-1) + self.b2.unsqueeze(0)


class MultitaskDeepLCModel(nn.Module):
    def __init__(
        self,
        atom_sequence_length: int = 60,
        atom_channels: int = 6,
        atom_sum_sequence_length: int = 30,
        global_feature_size: int = 55,
        one_hot_sequence_length: int = 60,
        one_hot_channels: int = 20,
        one_hot_kernel_size: int = 2,
        atom_cnn_blocks: int = 3,
        atom_cnn_kernel_size: int = 5,
        atom_cnn_filters_start: int = 256,
        atom_cnn_pool_size: int = 2,
        sum_cnn_blocks: int = 3,
        sum_cnn_kernel_size: int = 5,
        sum_cnn_filters_start: int = 256,
        global_layer_size: int = 64,
        global_num_layers: int = 4,
        shared_layer_size: int = 128,
        shared_num_layers: int = 4,
        n_datasets: int = 1020,
        head_hidden_size: int = 32,
        regularizer_val: float = 0.000005,
        quantile_regression: bool = False,
    ) -> None:
        super().__init__()

        a_layers = []
        in_channels = atom_channels
        for block_idx in range(atom_cnn_blocks):
            out_channels = int(atom_cnn_filters_start / (2**block_idx))
            use_pooling = block_idx < (atom_cnn_blocks - 1)
            a_layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    atom_cnn_kernel_size,
                    use_pooling=use_pooling,
                    pool_size=atom_cnn_pool_size,
                    regularizer_val=regularizer_val,
                )
            )
            in_channels = out_channels
        self.branch_a = nn.Sequential(*a_layers, nn.Flatten())

        b_layers = []
        in_channels = atom_channels
        for block_idx in range(sum_cnn_blocks):
            out_channels = int(sum_cnn_filters_start / (2**block_idx))
            use_pooling = block_idx < (sum_cnn_blocks - 1)
            b_layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    sum_cnn_kernel_size,
                    use_pooling=use_pooling,
                    pool_size=2,
                    regularizer_val=regularizer_val,
                )
            )
            in_channels = out_channels
        self.branch_b = nn.Sequential(*b_layers, nn.Flatten())

        self.branch_c = GlobalFeatureBranch(
            global_feature_size,
            num_layers=global_num_layers,
            hidden_size=global_layer_size,
            regularizer_val=regularizer_val,
        )
        self.branch_d = OneHotBranch(
            one_hot_channels,
            one_hot_sequence_length,
            kernel_size=one_hot_kernel_size,
        )

        with torch.no_grad():
            dummy_a = torch.zeros(1, atom_channels, atom_sequence_length)
            dummy_b = torch.zeros(1, atom_channels, atom_sum_sequence_length)
            dummy_c = torch.zeros(1, global_feature_size)
            dummy_d = torch.zeros(1, one_hot_channels, one_hot_sequence_length)
            concat_size = (
                self.branch_a(dummy_a).shape[1]
                + self.branch_b(dummy_b).shape[1]
                + self.branch_c(dummy_c).shape[1]
                + self.branch_d(dummy_d).shape[1]
            )

        shared_layers = []
        for layer_idx in range(shared_num_layers):
            in_features = concat_size if layer_idx == 0 else shared_layer_size
            shared_layers.extend(
                [
                    nn.Linear(in_features, shared_layer_size),
                    LeakyReLUSaturation(),
                ]
            )
        self.shared_trunk = nn.Sequential(*shared_layers)
        self.heads = BatchedHeads(
            input_size=shared_layer_size,
            n_datasets=n_datasets,
            head_hidden_size=head_hidden_size,
        )
        self.n_datasets = n_datasets
        self.quantile_regression = quantile_regression

    def forward(
        self,
        x_atom: torch.Tensor,
        x_atom_sum: torch.Tensor,
        x_global: torch.Tensor,
        x_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        x_atom = x_atom.transpose(1, 2)
        x_atom_sum = x_atom_sum.transpose(1, 2)
        x_one_hot = x_one_hot.transpose(1, 2)

        out_a = self.branch_a(x_atom)
        out_b = self.branch_b(x_atom_sum)
        out_c = self.branch_c(x_global)
        out_d = self.branch_d(x_one_hot)
        concatenated = torch.cat([out_a, out_b, out_c, out_d], dim=1)
        shared_output = self.shared_trunk(concatenated)
        return self.heads(shared_output)
