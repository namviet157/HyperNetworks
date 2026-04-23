"""PyTorch implementation of the HyperLSTM dynamic hypernetwork from Ha et al. (2016)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

GateState = Tuple[Tensor, Tensor, Tensor, Tensor]
StackedState = List[GateState]

GATES: Sequence[str] = ("i", "g", "f", "o")


class LayerNormLSTMCell(nn.Module):
    """Layer-normalized LSTM cell used inside the smaller hypernetwork."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        forget_bias: float = 1.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.use_layer_norm = use_layer_norm

        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        if use_layer_norm:
            self.gate_norms = nn.ModuleDict({gate: nn.LayerNorm(hidden_size) for gate in GATES})
            self.cell_norm = nn.LayerNorm(hidden_size)
        else:
            self.gate_norms = None
            self.cell_norm = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        h_prev, c_prev = state
        gates = F.linear(x, self.weight_ih) + F.linear(h_prev, self.weight_hh) + self.bias
        i_t, g_t, f_t, o_t = gates.chunk(4, dim=-1)

        if self.use_layer_norm:
            i_t = self.gate_norms["i"](i_t)
            g_t = self.gate_norms["g"](g_t)
            f_t = self.gate_norms["f"](f_t)
            o_t = self.gate_norms["o"](o_t)

        c_t = torch.sigmoid(f_t + self.forget_bias) * c_prev + torch.sigmoid(i_t) * torch.tanh(g_t)
        if self.use_layer_norm:
            h_t = torch.sigmoid(o_t) * torch.tanh(self.cell_norm(c_t))
        else:
            h_t = torch.sigmoid(o_t) * torch.tanh(c_t)

        return h_t, c_t


class GateModulator(nn.Module):
    """
    Factorized modulation head for one LSTM gate.

    This follows the paper's efficient formulation:
    hyper hidden state -> embeddings z -> scaling vectors d / bias shift.
    """

    def __init__(self, hyper_hidden_size: int, embedding_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding_size = embedding_size

        self.z_h = nn.Linear(hyper_hidden_size, embedding_size)
        self.z_x = nn.Linear(hyper_hidden_size, embedding_size)
        self.z_b = nn.Linear(hyper_hidden_size, embedding_size, bias=False)

        self.d_h = nn.Linear(embedding_size, hidden_size, bias=False)
        self.d_x = nn.Linear(embedding_size, hidden_size, bias=False)
        self.b = nn.Linear(embedding_size, hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.z_h.weight)
        nn.init.ones_(self.z_h.bias)

        nn.init.zeros_(self.z_x.weight)
        nn.init.ones_(self.z_x.bias)

        nn.init.normal_(self.z_b.weight, mean=0.0, std=0.01)

        nn.init.constant_(self.d_h.weight, 1.0 / self.embedding_size)
        nn.init.constant_(self.d_x.weight, 1.0 / self.embedding_size)

        nn.init.zeros_(self.b.weight)
        nn.init.zeros_(self.b.bias)

    def forward(self, hyper_hidden: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        scale_h = self.d_h(self.z_h(hyper_hidden))
        scale_x = self.d_x(self.z_x(hyper_hidden))
        bias = self.b(self.z_b(hyper_hidden))
        return scale_h, scale_x, bias


class HyperLSTMCell(nn.Module):
    """Main LSTM cell whose weights are modulated by a smaller hyper LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        hyper_hidden_size: int = 64,
        hyper_embedding_size: int = 16,
        forget_bias: float = 1.0,
        use_layer_norm: bool = True,
        recurrent_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.forget_bias = forget_bias
        self.use_layer_norm = use_layer_norm
        self.recurrent_dropout = recurrent_dropout

        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))

        self.hyper_cell = LayerNormLSTMCell(
            input_size=input_size + hidden_size,
            hidden_size=hyper_hidden_size,
            forget_bias=forget_bias,
            use_layer_norm=use_layer_norm,
        )

        self.modulators = nn.ModuleDict(
            {gate: GateModulator(hyper_hidden_size, hyper_embedding_size, hidden_size) for gate in GATES}
        )

        if use_layer_norm:
            self.gate_norms = nn.ModuleDict({gate: nn.LayerNorm(hidden_size) for gate in GATES})
            self.cell_norm = nn.LayerNorm(hidden_size)
        else:
            self.gate_norms = None
            self.cell_norm = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight_ih)
        nn.init.orthogonal_(self.weight_hh)

    def zero_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> GateState:
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        hyper_h = torch.zeros(batch_size, self.hyper_hidden_size, device=device, dtype=dtype)
        hyper_c = torch.zeros(batch_size, self.hyper_hidden_size, device=device, dtype=dtype)
        return h, c, hyper_h, hyper_c

    def forward(self, x: Tensor, state: GateState) -> Tuple[Tensor, GateState]:
        h_prev, c_prev, hyper_h_prev, hyper_c_prev = state

        hyper_input = torch.cat([h_prev, x], dim=-1)
        hyper_h, hyper_c = self.hyper_cell(hyper_input, (hyper_h_prev, hyper_c_prev))

        x_parts = F.linear(x, self.weight_ih).chunk(4, dim=-1)
        h_parts = F.linear(h_prev, self.weight_hh).chunk(4, dim=-1)

        gate_outputs = {}
        for gate_name, x_part, h_part in zip(GATES, x_parts, h_parts):
            scale_h, scale_x, bias = self.modulators[gate_name](hyper_h)
            gate_value = scale_h * h_part + scale_x * x_part + bias
            if self.use_layer_norm:
                gate_value = self.gate_norms[gate_name](gate_value)
            gate_outputs[gate_name] = gate_value

        i_t = gate_outputs["i"]
        g_t = gate_outputs["g"]
        f_t = gate_outputs["f"]
        o_t = gate_outputs["o"]

        candidate = torch.tanh(g_t)
        if self.recurrent_dropout > 0.0:
            candidate = F.dropout(candidate, p=self.recurrent_dropout, training=self.training)

        c_t = torch.sigmoid(f_t + self.forget_bias) * c_prev + torch.sigmoid(i_t) * candidate
        if self.use_layer_norm:
            h_t = torch.sigmoid(o_t) * torch.tanh(self.cell_norm(c_t))
        else:
            h_t = torch.sigmoid(o_t) * torch.tanh(c_t)

        return h_t, (h_t, c_t, hyper_h, hyper_c)


class HyperLSTM(nn.Module):
    """Sequence wrapper around one or more HyperLSTM cells."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        hyper_hidden_size: int = 64,
        hyper_embedding_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        batch_first: bool = True,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first

        cells: List[HyperLSTMCell] = []
        for layer_idx in range(num_layers):
            layer_input_size = input_size if layer_idx == 0 else hidden_size
            cells.append(
                HyperLSTMCell(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    hyper_hidden_size=hyper_hidden_size,
                    hyper_embedding_size=hyper_embedding_size,
                    recurrent_dropout=recurrent_dropout,
                    use_layer_norm=use_layer_norm,
                )
            )
        self.layers = nn.ModuleList(cells)

    def zero_state(
        self,
        batch_size: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> StackedState:
        return [layer.zero_state(batch_size, device=device, dtype=dtype) for layer in self.layers]

    def forward(
        self,
        inputs: Tensor,
        state: Optional[StackedState] = None,
    ) -> Tuple[Tensor, StackedState]:
        if inputs.dim() != 3:
            raise ValueError("inputs must have shape [batch, time, features] or [time, batch, features]")

        if not self.batch_first:
            inputs = inputs.transpose(0, 1)

        batch_size, seq_len, _ = inputs.shape
        if state is None:
            state = self.zero_state(batch_size, device=inputs.device, dtype=inputs.dtype)

        current_inputs = inputs
        next_state: StackedState = []

        for layer_idx, cell in enumerate(self.layers):
            outputs = []
            layer_state = state[layer_idx]

            for time_idx in range(seq_len):
                output, layer_state = cell(current_inputs[:, time_idx, :], layer_state)
                outputs.append(output.unsqueeze(1))

            current_inputs = torch.cat(outputs, dim=1)
            if layer_idx + 1 < self.num_layers and self.dropout > 0.0:
                current_inputs = F.dropout(current_inputs, p=self.dropout, training=self.training)

            next_state.append(layer_state)

        if not self.batch_first:
            current_inputs = current_inputs.transpose(0, 1)

        return current_inputs, next_state
