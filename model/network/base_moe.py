import torch
import torch.nn as nn
from collections import Counter
from torch.distributions.normal import Normal
from .adapter import Adapter
import math

#没有global_task_id
val_task_id = None

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # print(self._num_experts)
        # sort experts
        # print('gates', gates.shape) # 64, 22
        # [[0.0000, 0.0000, 0.5146, 0.4854, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #         [0.0000, 0.0000, 0.0000, 0.0000, 0.4666, 0.5334, 0.0000, 0.0000, 0.0000]]
        # print(torch.nonzero(gates).shape)  # torch.Size([128, 2])
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        # print(sorted_experts.shape, index_sorted_experts.shape) # torch.Size([128, 2]) torch.Size([128, 2])
        # [[0, 2],[0, 3],[1, 4],[1, 5]] sorted_experts 将feature和experts匹配上
        # [[1, 0],[0, 1],[2, 2],[3, 3]]

        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # print(self._batch_index)
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes

        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space

        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)  # weight


        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        # combine samples that have been processed by the same k experts

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class BaseMoE(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, args, tag: str=''):
        super().__init__()

        # self.layer = i#记录模块所在架构位置
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.task_id = args.task_id
        self.noise_epsilon = 1e-2
        self.input_dim = input_dim
        # output_dim=input_dim if output_dim is None else output_dim
        self.output_dim = output_dim
        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()
        self.apply_moe = args.apply_moe
        self.noisy_gating = True
        self.top_k = args.topk
        self.num_experts = args.num_experts  # e = 22
        self.router_num = 1  # n = 1
        self.tag = tag # 标记模块位置，用于组成router
        # choose map: 记录专家被选择的次数
        self.choose_map = torch.zeros([self.num_experts])
        self.rank = args.rank                               # low rank
        self.autorouter = args.autorouter
        self.disable_finetune=args.disable_finetune           #是否禁用微调
        self.adaptmlp_list = nn.ModuleList()

        if self.apply_moe:
            if self.task_id > -1:  # router > 1
                self.router_list = nn.ParameterList()
                self.w_noise_list = nn.ParameterList()
                for _ in range(args.num_tasks):  # Task number
                    self.router_list.append(nn.Parameter(torch.zeros(input_dim, self.num_experts), requires_grad=True))
                    self.w_noise_list.append(nn.Parameter(torch.zeros(input_dim, self.num_experts), requires_grad=True))
                for _ in range(self.num_experts):  # Expert number
                    self.adaptmlp_list.append(Adapter(input_dim=input_dim, output_dim=output_dim, dropout=0.1, bottleneck=self.rank,
                                                     init_option='lora',
                                                     adapter_scalar=0.1,
                                                     adapter_layernorm_option='none',
                                                     ))
            else:  # one router for all task
                self.router1 = nn.Parameter(torch.zeros(input_dim, self.num_experts), requires_grad=True)
                self.w_noise = nn.Parameter(torch.zeros(input_dim, self.num_experts), requires_grad=True)
                for _ in range(self.num_experts):
                    self.adaptmlp_list.append(Adapter(input_dim=input_dim, output_dim=output_dim, dropout=0.1, bottleneck=self.rank,
                                                     init_option='lora',
                                                     adapter_scalar=0.1,
                                                     adapter_layernorm_option='none',
                                                     ))
        else:  # without moe, just finetuning
                self.adaptmlp = Adapter(input_dim=input_dim, output_dim=output_dim, dropout=0.1, bottleneck=self.rank,
                                        init_option='lora',
                                        adapter_scalar=0.1,
                                        adapter_layernorm_option='none',
                                        )

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and self.training:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.top_k < self.num_experts and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x: torch.Tensor):
        need_unsqueeze=len(x.shape)==2
        if need_unsqueeze:
            x=x.unsqueeze(1)

        if self.disable_finetune:
            return x
        
        if not self.apply_moe or self.num_experts<=1:   #non-MoE, just finetune single expert
            x_re = x.permute(1, 0, 2)
            adapt_x = self.adaptmlp(x_re, add_residual=False).permute(1, 0, 2)
            x = x + adapt_x
            return x

        # 启用lora adapter微调
        if self.task_id > -1:                                                                            # multi router
            if self.autorouter and val_task_id == -1:                                                    # 验证阶段
                return x
            x_re = x.permute(1, 0, 2)[:, 0, :]  # ->shape[seq_len, batch_size, input_dim] 取出所有样本第一个时间步的特征
            if val_task_id is not None and self.autorouter:
                gates, load = self.noisy_top_k_gating(x_re, self.router_list[val_task_id],
                                                     self.w_noise_list[val_task_id])
            else:
                gates, load = self.noisy_top_k_gating(x_re, self.router_list[self.task_id],
                                                    self.w_noise_list[self.task_id])
            importance = gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
            loss *= 1e-2  # Todo

            nonzero_indices = torch.nonzero(gates)
            counter = Counter(nonzero_indices[:, 1].tolist())
            for number, count in counter.items():
                self.choose_map[number] += count
        else:                                                                                               #single router
            x_re = x.permute(1, 0, 2)[:, 0, :]
            gates, load = self.noisy_top_k_gating(x_re, self.router1, self.w_noise)
        
        # MoE 结果处理(multi/single router都需要)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).reshape(x.shape[1], -1))                        #view仅适用于permute后在内存连续的tensor，否则reshape

        expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].reshape(expert_inputs[i].shape[0],
                                                                      x.shape[0], x.shape[2]).to(x), add_residual=False)
                          for i in range(self.num_experts)]
        # 打印当前参与输出的专家索引
        # active_experts = torch.nonzero(gates.sum(0) > 0).squeeze()
        # print(f"Active experts: {active_experts.tolist()}")
        i = 0
        while i < len(expert_outputs):
            if expert_outputs[i].shape[0] == 0:
                expert_outputs.pop(i)
            else:
                expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1)
                i += 1

        y = dispatcher.combine(expert_outputs)
        y = y.view(x.shape[1], x.shape[0], x.shape[2])
        x = x + y.permute(1, 0, 2)

        if need_unsqueeze:
            x=x.squeeze(1)

        return x
    
