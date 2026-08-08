import torch
import torch.nn as nn
from .transformer.embed_unit import EmbedNet
from .transformer.block_unit import ResidualBlock, TransformerBlock
from .transformer.policy_unit import TPolicyNetwork, PolicyNetwork
from .transformer.value_unit import TValueNetwork, ValueNetwork


class TransformerAlphaZeroNetwork(nn.Module):
    """ResTNet: an AlphaZero network whose backbone mixes residual ('R') and
    transformer ('T') blocks, e.g. blocks_type="R_T_T".
    ref: Bridging Local and Global Knowledge via Transformer in Board Games (IJCAI 2025)
    """

    def __init__(self,
                 game_name,
                 num_input_channels,
                 input_channel_height,
                 input_channel_width,
                 num_hidden_channels,
                 hidden_channel_height,
                 hidden_channel_width,
                 action_size,
                 num_value_hidden_channels,
                 discrete_value_size,
                 embed_kernel_size,
                 blocks_type,
                 policy_type,
                 value_type):
        super(TransformerAlphaZeroNetwork, self).__init__()
        assert discrete_value_size == 1, "transformer blocks only support a scalar value head (discrete_value_size == 1)"
        self.game_name = game_name
        self.num_input_channels = num_input_channels
        self.input_channel_height = input_channel_height
        self.input_channel_width = input_channel_width
        self.num_hidden_channels = num_hidden_channels
        self.hidden_channel_height = hidden_channel_height
        self.hidden_channel_width = hidden_channel_width
        self.action_size = action_size
        self.num_value_hidden_channels = num_value_hidden_channels
        self.discrete_value_size = discrete_value_size

        self.num_head = 4
        self.mlp_ratio = 2
        self.blocks_type = blocks_type
        self.num_blocks = len(blocks_type.split("_"))

        self.embed = EmbedNet(num_input_channels, num_hidden_channels, embed_kernel_size)
        self.blocks = nn.ModuleList([self.get_backbone(block_type) for block_type in blocks_type.split("_")])
        self.policy = self.get_policy_net(policy_type)
        self.value = self.get_value_net(value_type)

        self.apply(self._init_weights_trunc_normal)

    def get_backbone(self, block_type):
        if block_type == "R":
            return ResidualBlock(self.num_hidden_channels, self.input_channel_height)
        elif block_type == "T":
            return TransformerBlock(self.num_hidden_channels,
                                    self.num_hidden_channels * self.mlp_ratio,
                                    self.num_head,
                                    self.input_channel_height,
                                    self.input_channel_width)
        else:
            raise ValueError(f"unsupported nn_blocks_type block '{block_type}', expected 'R' or 'T'")

    def get_policy_net(self, policy_type):
        if policy_type == "TP":
            return TPolicyNetwork(self.num_hidden_channels, self.input_channel_height, self.input_channel_width, self.action_size)
        elif policy_type == "P":
            return PolicyNetwork(self.num_hidden_channels, self.input_channel_height, self.input_channel_width, self.action_size)
        else:
            raise ValueError(f"unsupported nn_policy_type '{policy_type}', expected 'P' or 'TP'")

    def get_value_net(self, value_type):
        if value_type == "TV":
            return TValueNetwork(self.num_hidden_channels, self.input_channel_height)
        elif value_type == "V":
            return ValueNetwork(self.num_hidden_channels, self.input_channel_height, self.input_channel_width, self.num_value_hidden_channels)
        else:
            raise ValueError(f"unsupported nn_value_type '{value_type}', expected 'V' or 'TV'")

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.export
    def get_type_name(self):
        return "alphazero"

    @torch.jit.export
    def get_game_name(self):
        return self.game_name

    @torch.jit.export
    def get_num_input_channels(self):
        return self.num_input_channels

    @torch.jit.export
    def get_input_channel_height(self):
        return self.input_channel_height

    @torch.jit.export
    def get_input_channel_width(self):
        return self.input_channel_width

    @torch.jit.export
    def get_num_hidden_channels(self):
        return self.num_hidden_channels

    @torch.jit.export
    def get_hidden_channel_height(self):
        return self.hidden_channel_height

    @torch.jit.export
    def get_hidden_channel_width(self):
        return self.hidden_channel_width

    @torch.jit.export
    def get_num_blocks(self):
        return self.num_blocks

    @torch.jit.export
    def get_action_size(self):
        return self.action_size

    @torch.jit.export
    def get_num_value_hidden_channels(self):
        return self.num_value_hidden_channels

    @torch.jit.export
    def get_discrete_value_size(self):
        return self.discrete_value_size

    def forward(self, state):
        x = self.embed(state)
        for block in self.blocks:
            x = block(x)

        policy_logit = self.policy(x)
        policy = torch.softmax(policy_logit, dim=1)
        value = self.value(x)

        return {"policy_logit": policy_logit, "policy": policy, "value": value}
