import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn

from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, KL
from .cluster import PCM, Att_Block_Patch

allgather = AllGather.apply
allgather2 = AllGather2.apply

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int), nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class My_Model(nn.Module):
    def __init__(self, config):

        super(My_Model, self).__init__()

        self.config = config

        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        if torch.cuda.is_available():
            convert_weights(self.clip)

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)
        self.apply(self.init_weights)
        self.clip.load_state_dict(state_dict, strict=False)

        new_state_dict = OrderedDict()
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue

        self.alpha = self.config.alpha
        self.beta = self.config.beta
        self.loss_fct = CrossEn(config)
        embed_dim = state_dict["text_projection"].shape[1]
        sample_ratio = 0.5
        self.v_pcm_p_1 = PCM(sample_ratio=sample_ratio, embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_1 = Att_Block_Patch(dim=embed_dim, num_heads=8)
        self.v_pcm_p_2 = PCM(sample_ratio=sample_ratio, embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_2 = Att_Block_Patch(dim=embed_dim, num_heads=8)
        self.v_pcm_p_3 = PCM(sample_ratio=sample_ratio, embed_dim=embed_dim, dim_out=embed_dim, k=3)
        self.v_att_block_p_3 = Att_Block_Patch(dim=embed_dim, num_heads=8)

        self.word_weights = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1),
        )

        self.frame_weights = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1),
        )

        self.patch_weights = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1),
        )

        self.kl = KL()

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, text, text_mask, video, video_mask, idx=None, global_step=0):

        text_mask = text_mask.view(-1, text_mask.shape[-1])
        text = text.view(-1, text.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat, w_feat, f_feat, p_feat = self.get_text_video_feat(text, text_mask, video, video_mask, shaped=True)

        if self.training:
            if torch.cuda.is_available():
                idx = allgather(idx, self.config)
                text_mask = allgather(text_mask, self.config)
                s_feat = allgather(s_feat, self.config)
                w_feat = allgather(w_feat, self.config)
                video_mask = allgather(video_mask, self.config)
                f_feat = allgather(f_feat, self.config)
                p_feat = allgather(p_feat, self.config)
                torch.distributed.barrier()

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            logit_scale = self.clip.logit_scale.exp()
            loss = 0.

            ### See all (sentence & frame)
            b, f, d = f_feat.shape
            sims_f = torch.einsum('ad,bfd->abf', [s_feat, f_feat])
            s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
            f_feat = f_feat / f_feat.norm(dim=-1, keepdim=True)
            sims_f = sims_f.diagonal(dim1=0, dim2=1).transpose(0, 1)
            max_val, max_idx = torch.topk(sims_f, k=f//2, dim=-1)
            max_idx, _ = torch.sort(max_idx, dim=-1)

            # update frame features \in [B, f, D]
            f_feat = f_feat[torch.arange(b)[:, None], max_idx, :]
            f_w = self.frame_weights(f_feat).squeeze(-1)
            s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
            f_feat = f_feat / f_feat.norm(dim=-1, keepdim=True)
            sims_s_f_logits = torch.einsum("ad,bfd->abf", [s_feat, f_feat])
            sims_s_f = torch.einsum("abf,bf->ab", [sims_s_f_logits, f_w])
            loss_s_f = self.loss_fct(sims_s_f * logit_scale) + self.loss_fct(sims_s_f.T * logit_scale)

            ### See one (sentence & patch, word & patch)
            p_feat = p_feat.reshape(b, f, -1, d)
            p_feat = p_feat[torch.arange(b)[:, None], max_idx, :, :]
            # update patch features \in [B, p, D]
            p_feat = p_feat.reshape(b, -1, d)

            p_idx_token = torch.arange(p_feat.size(1))[None, :].repeat(p_feat.size(0), 1)
            p_agg_weight = p_feat.new_ones(p_feat.size(0), p_feat.size(1), 1)
            p_mask = torch.ones(p_feat.size(0), p_feat.size(1)).to(p_feat.device)
            p_token_dict = {'x': p_feat,
                            'token_num': p_feat.size(1),
                            'idx_token': p_idx_token,
                            'agg_weight': p_agg_weight,
                            'mask': p_mask.detach()}
            p_token_dict = self.v_att_block_p_1(self.v_pcm_p_1(p_token_dict))
            p_token_dict = self.v_att_block_p_2(self.v_pcm_p_2(p_token_dict))
            p_token_dict = self.v_att_block_p_3(self.v_pcm_p_3(p_token_dict))
            p_feat = p_token_dict["x"]

            p_w = self.patch_weights(p_feat).squeeze(-1)   # [B, P, D] => [B, P]
            w_w = self.word_weights(w_feat).squeeze(-1)    # [B, N, D] => [B, N]

            p_feat = p_feat / p_feat.norm(dim=-1, keepdim=True)
            w_feat = w_feat / w_feat.norm(dim=-1, keepdim=True)
            s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)

            # sentence & patch
            sims_s_p_logits = torch.einsum("ad,bpd->abp", [s_feat, p_feat])
            sims_s_p = torch.einsum("abp,bp->ab", [sims_s_p_logits, p_w])
            loss_s_p = self.loss_fct(sims_s_p * logit_scale) + self.loss_fct(sims_s_p.T * logit_scale)

            # word & patch
            sims_w_p_logits = torch.einsum("awd,bpd->abwp", [w_feat, p_feat])
            w2p_logits, _ = sims_w_p_logits.max(dim=-1)
            w2p_logits = torch.einsum('abw,bw->ab', [w2p_logits, w_w])
            p2w_logits, _ = sims_w_p_logits.max(dim=-2)
            p2w_logits = torch.einsum('abp,bp->ab', [p2w_logits, p_w])
            sims_w_p = (w2p_logits + p2w_logits) / 2.0
            loss_w_p = self.loss_fct(sims_w_p * logit_scale) + self.loss_fct(sims_w_p.T * logit_scale)

            loss = loss + loss_s_f + loss_s_p * self.alpha + loss_w_p * self.beta
            return loss
        else:
            return None

    def sim_matrix_training(self, t_feat, v_feat):
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        v_feat = v_feat / v_feat.norm(dim=-1, keepdim=True)

        sims = torch.mm(t_feat, v_feat.t())

        return sims

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        s_feat, w_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        s_feat = s_feat.float()
        s_feat = s_feat.view(bs_pair, -1, s_feat.size(-1))
        w_feat = w_feat.float()
        w_feat = w_feat.view(bs_pair, -1, w_feat.size(-1))
        return s_feat, w_feat

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        f_feat, p_feat = self.clip.encode_image(video, return_hidden=True)
        f_feat = f_feat.float()
        p_feat = p_feat.float()
        f_feat = f_feat.float().view(bs_pair, -1, f_feat.size(-1))
        p_feat = p_feat.float().view(bs_pair, -1, p_feat.size(-1))

        return f_feat, p_feat

    def get_text_video_feat(self, text, text_mask, video, video_mask, shaped=False):
        if shaped is False:
            text = text.view(-1, text.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat, w_feat = self.get_text_feat(text, text_mask, shaped=True)
        f_feat, p_feat = self.get_video_feat(video, video_mask, shaped=True)

        return s_feat.squeeze(1), w_feat, f_feat, p_feat

    def get_similarity_logits(self, text_mask, s_feat, w_feat, video_mask, f_feat, p_feat, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        ### See all (sentence & frame)
        b, f, d = f_feat.shape
        max_idx = torch.arange(0, f, 2).repeat(b, 1)
        # max_idx = torch.stack([torch.randperm(f)[:f // 2] for _ in range(b)], dim=0)
        max_idx, _ = torch.sort(max_idx, dim=-1)

        # update frame features \in [B, f, D]
        f_feat = f_feat[torch.arange(b)[:, None], max_idx, :]
        f_w = self.frame_weights(f_feat).squeeze(-1)
        s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)
        f_feat = f_feat / f_feat.norm(dim=-1, keepdim=True)
        sims_s_f_logits = torch.einsum("ad,bfd->abf", [s_feat, f_feat])
        sims_s_f = torch.einsum("abf,bf->ab", [sims_s_f_logits, f_w])

        ### See one (sentence & patch, word & patch)
        p_feat = p_feat.reshape(b, f, -1, d)
        p_feat = p_feat[torch.arange(b)[:, None], max_idx, :, :]
        # update patch features \in [B, p, D]
        p_feat = p_feat.reshape(b, -1, d)

        p_idx_token = torch.arange(p_feat.size(1))[None, :].repeat(p_feat.size(0), 1)
        p_agg_weight = p_feat.new_ones(p_feat.size(0), p_feat.size(1), 1)
        p_mask = torch.ones(p_feat.size(0), p_feat.size(1)).to(p_feat.device)
        p_token_dict = {'x': p_feat,
                        'token_num': p_feat.size(1),
                        'idx_token': p_idx_token,
                        'agg_weight': p_agg_weight,
                        'mask': p_mask.detach()}
        p_token_dict = self.v_att_block_p_1(self.v_pcm_p_1(p_token_dict))
        p_token_dict = self.v_att_block_p_2(self.v_pcm_p_2(p_token_dict))
        p_token_dict = self.v_att_block_p_3(self.v_pcm_p_3(p_token_dict))
        p_feat = p_token_dict["x"]

        p_w = self.patch_weights(p_feat).squeeze(-1)    # [B, P, D] => [B, P]
        w_w = self.word_weights(w_feat).squeeze(-1)     # [B, N, D] => [B, N]

        p_feat = p_feat / p_feat.norm(dim=-1, keepdim=True)
        w_feat = w_feat / w_feat.norm(dim=-1, keepdim=True)
        s_feat = s_feat / s_feat.norm(dim=-1, keepdim=True)

        # sentence & patch
        sims_s_p_logits = torch.einsum("ad,bpd->abp", [s_feat, p_feat])
        sims_s_p = torch.einsum("abp,bp->ab", [sims_s_p_logits, p_w])

        # word & patch
        sims_w_p_logits = torch.einsum("awd,bpd->abwp", [w_feat, p_feat])
        w2p_logits, _ = sims_w_p_logits.max(dim=-1)
        w2p_logits = torch.einsum('abw,bw->ab', [w2p_logits, w_w])
        p2w_logits, _ = sims_w_p_logits.max(dim=-2)
        p2w_logits = torch.einsum('abp,bp->ab', [p2w_logits, p_w])
        sims_w_p = (w2p_logits + p2w_logits) / 2.0

        sims_w = [1.0, 1.0, 1.0]
        return sims_s_f * sims_w[0] + sims_s_p * sims_w[1] + sims_w_p * sims_w[2]
