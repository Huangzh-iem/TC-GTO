"""Single preregistered LF-transmissibility conditioned PF backbone."""
from __future__ import annotations
import torch
from torch import nn
from .hetero_graph_operator.config import HeteroGNOConfig
from .pf0_model import ParameterFreeSTGNO2L


class LFTransmissibilityConditionedSTGNO2L(ParameterFreeSTGNO2L):
    def __init__(self,config:HeteroGNOConfig)->None:
        super().__init__(config);h=config.hidden_dim
        # Six directed pairs share this tiny spectral encoder. Pair geometry is
        # concatenated as constant channels before the convolution.
        self.lf_pair_encoder=nn.Sequential(nn.Conv1d(7,16,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        self.lf_pair_projection=nn.Sequential(nn.Linear(16,32),nn.GELU())
        self.lf_context_projection=nn.Linear(32,h)
        self.register_buffer('pair_order',torch.tensor([(i,j) for i in range(3) for j in range(3) if i!=j],dtype=torch.long))

    def forward(self,sparse_observation:torch.Tensor,sensor_mask:torch.Tensor,
                floor_coordinate:torch.Tensor,valid_node_mask:torch.Tensor,
                lf_transmissibility_signature:torch.Tensor)->dict[str,torch.Tensor]:
        # signature [B,6,F,4] = scaled log-amplitude/cos/sin + coherence.
        if lf_transmissibility_signature.ndim!=4 or lf_transmissibility_signature.shape[1]!=6 or lf_transmissibility_signature.shape[-1]!=4:
            raise ValueError('expected LF signature [batch,6,frequency,4]')
        sensor_coords=floor_coordinate[:,[0,3,7]] if floor_coordinate.ndim==2 else floor_coordinate[[0,3,7]][None].expand(sparse_observation.shape[0],-1)
        src=sensor_coords[:,self.pair_order[:,0]];dst=sensor_coords[:,self.pair_order[:,1]]
        geometry=torch.stack([src,dst,dst-src],-1)[:,:,None,:].expand(-1,-1,lf_transmissibility_signature.shape[2],-1)
        pair_input=torch.cat([lf_transmissibility_signature.to(sparse_observation),geometry],-1)
        b,p,f,c=pair_input.shape;x=pair_input.reshape(b*p,f,c).transpose(1,2)
        pair=self.lf_pair_projection(self.lf_pair_encoder(x).squeeze(-1)).reshape(b,p,32)
        context=self.lf_context_projection(pair.mean(1))
        output=self._forward_impl(sparse_observation,sensor_mask,floor_coordinate,valid_node_mask,context)
        # Expose the already-computed frozen system token for downstream
        # diagnostic heads without changing TS1 predictions or parameters.
        output['global_lf_context']=context
        return output
