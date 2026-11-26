"""Action-conditioned prediction of measured legal next-observation deltas."""
from __future__ import annotations
import numpy as np
import torch
from smriti.brain import VITALS_SCALE
from smriti.policy.actions import GROUP_SIZES


def transition_targets(before,after,done):
    def values(features):
        # Ray distances are already normalized by the official observation wrapper.
        proximity=1.-np.clip(features['ray_dist'],0.,1.)
        vitals=np.clip(features['vitals']/VITALS_SCALE,-5.,5.)
        return np.concatenate((proximity,vitals),axis=-1)
    mask=~np.asarray(done,dtype=bool)
    targets=np.asarray(values(after)-values(before),np.float32)
    targets[~mask]=0.  # reset-replacement observations are never transition targets
    return targets,mask


def dynamics_loss(net,hidden,actions,targets,mask):
    if not mask.any(): return hidden.sum()*0.
    onehot=torch.cat([torch.nn.functional.one_hot(actions[...,i],n) for i,n in enumerate(GROUP_SIZES)],-1)
    prediction=net.dynamics(torch.cat((hidden[mask],onehot[mask].to(hidden.dtype)),-1))
    return torch.nn.functional.smooth_l1_loss(prediction,targets[mask].detach())
