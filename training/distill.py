"""Frozen legal-observation recurrent teacher on student-visited observations."""
from __future__ import annotations
from dataclasses import asdict
import math
from pathlib import Path
import numpy as np
import torch
from smriti.brain import load_checkpoint
from smriti.features import FEATURE_SPEC
from smriti.hierarchy import GoalState, policy_step
from smriti.policy.actions import GROUP_SIZES
from scripts.test_use import digest


class FrozenTeacher:
    def __init__(self,path,batch,device,seed=0):
        path=Path(path)
        self.device=torch.device(device)
        devices=[self.device.index or 0] if self.device.type=='cuda' else []
        # Loading constructs a network; it must not advance the student's initialization RNG.
        with torch.random.fork_rng(devices=devices):
            self.net,extra=load_checkpoint(path,self.device)
        if type(extra.get('steps')) is not int or extra['steps']<=0:
            raise ValueError('teacher must be a trained checkpoint with positive decision provenance')
        self.net.eval().requires_grad_(False)
        self.state=self.net.initial_state(batch,self.device)
        self.goals=GoalState.initial(batch,self.device)
        self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.provenance=dict(sha256=digest(path),config=asdict(self.net.cfg),steps=extra['steps'],
                             observation_contract='official legal features; student-executed action history')

    @torch.no_grad()
    def logits(self,features,starts):
        x={k:torch.as_tensor(features[k],device=self.device) for k in FEATURE_SPEC}
        starts=torch.as_tensor(starts,device=self.device,dtype=torch.bool)
        _,_,self.state,self.goals,aux=policy_step(self.net,x,self.state,starts,self.goals,self.generator)
        return aux['worker_logits'].detach()


def distillation_loss(student,teacher,temperature=1.):
    if not math.isfinite(temperature) or temperature<=0: raise ValueError('temperature must be finite and positive')
    if student.shape!=teacher.shape or student.shape[-1]!=sum(GROUP_SIZES): raise ValueError('action logits mismatch')
    losses=[]
    for s,t in zip(student.split(GROUP_SIZES,-1),teacher.detach().split(GROUP_SIZES,-1)):
        losses.append(torch.nn.functional.kl_div((s/temperature).log_softmax(-1),
                      (t/temperature).softmax(-1),reduction='none').sum(-1).mean()*temperature**2)
    return torch.stack(losses).mean()
