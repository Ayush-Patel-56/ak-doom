"""Shared neural decision path for PPO and inference; cadence never selects a goal's identity."""
from dataclasses import dataclass

import torch

from .brain import sample_actions

GOALS = ('Explore', 'Fight', 'Collect', 'Go to landmark', 'Go to exit')


@dataclass
class GoalState:
    goal: torch.Tensor
    remaining: torch.Tensor

    @classmethod
    def initial(cls, batch, device):
        return cls(torch.zeros(batch, dtype=torch.long, device=device),
                   torch.zeros(batch, dtype=torch.long, device=device))


def policy_step(net, features, state, starts, goals, generator=None, greedy=False):
    """One real decision. Manager samples only at a boundary; worker samples every step."""
    x = {k: v[None] for k, v in features.items()}
    h, state, visual = net.recurrent(x, state, starts[None])
    h = h[0]
    aux = {'hidden': h}
    if visual is not None:
        aux['exit_logits'] = net.exit_head(visual[0]).squeeze(-1)
    if net.cfg.use_manager:
        boundary = (goals.remaining == 0) | starts
        logits = net.manager(h)
        selected = goals.goal.clone()
        if boundary.any():
            active = logits[boundary]
            selected[boundary] = (active.argmax(-1) if greedy else
                                  torch.multinomial(active.softmax(-1), 1, generator=generator).squeeze(-1))
        remaining = torch.where(boundary, net.cfg.goal_interval, goals.remaining) - 1
        goals = GoalState(selected, remaining)
        aux.update(goal=selected, boundary=boundary, manager_logits=logits,
                   manager_value=net.manager_value(h).squeeze(-1),
                   manager_logp=logits.log_softmax(-1).gather(-1, selected[:, None]).squeeze(-1))
        worker = net.worker(h, selected)
    else:
        worker = net.worker(h)
    choices = sample_actions(worker, generator, greedy)
    aux['worker_logits'] = worker
    return choices, net.value(h).squeeze(-1), state, goals, aux
