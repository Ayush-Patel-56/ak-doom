import numpy as np
import torch
import pytest
from smriti.brain import BrainConfig,SmritiNet,load_checkpoint,VITALS_SCALE
from smriti.features import empty
from training.warmstart import warm_start_from
from training.checkpoints import save_checkpoint


def test_actual_transition_targets_and_terminal_mask():
    from training.dynamics import transition_targets
    a,b=empty((2,)),empty((2,))
    b['ray_dist'][:]=.25;b['vitals'][:]=VITALS_SCALE
    targets,mask=transition_targets(a,b,np.array([False,True]))
    np.testing.assert_allclose(targets[0,:64],-.25)
    np.testing.assert_allclose(targets[0,64:],1.)
    assert not mask[1] and np.all(targets[1]==0)


def test_auxiliary_gradients_and_legacy_warm_start(tmp_path):
    from training.dynamics import dynamics_loss
    source=SmritiNet(BrainConfig())
    target=SmritiNet(BrainConfig(use_dynamics=True))
    warm_start_from(target,source)
    f={k:torch.as_tensor(v) for k,v in empty((2,1)).items()}
    f['ray_dist'].fill_(.5)
    old=source(f,source.initial_state(1))
    new=target(f,target.initial_state(1),return_aux=True)
    torch.testing.assert_close(old[0],new[0]);torch.testing.assert_close(old[1],new[1])
    loss=dynamics_loss(target,new[3]['hidden'],torch.zeros((2,1,6),dtype=torch.long),
                       torch.ones((2,1,81)),torch.ones((2,1),dtype=torch.bool))
    loss.backward()
    assert target.dynamics[0].weight.grad.abs().sum()>0
    assert target.rays.conv1.weight.grad.abs().sum()>0
    zero=dynamics_loss(target,new[3]['hidden'],torch.zeros((2,1,6),dtype=torch.long),
                       torch.full((2,1,81),float('nan')),torch.zeros((2,1),dtype=torch.bool))
    assert torch.isfinite(zero) and zero==0
    path=save_checkpoint(tmp_path/'old.pt',source)
    raw=torch.load(path,weights_only=True);raw['config'].pop('use_dynamics');torch.save(raw,path)
    assert not load_checkpoint(path)[0].cfg.use_dynamics


def test_auxiliary_cli_rejects_bad_coefficients():
    from training.train import build_args
    assert build_args(['--dynamics-coef','.1']).dynamics_coef==.1
    for value in ('-1','nan','inf'):
        with pytest.raises(SystemExit):build_args(['--dynamics-coef',value])
