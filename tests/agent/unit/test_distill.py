import numpy as np
import pytest
import torch
from smriti.brain import BrainConfig, SmritiNet
from smriti.features import empty
from training.checkpoints import save_checkpoint


def test_teacher_is_frozen_independent_and_resets(tmp_path):
    from training.distill import FrozenTeacher
    path=tmp_path/'teacher.pt'
    net=SmritiNet(BrainConfig(hidden=128,use_manager=True))
    save_checkpoint(path,net,{'steps':16})
    before=torch.get_rng_state().clone()
    teacher=FrozenTeacher(path,2,'cpu',seed=8)
    assert torch.equal(before,torch.get_rng_state())
    original={k:v.clone() for k,v in teacher.net.state_dict().items()}
    f=empty((2,)); starts=np.ones(2,bool)
    first=teacher.logits(f,starts)
    assert not first.requires_grad and not any(p.requires_grad for p in teacher.net.parameters())
    assert teacher.state.shape==(2,128) and teacher.goals.remaining.tolist()==[7,7]
    teacher.logits(f,np.zeros(2,bool));assert teacher.goals.remaining.tolist()==[6,6]
    teacher.logits(f,starts);assert teacher.goals.remaining.tolist()==[7,7]
    assert torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(v,original[k]) for k,v in teacher.net.state_dict().items())
    assert teacher.provenance['steps']==16 and teacher.provenance['sha256']


def test_kl_has_student_gradients_and_rejects_bad_temperature():
    from training.distill import distillation_loss
    student=torch.randn(3,2,15,requires_grad=True);teacher=torch.randn_like(student,requires_grad=True)
    loss=distillation_loss(student,teacher,2.)
    loss.backward()
    assert student.grad.abs().sum()>0 and teacher.grad is None
    assert distillation_loss(student,student,1.).abs()<1e-6
    for temp in (0,-1,float('nan')):
        with pytest.raises(ValueError):distillation_loss(student,teacher,temp)


def test_random_or_missing_teacher_is_rejected(tmp_path):
    from training.distill import FrozenTeacher
    path=tmp_path/'random.pt';save_checkpoint(path,SmritiNet(BrainConfig()))
    with pytest.raises(ValueError,match='trained'):FrozenTeacher(path,1,'cpu')
    with pytest.raises(FileNotFoundError):FrozenTeacher(tmp_path/'missing',1,'cpu')


def test_distillation_cli_validation():
    from training.train import build_args
    assert build_args(['--distill-coef','0']).distill_coef==0
    for args in (['--distill-coef','1'],['--distill-temperature','0'],['--distill-coef','nan']):
        with pytest.raises(SystemExit): build_args(args)


def test_zero_coefficient_does_not_use_teacher_targets():
    import copy
    from training.ppo import PPO,PPOConfig,Rollout
    net=SmritiNet(BrainConfig())
    a,b=PPO(net,PPOConfig(T=2,epochs=1,minibatches=1),1,'cpu'),PPO(copy.deepcopy(net),PPOConfig(T=2,epochs=1,minibatches=1),1,'cpu')
    x,y=Rollout(2,1),Rollout(2,1)
    y.teacher_logits=np.full((2,1,15),float('nan'),np.float32)  # ignored while disabled
    for p,buf in ((a,x),(b,y)):
        np.random.seed(7)
        p.update(buf,p.net.initial_state(1),np.zeros(1,np.float32))
    assert all(torch.equal(v,b.net.state_dict()[k]) for k,v in a.net.state_dict().items())


def test_actual_stage5_distillation_resume_and_teacher_identity():
    import shutil,uuid
    from pathlib import Path
    from training.train import main
    from smriti.brain import load_checkpoint
    name='distill-test-'+uuid.uuid4().hex[:8]
    root=Path(__file__).resolve().parents[3];out=root/'runs'/name
    source=root/'submission/weights/champion.pt'
    args=['--stage','5','--kinds','arena','--memory','on','--teacher',str(source),'--distill-coef','.2',
          '--workers','1','--games','1','--T','8','--epochs','1','--minibatches','1','--threads','1',
          '--ticks','40','--name',name]
    try:
        assert main(args+['--steps','8'])==0
        assert main(args+['--steps','16','--resume'])==0
        _,extra=load_checkpoint(out/'latest.pt')
        assert extra['training']['distillation']['teacher']['steps']==load_checkpoint(source)[1]['steps']
        # Positive decision count is actual upstream source provenance, not assumed policy quality.
        changed=args.copy();changed[changed.index(str(source))]=str(out/'latest.pt')
        with pytest.raises(ValueError,match='config differs'):main(changed+['--steps','24','--resume'])
    finally:shutil.rmtree(out,ignore_errors=True)
