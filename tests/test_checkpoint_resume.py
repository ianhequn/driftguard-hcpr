import torch

from driftguard.training.vspw_segformer import load_training_checkpoint, save_training_checkpoint


def test_checkpoint_restores_optimizer_scheduler_step_and_best(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "state.pt"
    save_training_checkpoint(path, model, optimizer, scheduler, step=7, best_miou=0.25)
    restored = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    state = load_training_checkpoint(path, restored, restored_optimizer, restored_scheduler)
    assert state == {"step": 7, "best_miou": 0.25}
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
