trading = {
    "TRADING_CHARGE": 0.00015,
    "TRADING_TAX": 0.0025,
}

setting = {"PRICE_IDX": -1, "window": 50}  # 종가의 위치

network_setting = {
    "d_model": 256,
    "actor_hidden_dim": 2048,
    "critic_hidden_dim": 2048,
    "transformer_layers": 2,
    "nhead": 4,
    "dropout": 0.1,
    "max_seq_length": 200,
}

trainer_setting = {
    "lr_encoder": 3e-4,
    "lr_actor": 3e-4,
    "lr_critic": 1e-3,
    "gamma": 0.99,
    "K_epochs": 5,
    "batch_size": 128,
    "eps_clip": 0.2,
    "checkpoint_path": "./weight/policy_weights.pt",
}
