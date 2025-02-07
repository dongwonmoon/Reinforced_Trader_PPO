trading = {
    "TRADING_CHARGE": 0.00015,
    "TRADING_TAX": 0.0025,
}

setting = {"PRICE_IDX": 3, "window": 50}  # 종가의 위치

network_setting = {
    "actor_hidden_dim": 128,
    "critic_hidden_dim": 128,
    "transformer_layers": 4,
    "nhead": 4,
    "dropout": 0.1,
}

trainer_setting = {
    "lr_encoder": 3e-4,
    "lr_actor": 3e-4,
    "lr_critic": 1e-3,
    "gamma": 0.99,
    "K_epochs": 20,
    "eps_clip": 0.2,
    "checkpoint_path": "./model/",
}
