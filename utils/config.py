config = {
    'sr': 88200,
    'frame_rate': 60,
    'hidden_dim': 1024,
    'n_layers': 8, 
    'num_heads': 16, 
    'dropout': 0.0,
    'output_dim': 68, # if you trained your own, this should also be 61
    'input_dim': 256,
    'frame_size': 64, 
    'overlap': 16,
    'use_half_precision': True
}