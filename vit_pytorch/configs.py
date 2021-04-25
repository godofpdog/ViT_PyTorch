""" model configs """

_B_8_384 = {
    'image_size': 384,
    'patch_size': 8,
    'in_channels': 3,
    'num_classes': 1000,
    'embed_dim': 768,
    'num_layers': 12,
    'num_heads': 12,
    'hidden_dim': 3072,
    'atten_drop': 0.0,
    'proj_drop': 0.1,
    'repr_dim': None,
    'classifier': 'token'
}

_B_16_224 = {
    'image_size': 224,
    'patch_size': 16,
    'in_channels': 3,
    'num_classes': 1000,
    'embed_dim': 768,
    'num_layers': 12,
    'num_heads': 12,
    'hidden_dim': 3072,
    'atten_drop': 0.0,
    'proj_drop': 0.1,
    'repr_dim': None,
    'classifier': 'token' 
}

_B_16_384 = {
    'image_size': 384,
    'patch_size': 16,
    'in_channels': 3,
    'num_classes': 1000,
    'embed_dim': 768,
    'num_layers': 12,
    'num_heads': 12,
    'hidden_dim': 3072,
    'atten_drop': 0.0,
    'proj_drop': 0.1,
    'repr_dim': None,
    'classifier': 'token'
}

_B_32_384 = {
    'image_size': 384,
    'patch_size': 32,
    'in_channels': 3,
    'num_classes': 1000,
    'embed_dim': 768,
    'num_layers': 12,
    'num_heads': 12,
    'hidden_dim': 3072,
    'atten_drop': 0.0,
    'proj_drop': 0.1,
    'repr_dim': None,
    'classifier': 'token'
}

_L_16_224 = {
    'image_size': 224,
    'patch_size': 16,
    'in_channels': 3,
    'num_classes': 1000,
    'embed_dim': 1024,
    'num_layers': 24,
    'num_heads': 16,
    'hidden_dim': 4096,
    'atten_drop': 0.0,
    'proj_drop': 0.1,
    'repr_dim': None,
    'classifier': 'token' 
}

_L_16_384 = {
    'image_size': 384,
    'patch_size': 16,
    'in_channels': 3,
    'num_classes': 1000,
    'embed_dim': 1024,
    'num_layers': 24,
    'num_heads': 16,
    'hidden_dim': 4096,
    'atten_drop': 0.0,
    'proj_drop': 0.1,
    'repr_dim': None,
    'classifier': 'token' 
}

_L_32_384 = {
    'image_size': 384,
    'patch_size': 32,
    'in_channels': 3,
    'num_classes': 1000,
    'embed_dim': 1024,
    'num_layers': 24,
    'num_heads': 16,
    'hidden_dim': 4096,
    'atten_drop': 0.0,
    'proj_drop': 0.1,
    'repr_dim': None,
    'classifier': 'token' 
}

_21K_B_16_224 = {
    'image_size': 224,
    'patch_size': 16,
    'in_channels': 3,
    'num_classes': 21843,
    'embed_dim': 768,
    'num_layers': 12,
    'num_heads': 12,
    'hidden_dim': 3072,
    'atten_drop': 0.0,
    'proj_drop': 0.1,
    'repr_dim': None,
    'classifier': 'token'
}

MODEL_CFGS = {
    'B_8_384': _B_8_384,
    'B_16_224': _B_16_224,
    'B_16_384': _B_16_384,
    'B_32_384': _B_32_384,
    'L_16_224': _L_16_224,
    'L_16_384': _L_16_384,
    'L_32_384': _L_32_384,
    # '21K_B_16_384': _21K_B_16_384,
    '21K_B_16_224': _21K_B_16_224,
} 
