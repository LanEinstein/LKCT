from .lit import LIT


def build_model(model_type):
    if model_type == 'lit':
        model = LIT(img_size=224,
                    patch_size=4,
                    in_chans=3,
                    num_classes=1,
                    embed_dim=128,
                    depths=[2, 2, 18, 2],
                    num_heads=[4, 8, 16, 32],
                    mlp_ratio=4.,
                    qkv_bias=True,
                    qk_scale=None,
                    drop_rate=0.0,
                    drop_path_rate=0.5,
                    ape=False,
                    patch_norm=True,
                    use_checkpoint=False,
                    alpha=0.4,
                    local_ws=[0, 0, 2, 1]
                    )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
