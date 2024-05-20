from models.bs_roformer.bs_roformer import BSRoformer

bsr_model = BSRoformer(
    dim=512,
    depth=12,
    stereo=True,
    num_stems=1,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    linear_transformer_depth=0,
    freqs_per_bands=tuple(
        [2] * 24 + [4] * 12 + [12] * 8 + [24] * 8 + [48] * 8 + [128, 129]
    ),
    dim_head=64,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
    flash_attn=True,
    dim_freqs_in=1025,
    stft_n_fft=2048,
    stft_hop_length=441,
    stft_win_length=2048,
    stft_normalized=False,
    mask_estimator_depth=2,
    multi_stft_resolution_loss_weight=1.0,
    multi_stft_resolutions_window_sizes=(4096, 2048, 1024, 512, 256),
    multi_stft_hop_size=147,
    multi_stft_normalized=False,
)