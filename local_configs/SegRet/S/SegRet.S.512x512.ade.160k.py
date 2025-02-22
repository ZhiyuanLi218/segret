_base_ = [
    '../../_base_/models/SegRet.py',
    '../../_base_/datasets/ade20k_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)  # SyncBN
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/RMT_S_pre.pth',
    backbone=dict(
        type='RMT_S'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dist_params = dict(backend='nccl', port=29517)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=0.9, min_lr=0.0, by_epoch=False)

data = dict(samples_per_gpu=4, workers_per_gpu=16)
evaluation = dict(interval=4000, metric='mIoU')
