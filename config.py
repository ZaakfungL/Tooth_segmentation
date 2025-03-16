# Parameters for training
train_params = {
    'image_dirs': 'data/2DTooth/images',
    'mask_dirs': 'data/2DTooth/masks',
    'checkpoint_dir': 'checkpoints',
    'model_name': 'unet',  # Choose from 'unet', 'unetplusplus', 'segformer'
    'num_epochs': 100,
    'batch_size': 8,
    'learning_rate': 0.001,
    'val_split': 0.2,
    'use_pretrained': False,  # Set this to False if you don't want to use pretrained weights
    'pretrained_file': 'checkpoints/segformer_dice_0.9325_iou_0.8748_hausdorff_7.8131.pth'  # Path to the weight file
}

# Parameters for testing
test_params = {
    'model_name': 'segformer',  # Choose from 'unet', 'unetplusplus', 'segformer'
    'weight_file': 'checkpoints/segformer_dice_0.9325_iou_0.8748_hausdorff_7.8131.pth',  # Path to the weight file
    'test_image_path': 'data/2DTooth/A-PXI/Unlabeled/Image/A_U_0002.png',  # Replace with your test image path
    'predict_image_path': 'output_image.jpg'  # Path to save the predicted output image
}

# 添加进化搜索相关参数
evolution_params = {
    'population_size': 20,
    'generations': 30,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8
}

# 搜索空间的配置
search_space_config = {
    # 编码器block的搜索空间
    'encoder_block': {
        'operations': [
            'conv3x3',
            'depthwise_separable_conv',
            'dilated_conv3x3',
            'inception_block'
        ],
        'num_ops_per_block': [2, 3],  # 每个block中的操作数量
        'channels': [32, 64, 128, 256],  # 每个stage的通道数
        'activation': ['relu', 'leaky_relu', 'silu'],
        'normalization': ['batch', 'instance', 'group'],
        'use_attention': [True, False],
        'skip_connection': ['residual', 'dense', 'none']
    },

    # 解码器block的搜索空间
    'decoder_block': {
        'operations': [
            'conv3x3',
            'transpose_conv',
            'upsample_conv'
        ],
        'num_ops_per_block': [2, 3],
        'activation': ['relu', 'leaky_relu', 'silu'],
        'normalization': ['batch', 'instance', 'group'],
        'use_attention': [True, False],
        'skip_fusion': ['add', 'concat']
    },

    # UNet整体结构配置
    'network': {
        'num_stages': 4,  # UNet的深度
        'initial_channels': [32, 64],  # 初始通道数
        'channel_multiplier': [2],  # 每个stage通道数的倍数
    }
}

# 零成本评估参数
zero_cost_params = {
    'ntk_batch_size': 8,
    'linear_regions_samples': 100,
    'score_threshold': 0.1,
    'ntk_weight': 0.3,
    'linear_regions_weight': 0.7
}

