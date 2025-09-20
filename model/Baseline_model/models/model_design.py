import tensorflow as tf
import numpy as np

def conv3d_bn_act(x, f, k=3, s=1, act='relu', dropout=0.0):
    """Enhanced conv block with dropout"""
    x = tf.keras.layers.Conv3D(f, k, strides=s, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    if act:
        x = tf.keras.layers.Activation(act)(x)
    return x

def residual_conv_block(x, f, k=3, s=1, dropout=0.1):
    """Residual convolution block inspired by V-Net"""
    shortcut = x
    x = conv3d_bn_act(x, f, k, s, dropout=dropout)
    x = conv3d_bn_act(x, f, k, 1, act=None, dropout=dropout)
    
    # 1x1 conv for shortcut if channel or spatial dimensions don't match
    if shortcut.shape[-1] != f or s != 1:
        shortcut = tf.keras.layers.Conv3D(f, 1, strides=s, padding='same', use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def se_block(x, r=16):
    """Squeeze-and-Excitation block for channel-wise attention"""
    c = x.shape[-1]
    s = tf.keras.layers.GlobalAveragePooling3D()(x)
    s = tf.keras.layers.Dense(max(c//r, 4), activation='relu')(s)
    s = tf.keras.layers.Dense(c, activation='sigmoid')(s)
    s = tf.keras.layers.Reshape((1,1,1,c))(s)
    return tf.keras.layers.Multiply()([x, s])

def vnet_encoder_block(x, f, depth=2):
    """V-Net style encoder block with residual connections and SE"""
    for i in range(depth):
        x = residual_conv_block(x, f)
    x = se_block(x)
    return x

def vnet_decoder_block(x, skip, f, depth=2):
    """V-Net style decoder block with skip connections"""
    x = tf.keras.layers.UpSampling3D()(x)
    x = tf.keras.layers.Concatenate()([x, skip])
    for i in range(depth):
        x = residual_conv_block(x, f)
    return x

def improved_vnet_segmentation(input_shape=(128,128,128,1), base=16, depth=2):
    """
    Improved V-Net for hippocampus segmentation.
    This model uses residual blocks and SE attention.
    """
    i = tf.keras.Input(shape=input_shape)
    
    # --- Encoder path ---
    e1 = vnet_encoder_block(i, base, depth)
    p1 = tf.keras.layers.MaxPool3D()(e1)
    
    e2 = vnet_encoder_block(p1, base*2, depth)
    p2 = tf.keras.layers.MaxPool3D()(e2)
    
    e3 = vnet_encoder_block(p2, base*4, depth)
    p3 = tf.keras.layers.MaxPool3D()(e3)
    
    # --- Bottleneck ---
    b = vnet_encoder_block(p3, base*8, depth)
    
    # --- Decoder path ---
    d3 = vnet_decoder_block(b, e3, base*4, depth)
    d2 = vnet_decoder_block(d3, e2, base*2, depth)
    d1 = vnet_decoder_block(d2, e1, base, depth)
    
    # --- Output layer ---
    output = tf.keras.layers.Conv3D(1, 1, activation='sigmoid', name='seg_output')(d1)
    
    model = tf.keras.Model(i, output, name='improved_vnet_seg')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=combined_dice_bce_loss, # Custom loss
        metrics=[dice_coefficient, sensitivity, specificity] # Custom metrics
    )
    return model

def improved_resnet3d_classifier(input_shape=(96,96,96,1), widths=(32,64,128,256), 
                                blocks=(2,2,2,2), dropout=0.4):
    """
    Enhanced ResNet3D with SE blocks and uncertainty estimation.
    Takes a 3D patch (e.g., hippocampus) and outputs classification + uncertainty.
    """
    i = tf.keras.Input(shape=input_shape)
    
    # Initial stem
    x = conv3d_bn_act(i, widths[0], 7, 2) # Downsample 1
    x = tf.keras.layers.MaxPool3D(pool_size=3, strides=2, padding='same')(x) # Downsample 2
    
    # Residual stages
    for wi, bi in zip(widths, blocks):
        for b in range(bi):
            # Downsample at the start of each stage (except the first)
            stride = 2 if (b==0 and wi != widths[0]) else 1 
            x = residual_conv_block(x, wi, s=stride, dropout=dropout*0.5)
        x = se_block(x) # Apply attention at the end of each stage
        
    # --- Classification Head ---
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    # Dual output: prediction and uncertainty
    main_output = tf.keras.layers.Dense(1, activation='sigmoid', name='main_pred')(x)
    # 'softplus' ensures uncertainty is always positive
    uncertainty_output = tf.keras.layers.Dense(1, activation='softplus', name='uncertainty')(x) 
    
    model = tf.keras.Model(i, [main_output, uncertainty_output], name='improved_resnet3d_cls')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss={'main_pred': 'binary_crossentropy', 'uncertainty': 'mse'}, # Multi-task loss
        loss_weights={'main_pred': 1.0, 'uncertainty': 0.2}, # Weight main task more
        metrics={'main_pred': ['AUC', 'accuracy', sensitivity, specificity]}
    )
    return model

# --- Custom metrics and losses ---

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for segmentation evaluation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss (1 - Dice coefficient)"""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_dice_bce_loss(y_true, y_pred, alpha=0.8): 
    """
    Combined loss: 80% Dice Loss + 20% Binary Cross-Entropy
    Good for balancing global (BCE) and local (Dice) segmentation performance.
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return alpha * dice + (1 - alpha) * bce

def sensitivity(y_true, y_pred):
    """Sensitivity (True Positive Rate or Recall)"""
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def specificity(y_true, y_pred):
    """Specificity (True Negative Rate)"""
    true_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())