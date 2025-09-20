"""
model_design.py — Advanced 3D model designs for OASIS-3 (TF2.10, Py3.9, CUDA 11.2)
- 3D ResNet-style classifier (Residual blocks)
- 3D Attention-Pooling head (SE-style channel attention + gated pooling)
- Multi-task net: shared encoder → (seg head, cls head)
All models are Keras models compatible with your OASIS-3 pipeline.
"""

import tensorflow as tf

def conv3d_bn_act(x, f, k=3, s=1, act='relu'):
    x = tf.keras.layers.Conv3D(f, k, strides=s, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if act:
        x = tf.keras.layers.Activation(act)(x)
    return x

def se_block(x, r=8):
    c = x.shape[-1]
    s = tf.keras.layers.GlobalAveragePooling3D()(x)
    s = tf.keras.layers.Dense(max(c//r, 4), activation='relu')(s)
    s = tf.keras.layers.Dense(c, activation='sigmoid')(s)
    s = tf.keras.layers.Reshape((1,1,1,c))(s)
    return tf.keras.layers.Multiply()([x, s])

def residual_block(x, f, stride=1, use_se=True):
    shortcut = x
    x = conv3d_bn_act(x, f, 3, stride)
    x = conv3d_bn_act(x, f, 3, 1, act=None)
    if shortcut.shape[-1] != f or stride != 1:
        shortcut = tf.keras.layers.Conv3D(f, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    if use_se:
        x = se_block(x)
    return x

def resnet3d_classifier(input_shape=(96,96,96,1), widths=(32,64,128,256), blocks=(2,2,2,2), dropout=0.4):
    i = tf.keras.Input(shape=input_shape)
    x = conv3d_bn_act(i, widths[0], 7, 2)
    x = tf.keras.layers.MaxPool3D(pool_size=3, strides=2, padding='same')(x)
    for wi, bi in zip(widths, blocks):
        for b in range(bi):
            stride = 2 if (b==0 and x.shape[-1] != wi) else 1
            x = residual_block(x, wi, stride=stride, use_se=True)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    o = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    m = tf.keras.Model(i, o, name='resnet3d_cls')
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['AUC','accuracy'])
    return m

def attention_pool_head(x, dropout=0.3):
    f = x.shape[-1]
    q = tf.keras.layers.GlobalAveragePooling3D()(x)
    q = tf.keras.layers.Dense(f, activation='relu')(q)
    a = tf.keras.layers.Conv3D(1, 1, padding='same')(x)
    a = tf.keras.layers.Activation('sigmoid')(a)
    z = tf.keras.layers.Multiply()([x, a])
    z = tf.keras.layers.GlobalAveragePooling3D()(z)
    z = tf.keras.layers.Concatenate()([z, q])
    z = tf.keras.layers.Dropout(dropout)(z)
    o = tf.keras.layers.Dense(1, activation='sigmoid')(z)
    return o

def resnet3d_with_attention(input_shape=(96,96,96,1), widths=(32,64,128), blocks=(2,2,2), dropout=0.3):
    i = tf.keras.Input(shape=input_shape)
    x = conv3d_bn_act(i, widths[0], 7, 2)
    x = tf.keras.layers.MaxPool3D(pool_size=3, strides=2, padding='same')(x)
    for wi, bi in zip(widths, blocks):
        for b in range(bi):
            stride = 2 if (b==0 and x.shape[-1] != wi) else 1
            x = residual_block(x, wi, stride=stride, use_se=True)
    o = attention_pool_head(x, dropout=dropout)
    m = tf.keras.Model(i, o, name='resnet3d_attn_cls')
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy',
              metrics=['AUC','accuracy'])
    return m

def multitask_seg_cls(input_shape_seg=(128,128,128,1), input_shape_cls=(96,96,96,1), base=16, dropout=0.3):
    i_seg = tf.keras.Input(shape=input_shape_seg, name='seg_input')
    c1 = conv3d_bn_act(i_seg, base); c1 = conv3d_bn_act(c1, base); p1 = tf.keras.layers.MaxPool3D()(c1)
    c2 = conv3d_bn_act(p1, base*2); c2 = conv3d_bn_act(c2, base*2); p2 = tf.keras.layers.MaxPool3D()(c2)
    c3 = conv3d_bn_act(p2, base*4); c3 = conv3d_bn_act(c3, base*4); p3 = tf.keras.layers.MaxPool3D()(c3)
    b  = conv3d_bn_act(p3, base*8); b  = conv3d_bn_act(b, base*8)

    u3 = tf.keras.layers.UpSampling3D()(b);  u3 = tf.keras.layers.Concatenate()([u3, c3])
    s4 = conv3d_bn_act(u3, base*4); s4 = conv3d_bn_act(s4, base*4)
    u2 = tf.keras.layers.UpSampling3D()(s4); u2 = tf.keras.layers.Concatenate()([u2, c2])
    s5 = conv3d_bn_act(u2, base*2); s5 = conv3d_bn_act(s5, base*2)
    u1 = tf.keras.layers.UpSampling3D()(s5); u1 = tf.keras.layers.Concatenate()([u1, c1])
    s6 = conv3d_bn_act(u1, base);  s6 = conv3d_bn_act(s6, base)
    seg_out  = tf.keras.layers.Conv3D(1, 1, activation='sigmoid', name='seg_out')(s6)

    i_cls = tf.keras.Input(shape=input_shape_cls, name='cls_input')
    x = conv3d_bn_act(i_cls, base); x = tf.keras.layers.MaxPool3D()(x)
    x = conv3d_bn_act(x, base*2); x = tf.keras.layers.MaxPool3D()(x)
    x = conv3d_bn_act(x, base*4); x = tf.keras.layers.GlobalAveragePooling3D()(x)
    b_flat = tf.keras.layers.GlobalAveragePooling3D()(b)
    h = tf.keras.layers.Concatenate()([x, b_flat])
    h = tf.keras.layers.Dropout(dropout)(h)
    cls_out = tf.keras.layers.Dense(1, activation='sigmoid', name='cls_out')(h)

    m = tf.keras.Model([i_seg, i_cls], [seg_out, cls_out], name='multitask_seg_cls')
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss={'seg_out':'binary_crossentropy','cls_out':'binary_crossentropy'},
              loss_weights={'seg_out':1.0,'cls_out':1.0},
              metrics={'cls_out':['AUC','accuracy']})
    return m
