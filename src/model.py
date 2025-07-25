import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

def conv_block(x, filters):
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = MaxPooling2D((2, 2))(f)
    return f, p

def decoder_block(x, skip, filters):
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
    x = concatenate([x, skip])
    x = conv_block(x, filters)
    return x

def build_model(input_shape=(256, 256, 3)):
    input_a = Input(input_shape)
    input_b = Input(input_shape)

    def shared_encoder(x):
        f1, p1 = encoder_block(x, 64)
        f2, p2 = encoder_block(p1, 128)
        f3, p3 = encoder_block(p2, 256)
        f4, p4 = encoder_block(p3, 512)
        bottleneck = conv_block(p4, 1024)
        return [f1, f2, f3, f4, bottleneck]

    feats_a = shared_encoder(input_a)
    feats_b = shared_encoder(input_b)

    diff = tf.keras.layers.Subtract()([feats_a[-1], feats_b[-1]])

    x = decoder_block(diff, feats_a[3], 512)
    x = decoder_block(x, feats_a[2], 256)
    x = decoder_block(x, feats_a[1], 128)
    x = decoder_block(x, feats_a[0], 64)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=[input_a, input_b], outputs=outputs)
    return model
