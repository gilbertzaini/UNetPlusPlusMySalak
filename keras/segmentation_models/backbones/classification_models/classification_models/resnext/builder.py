import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
# from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.keras.utils import get_source_inputs

def get_batch_normalization():
    if tf.executing_eagerly():
        return tf.keras.layers.BatchNormalization
    else:
        # If using v1 behavior or manually configured TensorFlow 1.x behavior
        return tf.compat.v1.keras.layers.BatchNormalization

# Use `get_batch_normalization` to get the appropriate BatchNormalization layer
BatchNormalization = get_batch_normalization()

# deprecated function, so pasted the function manually
def _obtain_input_shape(input_shape,
                       default_size,
                       min_size,
                       data_format=None,
                       require_flatten=True,
                       weights=None):
    """Determines the proper input shape for the model.
    
    Arguments:
        input_shape: Optional shape tuple, only to be specified if `include_top` is False.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use (default is None to use `K.image_data_format()`).
        require_flatten: Whether the model is expected to be flattened.
        weights: Pretrained weights (None indicates random initialization).
    
    Returns:
        An input shape tuple.
    
    Raises:
        ValueError: In case of invalid argument values.
    """
    
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] is not None and input_shape[0] < min_size:
                raise ValueError(f'Input size must be at least {min_size}x{min_size}; got {input_shape}')
        else:
            if input_shape[-1] is not None and input_shape[-1] < min_size:
                raise ValueError(f'Input size must be at least {min_size}x{min_size}; got {input_shape}')
        return input_shape
    
    if data_format is None:
        data_format = K.image_data_format()
    
    if data_format == 'channels_first':
        default_shape = (3, default_size, default_size)
    else:
        default_shape = (default_size, default_size, 3)
    
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None and input_shape != default_shape:
            raise ValueError(f'Invalid input shape {input_shape} for weights "imagenet". '
                             f'Expected shape {default_shape}.')
        return default_shape
    else:
        if input_shape:
            if data_format == 'channels_first':
                if input_shape[0] != 3:
                    raise ValueError(f'Invalid input shape {input_shape} for data format "channels_first".')
            else:
                if input_shape[-1] != 3:
                    raise ValueError(f'Invalid input shape {input_shape} for data format "channels_last".')
            return input_shape
        else:
            return default_shape

from .params import get_conv_params
from .params import get_bn_params

from .blocks import conv_block
from .blocks import identity_block


def build_resnext(
     repetitions=(2, 2, 2, 2),
     include_top=True,
     input_tensor=None,
     input_shape=None,
     classes=1000,
     first_conv_filters=64,
     first_block_filters=64):
    
    """
    TODO
    """
    
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format='channels_last',
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    # get parameters for model layers
    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = first_block_filters
    
    # resnext bottom
    x = BatchNormalization(name='bn_data', **no_scale_bn_params)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(first_conv_filters, (7, 7), strides=(2, 2), name='conv0', **conv_params)(x)
    x = BatchNormalization(name='bn0', **bn_params)(x)
    x = Activation('relu', name='relu0')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)
    
    # resnext body
    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            
            filters = init_filters * (2**stage)
            
            # first block of first stage without strides because we have maxpooling before
            if stage == 0 and block == 0:
                x = conv_block(filters, stage, block, strides=(1, 1))(x)
                
            elif block == 0:
                x = conv_block(filters, stage, block, strides=(2, 2))(x)
                
            else:
                x = identity_block(filters, stage, block)(x)

    # resnext top
    if include_top:
        x = GlobalAveragePooling2D(name='pool1')(x)
        x = Dense(classes, name='fc1')(x)
        x = Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    # Create model
    model = Model(inputs, x)

    return model
