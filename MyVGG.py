from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model

class MyVGG:

    def __init__(self, kernel_size=3, strides=1, kernel_initializer='he_normal',
                 padding='same', input_shape=(256 * 3, 1), num_classes=5, classifier_activation='softmax'):
                 self.kernel_size = kernel_size
                 self.strides = strides
                 self.kernel_initializer = kernel_initializer
                 self.padding = padding
                 self.input_shape = input_shape
                 self.num_classes = num_classes
                 self.classifier_activation = classifier_activation

    def __call__(self):
        return self.load_model()

    def load_model(self):
        inputs = Input(shape=self.input_shape)
        x = Conv1D(64, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(inputs)
        x = Dropout(0.5)(x)
        x = Conv1D(64, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = MaxPooling1D()(x)
        x = Conv1D(128, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Dropout(0.5)(x)
        x = Conv1D(128, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = MaxPooling1D()(x)
        x = Conv1D(256, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Conv1D(256, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Conv1D(256, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = MaxPooling1D()(x)
        x = Conv1D(512, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Conv1D(512, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Conv1D(512, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = MaxPooling1D()(x)
        x = Conv1D(1024, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Conv1D(1024, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = Conv1D(1024, kernel_size= self.kernel_size, stride=self.strides, padding=self.padding, kernel_initializer=self.kernel_initializer)(x)
        x = GlobalAveragePooling1D()(x)
        y = Dense(self.num_classes, activation=self.classifier_activation)(x)

        model = Model(inputs=inputs, outputs=y)

        return model