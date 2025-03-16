kernel_initializer =  'he_uniform'

def nestedunet_convglu(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    layer1 = Lambda(lambda x: x / 255)(inputs)
    print(layer1.shape)
    # x1_0    1st layer
    layer1 = Conv3D(16, (3, 3, 3),activation='relu', kernel_regularizer=regularizers.L2(0.01), kernel_initializer=kernel_initializer, padding='same')(layer1)
    print(layer1.shape)
    out  = MaxPooling3D((2, 2, 2))(layer1)
    print(out.shape)
    branch1,branch2 = tf.split(out, num_or_size_splits=2, axis=4)

    #x2_0      64
    branch1 = Conv3D(32, (3, 3, 3),kernel_regularizer=regularizers.L2(0.01), kernel_initializer=kernel_initializer, padding='same')(branch1)
    branch2 = Conv3D(32, (3, 3, 3),kernel_regularizer=regularizers.L2(0.01), kernel_initializer=kernel_initializer, padding='same')(branch2)
    branch1 = Activation('relu')(branch1)
    branch2 = Activation('sigmoid')(branch2)
    layer2 = branch1 * branch2
    branch1,branch2 = tf.split(MaxPooling3D((2, 2, 2))(layer2), num_or_size_splits=2, axis=4)

    # x1_1         d
    layer1 = concatenate([layer1,Conv3DTranspose(16, (2, 2, 2), kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer2)])
  
    #x3_0         d
    branch1 = Conv3D(64, (3, 3, 3),kernel_regularizer=regularizers.L2(0.01), kernel_initializer=kernel_initializer, padding='same')(branch1)
    branch2 = Conv3D(64, (3, 3, 3),kernel_regularizer=regularizers.L2(0.01), kernel_initializer=kernel_initializer, padding='same')(branch2)
    branch1 = Activation('relu')(branch1)
    branch2 = Activation('sigmoid')(branch2)
    layer3 = branch1 * branch2
    branch1,branch2 = tf.split(MaxPooling3D((2, 2, 2))(layer3), num_or_size_splits=2, axis=4)

    #x2_1        d
    layer2 = concatenate([layer2,Conv3DTranspose(32, (2, 2, 2) , kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer3)])

    #x1_2
    layer1 = concatenate([layer1,Conv3DTranspose(16, (2, 2, 2) , kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer2)])

    #x4_0
    branch1 = Conv3D(128, (3, 3, 3),kernel_regularizer=regularizers.L2(0.01), kernel_initializer=kernel_initializer, padding='same')(branch1)
    branch2 = Conv3D(128, (3, 3, 3),kernel_regularizer=regularizers.L2(0.01), kernel_initializer=kernel_initializer, padding='same')(branch2)
    branch1 = Activation('relu')(branch1)
    branch2 = Activation('sigmoid')(branch2)
    layer4 = branch1 * branch2
    branch1,branch2 = tf.split(MaxPooling3D((2, 2, 2))(layer4), num_or_size_splits=2, axis=4)

    #x3_1
    layer3 = concatenate([layer3,Conv3DTranspose(64, (2, 2, 2) , kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer4)])

    #2_2
    layer2 = concatenate([layer2,Conv3DTranspose(32, (2, 2, 2) , kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer3)])

    #x1_3
    layer1 = concatenate([layer1,Conv3DTranspose(16, (2, 2, 2) , kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer2)])

    #x5_0
    branch1 = Conv3D(256, (3, 3, 3), kernel_regularizer=regularizers.L2(0.01),kernel_initializer=kernel_initializer, padding='same')(branch1)
    branch2 = Conv3D(256, (3, 3, 3),kernel_regularizer=regularizers.L2(0.01), kernel_initializer=kernel_initializer, padding='same')(branch2)
    branch1 = Activation('relu')(branch1)
    branch2 = Activation('sigmoid')(branch2)
    layer5 = branch1 * branch2
    branch1,branch2 = tf.split(MaxPooling3D((2, 2, 2))(layer5), num_or_size_splits=2, axis=4)

    #x4_1
    layer4 = concatenate([layer4,Conv3DTranspose(128, (2, 2, 2) , kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer5)])

    #x3_2
    layer3 = concatenate([layer3,Conv3DTranspose(64, (2, 2, 2), kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer4)])

    #x2_3
    layer2 = concatenate([layer2,Conv3DTranspose(32, (2, 2, 2), kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer3)])


    #x1_4
    layer1 = concatenate([layer1,Conv3DTranspose(16, (2, 2, 2), kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer2)])
    
    #x6_0
    branch1 = Conv3D(512, (3, 3, 3), kernel_regularizer=regularizers.L2(0.01),kernel_initializer=kernel_initializer, padding='same')(branch1)
    branch2 = Conv3D(512, (3, 3, 3),kernel_regularizer=regularizers.L2(0.01), kernel_initializer=kernel_initializer, padding='same')(branch2)
    branch1 = Activation('relu')(branch1)
    branch2 = Activation('sigmoid')(branch2)
    layer6 = branch1 * branch2
    
    #x5_1
    layer5 = concatenate([layer5,Conv3DTranspose(256, (2, 2, 2), kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer6)])

    #x4_2
    layer4 = concatenate([layer4,Conv3DTranspose(128, (2, 2, 2), kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer5)])

    #x3_3
    layer3 = concatenate([layer3,Conv3DTranspose(64, (2, 2, 2), kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer4)])

    #x2_4
    layer2 = concatenate([layer2,Conv3DTranspose(32, (2, 2, 2), kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer3)])

    #1_5
    layer1 = concatenate([layer1,Conv3DTranspose(16, (2, 2, 2), kernel_regularizer=regularizers.L2(0.01), strides=(2, 2, 2), padding='same')(layer2)])

    #final layer
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(layer1)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=CategoricalCrossentropy(),
              metrics=[Accuracy(name="accuracy"), MeanIoU(num_classes=4,name="iou"),AUC(name="auc"),Recall(name="recall"),Precision(name="precision")])
    model.summary()

    return model