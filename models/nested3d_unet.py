kernel_initializer =  'he_uniform'

def simple_3dnestedunet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    # x0_0    1st layer
    c1 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    print("c1",c1.shape,"p1",p1.shape)

    #x1_0      64
    c2 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)
    print("c2",c2.shape,"p2",p2.shape)


    # x0_1         d
    u1 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c2)
    u1 = concatenate([c1,u1])
    u1 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u1)
    u1 = BatchNormalization()(u1)
    u1 = Dropout(0.2)(u1)
    u1 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u1)
    u1 = BatchNormalization()(u1)
    print("u1",u1.shape)

    #x2_0         d
    c3 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.1)(c3)
    c3 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    #x1_1        d
    u2 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c3)
    u2 = concatenate([c2,u2])
    u2 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u2)
    u2 = BatchNormalization()(u2)
    u2 = Dropout(0.2)(u2)
    u2 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u2)
    u2 = BatchNormalization()(u2)

    #x0_2
    u3 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(u2)
    u3 = concatenate([c1,u1,u3])
    u3 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u3)
    u3 = BatchNormalization()(u3)
    u3 = Dropout(0.2)(u3)
    u3 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u3)
    u3 = BatchNormalization()(u3)

    #x3_0
    c4 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.1)(c4)
    c4 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling3D((2, 2, 2))(c4)

    #x2_1
    u4 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c4)
    u4 = concatenate([c3,u4])
    u4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u4)
    u4 = BatchNormalization()(u4)
    u4 = Dropout(0.2)(u4)
    u4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u4)
    u4 = BatchNormalization()(u4)

    #1_2
    u5 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(u4)
    u5 = concatenate([c2,u2,u5])
    u5 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u5)
    u5 = BatchNormalization()(u5)
    u5 = Dropout(0.2)(u5)
    u5 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u5)
    u5 = BatchNormalization()(u5)

    #x0_3
    u6 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(u5)
    u6 = concatenate([c1,u1,u3,u6])
    u6 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    u6 = BatchNormalization()(u6)
    u6 = Dropout(0.2)(u6)
    u6 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    u6 = BatchNormalization()(u6)


    #x4_0
    c5 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.1)(c5)
    c5 = Conv3D(512, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)
    c5 = BatchNormalization()(c5)

    #x3_1
    u7 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u7 = concatenate([c4,u7])
    u7 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    u7 = BatchNormalization()(u7)
    u7 = Dropout(0.2)(u7)
    u7 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    u7 = BatchNormalization()(u7)

    #x2_2
    u8 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(u7)
    u8 = concatenate([c3,u4,u8])
    u8 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    u8 = BatchNormalization()(u8)
    u8 = Dropout(0.2)(u8)
    u8 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    u8 = BatchNormalization()(u8)

    #x1_3
    u9 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(u8)
    u9 = concatenate([c2,u2,u5,u9])
    u9 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    u9 = BatchNormalization()(u9)
    u9 = Dropout(0.2)(u9)
    u9 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    u9 = BatchNormalization()(u9)

    #x0_4
    u10 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(u9)
    u10 = concatenate([c1,u1,u3,u6,u10])
    u10 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u10)
    u10 = BatchNormalization()(u10)
    u10 = Dropout(0.2)(u10)
    u10 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u10)
    u10 = BatchNormalization()(u10)

    #final layer
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(u10)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=CategoricalCrossentropy(),
              metrics=[Accuracy(name="accuracy"), MeanIoU(num_classes=4,name="iou"),AUC(name="auc"),Recall(name="recall"),Precision(name="precision")])
    model.summary()

    return model
