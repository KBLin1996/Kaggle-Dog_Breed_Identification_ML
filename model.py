import tensorflow as tf

def VGG16(x, is_training, n_classes):
    with tf.name_scope('Block1'):
        conv1_1 = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv1_1'
        )
        conv1_2 = tf.layers.conv2d(
            inputs=conv1_1,
            filters=64,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv1_2'
        )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1_2,
            pool_size=2,
            strides=(2, 2),
            padding='valid',
            name='pool1'
        )

    with tf.name_scope('Block2'):
        conv2_1 = tf.layers.conv2d(
            inputs=pool1,
            filters=128,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv2_1'
        )
        conv2_2 = tf.layers.conv2d(
            inputs=conv2_1,
            filters=128,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv2_2'
        )
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2_2,
            pool_size=2,
            strides=(2, 2),
            padding='valid',
            name='pool2'
        )

    with tf.name_scope('Block3'):
        conv3_1 = tf.layers.conv2d(
            inputs=pool2,
            filters=256,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv3_1'
        )
        conv3_2 = tf.layers.conv2d(
            inputs=conv3_1,
            filters=256,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv3_2'
        )
        conv3_3 = tf.layers.conv2d(
            inputs=conv3_2,
            filters=256,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv3_3'
        )
        pool3 = tf.layers.max_pooling2d(
            inputs=conv3_3,
            pool_size=2,
            strides=(2, 2),
            padding='valid',
            name='pool3'
        )

    with tf.name_scope('Block4'):
        conv4_1 = tf.layers.conv2d(
            inputs=pool3,
            filters=512,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv4_1'
        )
        conv4_2 = tf.layers.conv2d(
            inputs=conv4_1,
            filters=512,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv4_2'
        )
        conv4_3 = tf.layers.conv2d(
            inputs=conv4_2,
            filters=512,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv4_3'
        )
        pool4 = tf.layers.max_pooling2d(
            inputs=conv4_3,
            pool_size=2,
            strides=(2, 2),
            padding='valid',
            name='pool4'
        )

    with tf.name_scope('Block5'):
        conv5_1 = tf.layers.conv2d(
            inputs=pool4,
            filters=512,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv5_1'
        )
        conv5_2 = tf.layers.conv2d(
            inputs=conv5_1,
            filters=512,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv5_2'
        )
        conv5_3 = tf.layers.conv2d(
            inputs=conv5_2,
            filters=512,
            kernel_size=3,
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            name='conv5_3'
        )
        pool5 = tf.layers.max_pooling2d(
            inputs=conv5_3,
            pool_size=2,
            strides=(2, 2),
            padding='valid',
            name='pool5'
        )

    with tf.name_scope('FullyConnected1'):
        flat = tf.layers.flatten(inputs=pool5, name='flat')
        fc1 = tf.layers.dense(
            inputs=flat,
            units=512,
            name='fc1'
        )
        drop1 = tf.layers.dropout(
            inputs=fc1,
            rate=0.5,
            training=is_training,
            name='drop1'
        )

    with tf.name_scope('FullyConnected2'):
        fc2 = tf.layers.dense(
            inputs=drop1,
            units=512,
            name='fc2'
        )
        drop2 = tf.layers.dropout(
            inputs=fc2,
            rate=0.5,
            training=is_training,
            name='drop2'
        )

    with tf.name_scope('classification'):
        classifier = tf.layers.dense(
            inputs=drop2,
            units=n_classes,
            name='class'
        )

    return classifier

if __name__ == "__main__":
    main()
