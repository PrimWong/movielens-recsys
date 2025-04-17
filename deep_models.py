import tensorflow as tf
from tensorflow.keras import layers, Model

def build_ncf_model(num_users, num_items,
                    embed_size=50,
                    dense_layers=[64, 32, 16, 8]):
    """
    Neural Collaborative Filtering (NCF) with MLP on concatenated embeddings.
    """
    # Inputs
    user_input = layers.Input(shape=(1,), name='user_id')
    item_input = layers.Input(shape=(1,), name='item_id')

    # Embeddings
    user_embedding = layers.Embedding(
        input_dim=num_users + 1, output_dim=embed_size,
        name='user_embedding')(user_input)
    item_embedding = layers.Embedding(
        input_dim=num_items + 1, output_dim=embed_size,
        name='item_embedding')(item_input)

    # Flatten
    user_vec = layers.Flatten()(user_embedding)
    item_vec = layers.Flatten()(item_embedding)

    # MLP
    x = layers.Concatenate()([user_vec, item_vec])
    for units in dense_layers:
        x = layers.Dense(units, activation='relu')(x)

    # Final rating prediction
    output = layers.Dense(1, name='rating')(x)

    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model


def train_ncf(model, df, epochs=5, batch_size=256, validation_split=0.1):
    """
    Train NCF model on a DataFrame with columns ['userId','movieId','rating'].
    """
    users = df.userId.values
    items = df.movieId.values
    ratings = df.rating.values.astype('float32')

    history = model.fit(
        x=[users, items],
        y=ratings,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1
    )
    return history
