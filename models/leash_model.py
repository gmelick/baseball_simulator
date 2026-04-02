import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import pandas as pd

def create_models(game_date):
    full_df = pd.DataFrame()
    for season in range(game_date.year - 3, game_date.year + 1):
        if not os.path.exists(f'{season} Pitcher Hooks.csv'):
            continue
        season_df = pd.read_csv(f'{season} Pitcher Hooks.csv', parse_dates=['Date'])
        full_df = pd.concat((full_df, season_df[season_df.Date < pd.to_datetime(game_date)]))

    models = []
    for starter_reliever in range(2):
        df = full_df[full_df.Starter == 1 - starter_reliever]
        x = df.iloc[:, 2:-2].values
        x = x.reshape(x.shape[0], x.shape[1], 1)
        y = df.iloc[:, -1].values
        y = y.reshape(y.shape[0], 1)

        input_layer = keras.Input((x.shape[1], 1))
        conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(1, activation='sigmoid')(gap)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            optimizer="adam",
            loss='binary_crossentropy',
            metrics=[keras.metrics.PrecisionAtRecall(.5)],
        )

        epochs = 20
        batch_size = 256

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                f"best_model_{starter_reliever}.keras", save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=0),
        ]
        model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.2,
            verbose=2,
        )
        models.append(keras.models.load_model(f'best_model_{starter_reliever}.keras'))
    return models
