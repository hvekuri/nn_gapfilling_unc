import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from ngboost.distns.laplace import Laplace
from tensorflow.keras.initializers import (
    HeNormal,
    GlorotUniform,
)
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential


def laplace_nll(y_true, y_pred):
    """
    Laplace negative log likelihood.

    NOTE: Assumes that model has two outputs, with the first output being
    the location (mu) and the second output being the log scale (log_b).
    """

    # Split the output into mu and log_b
    mu, log_b = tf.split(y_pred, num_or_size_splits=2, axis=-1)
    b = tf.exp(log_b)
    log_likelihood = -tf.math.log(2.0 * b) - tf.abs(y_true - mu) / b
    return -tf.reduce_mean(log_likelihood) + 5


def generate_model(in_d, lr, l2_reg):
    """
    Generates a neural network model for gap filling.

    Args:
        in_d (int): The input dimension of the model.
        lr (float): The learning rate for the optimizer.
        l2_reg (float): The L2 regularization parameter.
        network_struct (list): A list specifying the number of neurons in each hidden layer.
        seed (int): The seed value for random initialization.

    Returns:
        tf.keras.Model: The compiled neural network model.
    """
    model = Sequential()
    model.add(Flatten())
    model.add(
        Dense(
            50,
            activation="relu",
            kernel_regularizer=l2(l2_reg),
            kernel_initializer=HeNormal(),
            input_dim=in_d,
        )
    )
    model.add(
        Dense(
            50,
            activation="relu",
            kernel_regularizer=l2(l2_reg),
            kernel_initializer=HeNormal(),
        )
    )
    model.add(Dense(2, activation="linear", kernel_initializer=GlorotUniform()))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=[laplace_nll]
    )
    return model


def fit_model(X_train, y_train, X_val, y_val, learning_rate, bsize, l2_reg):
    """Fits a neural network

    Args:
        X_train (np.array): Scaled independent variables, training data
        y_train (np.array): Scaled dependent variable, training data, 1D
        X_val (np.array): Scaled independent variables, validation data
        y_val (np.array): Scaled dependent variable, validation data, 1D
        learning_rate (float): Learning rate
        bsize (int): Batch size
        l2_reg (float): L2 regularization factor

    Returns:
        tf.keras.Model
    """
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        verbose=1,
        restore_best_weights=False,
    )

    model = generate_model(X_train.shape[1], learning_rate, l2_reg)

    history = model.fit(
        X_train,
        y_train,
        epochs=500,
        batch_size=bsize,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=False,
    )
    return model


def get_train_val(all_train_data, scaler_x, scaler_y, x_cols, y_col):
    """Splits data into training and validation sets

    Args:
        all_train_data (pd.DataFrame): All training data (no missing values in y_col)
        scaler_x (StandardScaler): Fitted standard scaler for X
        scaler_y (StandardScaler): Fitted standard scaler for y
        x_cols (list): Driver names
        y_col (str): Name of dependent variable

    Returns:
        tuple: Scaled training and validation sets
    """
    train_data, val_data = train_test_split(all_train_data, test_size=0.1)
    X_train = scaler_x.transform(np.asarray(train_data[x_cols]))
    y_train = scaler_y.transform(np.asarray(train_data[y_col]).reshape(-1, 1))
    X_val = scaler_x.transform(np.asarray(val_data[x_cols]))
    y_val = scaler_y.transform(np.asarray(val_data[y_col]).reshape(-1, 1))
    return X_train, y_train, X_val, y_val


def gapfill_NN(orig_data, x_cols, y_col, learning_rate, bsize, l2_reg, N):
    """Gap-fills y_col using a deep ensemble

    Args:
        orig_data (pd.DataFrame): Original data
        x_cols (list): Driver names
        y_col (str): Name of dependent variable
        learning_rate (float): Learning rate
        bsize (int): Batch size
        l2_reg (float): L2 refularization factor
        N (int): Number of models in the ensemble

    Returns:
        pd.DataFrame: Original and gap-filled data including uncertainty estimates
    """

    all_train_data = orig_data[(orig_data[y_col].notnull())].copy()

    scaler_x = StandardScaler().fit(np.asarray(orig_data[x_cols]))
    scaler_y = StandardScaler().fit(np.asarray(all_train_data[y_col]).reshape(-1, 1))

    X_all = scaler_x.transform(np.asarray(orig_data[x_cols]))

    scaled_data = [
        get_train_val(all_train_data.copy(), scaler_x, scaler_y, x_cols, y_col)
        for _ in range(N)
    ]

    trained_models = [
        fit_model(
            scaled_data[i][0],
            scaled_data[i][1],
            scaled_data[i][2],
            scaled_data[i][3],
            learning_rate,
            bsize,
            l2_reg,
        )
        for i in range(N)
    ]

    emods = []
    means = []

    for i in range(N):
        pred_all_y = trained_models[i].predict(X_all)
        cur_mean = scaler_y.inverse_transform(pred_all_y[:, 0].reshape(-1, 1))
        means.append(cur_mean)
        cur_log_scale = pred_all_y[:, 1]
        ensemble_dist = Laplace(
            np.array([pred_all_y[:, 0], cur_log_scale])
        )  # Parameters are mean and log scale
        y_hat_uncertainty_95 = np.array(
            [dist.dist.interval(0.95) for dist in ensemble_dist]
        )
        ub = scaler_y.inverse_transform(y_hat_uncertainty_95[:, 1].reshape(-1, 1))
        emod = ub - cur_mean
        emods.append(emod)
        orig_data[f"modelled_{y_col}{i}"] = cur_mean
    aleatoric = np.mean(emods, axis=0)
    epistemic = np.std(means, axis=0) * 1.96
    orig_data["Emod"] = np.sqrt(epistemic**2 + aleatoric**2)
    orig_data["Aleatoric"] = aleatoric
    orig_data["Epistemic"] = epistemic
    orig_data[f"modelled_{y_col}"] = np.mean(
        orig_data.filter(regex=f"modelled_{y_col}\d"), axis=1
    )
    orig_data[f"gapfilled_{y_col}"] = orig_data[y_col].copy()

    orig_data.loc[orig_data[y_col].isnull(), f"gapfilled_{y_col}"] = orig_data[
        f"modelled_{y_col}"
    ]

    return orig_data
