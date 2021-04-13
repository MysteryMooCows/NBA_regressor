import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import log_loss, make_scorer


def read_data():
    games_df = pd.read_csv("data/games.csv")
    games_df = games_df.set_index("GAME_ID")
    games_df = games_df.loc[:, ["HOME_TEAM_ID", "VISITOR_TEAM_ID", "PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home", "PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away", "HOME_TEAM_WINS"]]

    teams_df = pd.read_csv("data/teams.csv")
    teams_df = teams_df.set_index("TEAM_ID")
    teams_df = teams_df.loc[:, ["ABBREVIATION", "NICKNAME", "CITY"]]

    players_df = pd.read_csv("data/players.csv")
    players_df = players_df.set_index("PLAYER_ID")
    players_df = players_df.loc[:, ["TEAM_ID", "SEASON"]]

    return games_df, teams_df, players_df


def get_xy_dfs(games_df, teams_df, players_df):
    x_df = games_df.loc[:, ["FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away"]]
    y_df = games_df.loc[:, ["HOME_TEAM_WINS"]]

    return x_df, y_df


def preprocess(x_df, y_df):
    keep_indices = x_df.loc[~x_df.isna().any(axis=1), :].index
    
    x_df = x_df.loc[keep_indices, :]
    y_df = y_df.loc[keep_indices, :]

    num_categories = y_df.value_counts().count()

    x_train_df = x_df.iloc[len(x_df)//10:len(x_df), :] # TODO: Random selection of examples for train and test set (randomize order of concat of x_df and y_df before k-fold)
    x_test_df = x_df.iloc[0:len(x_df)//10, :]           # TODO: k-fold cross-validation

    y_train_df = y_df.iloc[len(y_df)//10:len(y_df), :]
    y_test_df = y_df.iloc[0:len(y_df)//10, :]

    x_train = x_train_df.to_numpy()
    x_test = x_test_df.to_numpy()
    
    y_train = keras.utils.to_categorical(y_train_df, num_categories)
    y_test = keras.utils.to_categorical(y_test_df, num_categories)

    # Verify one-hot representation of 1 and 0
    '''
    for i in range(5):
        print(y_df.iloc[i, :])
        print(y_test[i])
    '''

    return x_train, y_train, x_test, y_test


def get_model(x_shape=(6,), num_categories=2):
    inputs = Input(shape=x_shape)
    hidden = Dense(64, activation="relu")(inputs)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(64, activation="relu")(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(32, activation="relu")(hidden)
    outputs = Dense(num_categories, activation="softmax")(hidden)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss=keras.losses.binary_crossentropy, metrics=["accuracy"], )

    return model


def loss(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)

    loss = log_loss(y, y_pred)

    return loss


def accuracy(y, y_pred):
    y_pred = np.round(y_pred)

    y_df = pd.DataFrame(y)
    y_pred_df = pd.DataFrame(y_pred)

    y_away_series = y_df.iloc[:, 0]
    y_pred_away_series = y_pred_df.iloc[:, 0]

    value_count_table = (y_away_series == y_pred_away_series).value_counts()

    print(value_count_table)
    if (True in value_count_table):
        accuracy = value_count_table[True] / y_away_series.size
    else:
        accuracy = 0

    return accuracy


class NeuralClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = get_model(x_shape=(6,), num_categories=2) # TODO: fix magic numbers
        self.scorer = make_scorer(accuracy, greater_is_better=True) # log_loss, greater_is_better=False

    def fit(self, xs, ys):
        self.model.fit(xs, ys, epochs=12, batch_size=64, verbose=1)
        return self

    def predict(self, test_input):
        return self.model.predict(test_input)


if __name__ == "__main__":
    games_df, teams_df, players_df = read_data()
    x_df, y_df = get_xy_dfs(games_df, teams_df, players_df)
    x_train, y_train, x_test, y_test = preprocess(x_df, y_df)

    model = get_model(x_shape=(6,), num_categories=2)
    model.fit(x_train, y_train, epochs=12, batch_size=64, validation_data=(x_test, y_test), verbose=0)

    print(model.evaluate(x_train, y_train))

    test_x = np.array([0.402, 0.826, 0.243, 0.388, 0.900, 0.333]).reshape(1, 6)
    test_prediction = model.predict(test_x) # should be 1

    print(f"test_x: {test_x}")
    print(f"p(away team wins) = {test_prediction[0][0]}")
    print(f"p(home team wins) = {test_prediction[0][1]}")

    print(loss(np.array([[0, 1]]), test_prediction))
    print(accuracy(np.array([[0, 1]]), test_prediction))
    print("Home team should be predicted to win")

    keep_indices = x_df.loc[~x_df.isna().any(axis=1), :].index

    x_df = x_df.loc[keep_indices, :]
    y_df = y_df.loc[keep_indices, :]

    x_array = np.array(x_df)
    y_array = keras.utils.to_categorical(y_df, 2)
    
    classifier = NeuralClassifier()
    k=10
    cvScores = model_selection.cross_val_score(classifier, x_array, y_array, cv=k, scoring=classifier.scorer)
    mean = np.mean(cvScores)
    print(f"Average accuracy for k={k} is {1 * mean}")
    


