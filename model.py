import numpy as np
import pandas as pd
from keras.layers import Conv1D, BatchNormalization, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

class SonarModel:
    def __init__(self):
        self.data_path = 'data/'
        self.model_path = '{}models/'.format(self.data_path)
        self.batch_size = 32

    def create_data_df(self, data, header=True):
        """
        Creates a pandas data frame from a csv file.

        Args:
            header = If True then the csv file has header rows, if False then no
                header rows

        Returns:
            A pandas dataframe.
        """
        if header:
            sonar = pd.read_csv(data)
        else:
            sonar = pd.read_csv(data, header=None)

        return sonar

    def create_model(self):
        """
        Create the model to train.

        Returns:
            The model.
        """
        model = Sequential()
        model.add(Conv1D(32, 4, input_shape=(60,1), activation='relu', name='conv1'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(30, activation='relu', name='fc1'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid', name='prediction'))
        
        return model

    def train_model(self):
        """
        Train the model and save weights.
        """
        sonar_df = self.create_data_df('{}sonar.all-data'.format(self.data_path), header=False)
        trn, val = self.train_test_split(sonar_df)

        # The last column of the sonar data is the label specifiying if the data is for a rock "R"
        # or a mine "M". This splits the data into a dataframe for the data and one for the labels
        # for both the training and validation data.
        trn_data = trn.iloc[:, :-1]
        trn_label = trn.iloc[:, -1]
        val_data = val.iloc[:, :-1]
        val_label = val.iloc[:, -1]

        # expand_dims makes the data useable as input for the Conv1D layer of the model.
        trn_data = np.expand_dims(trn_data.values.astype(float), axis=2)
        val_data = np.expand_dims(val_data.values.astype(float), axis=2)

        # Endcode the labels as 0 and 1 instead of R and M.
        encoder = LabelEncoder()
        encoder.fit(val_label)
        trn_label = encoder.transform(trn_label)
        val_label = encoder.transform(val_label)

        lr = 1e-4
        epochs = 500

        model = self.create_model()
        model.compile(Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(trn_data,
                  trn_label,
                  epochs=epochs,
                  batch_size = self.batch_size,
                  validation_data=(val_data, val_label))
        model.save_weights('{}ft1.h5'.format(self.model_path))

    def train_test_split(self, data):
        """
        Randomly splits the data into a training and validation set with 80%
        being used for training and 20% for validation.

        Args:
            data = The dataset to split into training and validaiton sets.

        Returns:
            trn = The training dataset.
            val = The validation dataset.
        """
        np.random_seed = 42
        msk = np.random.rand(len(data)) < 0.8
        trn = data[msk]
        val = data[~msk]
        return trn, val
