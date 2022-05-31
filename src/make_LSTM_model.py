from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input,LSTM,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Normalization


# define and fit LSTM model - 1 layer w/ input Normalize layer
def fit_LSTM_type2(X_train,y_train,X_test,y_test,n_lag,n_features,n_neurons,batch_size,n_epochs,learning_rate,verbose,callbacks):
    normalizer = Normalization(name='normalize_layer')
    normalizer.adapt(X_train)
    
    inputs = Input(shape=(n_lag, n_features),name='input_layer')
    x = normalizer(inputs)
    x = LSTM(n_neurons,activation='tanh', return_sequences=False, name='first_layer') (x)
    output = Dense(1,activation='linear',name='output_layer') (x)
    model = keras.Model(inputs,output)
    opt=Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics='mae')
    
    history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test), 
                    epochs=n_epochs, 
                    batch_size=batch_size,
                    shuffle=False,
                    verbose=verbose,
                    callbacks=callbacks
    )
    
    return model,history


# define and fit LSTM model - 2 layer w/ input Normalize layer
def fit_LSTM_type3(X_train,y_train,X_test,y_test,n_lag,n_features,n1_neurons,n2_neurons,batch_size,n_epochs,learning_rate,verbose,callbacks):
    normalizer = Normalization(name='normalize_layer')
    normalizer.adapt(X_train)
    
    inputs = Input(shape=(n_lag, n_features),name='input_layer')
    x = normalizer(inputs)
    x = LSTM(n1_neurons,activation='tanh', return_sequences=True, name='first_layer') (x)
    x = LSTM(n2_neurons,activation='tanh', return_sequences=False, name='second_layer') (x)
    output = Dense(1,activation='linear',name='output_layer') (x)
    model = keras.Model(inputs,output)
    opt=Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics='mae')
    
    history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test), 
                    epochs=n_epochs, 
                    batch_size=batch_size,
                    shuffle=False,
                    verbose=verbose,
                    callbacks=callbacks
    )
    
    return model,history


def evaluate_model(model, X_train, y_train, X_test, y_test, verbose=0):
    train_mae = model.evaluate(X_train, y_train, verbose=0)[1]
    test_mae = model.evaluate(X_test, y_test, verbose=0)[1]
    
    return train_mae,test_mae















# def fit_LSTM_type3(X_train,y_train,X_test,y_test,n_lag,n_features,l1_neurons,batch_size,n_epochs,learning_rate,verbose,callbacks):
#     model = Sequential()
#     model.add(LSTM(l1_neurons, activation='relu', input_shape=(n_lag, n_features), name='first_layer'))
#     model.add(Dense(1, activation='linear',name='output'))
#     opt=Adam(learning_rate=learning_rate)
#     model.compile(optimizer=opt, loss='mse', metrics='mae')
    
#     history = model.fit(X_train, y_train,
#                     validation_data=(X_test, y_test), 
#                     epochs=n_epochs, 
#                     batch_size=batch_size,
#                     shuffle=False,
#                     verbose=verbose,
#                     callbacks=callbacks
#     )
#     return model,history

    
# def fit_LSTM_type4(X_train,y_train,X_test,y_test,n_lag,n_features,l1_neurons,batch_size,n_epochs,learning_rate,verbose,callbacks):
#     model = Sequential()
#     model.add(LSTM(l1_neurons, activation='relu',batch_input_shape=(batch_size, n_lag, n_features), stateful=True, name='first_layer'))
#     model.add(Dense(1, activation='linear'))
#     opt=Adam(learning_rate=learning_rate)
#     model.compile(optimizer=opt, loss='mse', metrics='mae')
    
#     for i in range(n_epochs):
#         history = model.fit(X_train, y_train,
#             validation_data=(X_test, y_test), 
#             epochs=1,
#             batch_size=batch_size,
#             shuffle=False,
#             verbose=verbose
#             # callbacks=callbacks
#         )
#         model.reset_states()
        
#     return model,history
