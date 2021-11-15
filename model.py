import numpy as np

from keras import backend as K
from keras import metrics
from keras.layers import (GRU, Conv1D, Conv2D, Dense, Dropout, Embedding, Flatten,
                          Input, Lambda, RepeatVector, Reshape,
                          TimeDistributed, Masking)
from keras.layers.merge import Concatenate, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

class DQN(object):
    def __init__(self):
        self.dqn = None

    def build_nn_model(self, 
                    rnn_dim=32, conv_window=3, conv_filters=16,
                    intermediate_dim=64, latent_dim=3, target_charset_size=115,
                    use_conditionals=True, use_kl_loss=True,
                    temp_time_len=8, max_material_length=14, charset_size=114,
                    max_num_precs=5, prec_conv_filters=64, prec_conv_window=3,
                    intermediate_dim_prec = 64, max_target_length=40,
                    stdev=1.0, kl_coeff=1.0, rec_coeff=1.0, 
                    max_step_size = 5,
                    num_elem = 80,
                    num_comp = 10,
                    ):

        s_material = Input(shape=(max_target_length, target_charset_size,), name="s_material") # State: One-hot encoding of material string
        s_step     = Input(shape=(max_step_size,), name="s_step") # State: One-hot encoding of step
        a_elem     = Input(shape=(num_elem,), name="a_elem") # Action: One-hot encoding of element to add
        a_comp     = Input(shape=(num_comp,), name="a_comp") # Action: One-hot encoding of composition of element

        # convolution for material
        conv_mat_x1   = Conv1D(filters=prec_conv_filters, kernel_size=prec_conv_window, padding="valid", activation="relu", name='conv_enc_mat_1')(s_material)
        conv_mat_x2   = Conv1D(filters=prec_conv_filters, kernel_size=prec_conv_window, padding="valid", activation="relu", name='conv_enc_mat_2')(conv_mat_x1)
        h_mat_flatten = Flatten()(conv_mat_x2)
        h_mat         = Dense(intermediate_dim, activation="relu", name="h_mat")(h_mat_flatten)

        # dense for the rest
        h_s_step = Dense(intermediate_dim, activation="relu", name="h_s_step")(s_step) # Hidden of s_step
        h_a_elem = Dense(intermediate_dim, activation="relu", name="h_a_elem")(a_elem) # Hidden of a_elem
        h_a_comp = Dense(intermediate_dim, activation="relu", name="h_a_comp")(a_comp) # Hidden of a_comp

        # Concatenate hidden embeddings of all 4 inputs
        h_combined = Concatenate(name="h_combined")([h_mat, h_s_step, h_a_elem, h_a_comp])
        
        # Prediction of Q_value
        penult_layer = Dense(intermediate_dim, activation="relu", name="penult_layer")(h_combined) # Penultimate layer
        Q_pred       = Dense(1, activation="linear", name="Q_pred")(penult_layer) # Final prediction - check if last layer requires a relu or linear - many websites use linear



        def mse_loss(Q, Q_pred):
            loss = metrics.mean_squared_error(Q, Q_pred)
            return loss

        dqn = Model(inputs=[
                            s_material, 
                            s_step,
                            a_elem,
                            a_comp
                            ],
                        outputs=[Q_pred])

        dqn.compile(
            optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=True),
            loss=[mse_loss],
            metrics=['mse']
            )

        self.dqn = dqn

    def train(self, inputs, outputs, epochs=300, val_data=None, val_split=0.0, batch_size=16, callbacks=None, verbose=2):
        fitting_results = self.dqn.fit(
            x=inputs,
            y=outputs,
            epochs=epochs,
            validation_split=val_split,
            validation_data=val_data,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
    #     For formatting on multiple inputs - https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
    #     It appears that the input should be a LIST of the different inputs
    #     model.fit(
	#     x=[trainAttrX, trainImagesX], y=trainY,
	#     validation_data=([testAttrX, testImagesX], testY),
	#     epochs=200, batch_size=8)

        return fitting_results.history


    # def save_models(self, model_variant="default", save_path="bin/CJK/"):
    #     self.vae.save_weights(save_path + model_variant + "_small_temp_time_vae.h5")
    #     self.encoder.save_weights(save_path + model_variant + "_small_temp_time_encoder.h5")
    #     self.decoder.save_weights(save_path + model_variant + "_small_temp_time_decoder.h5")

    # def load_models(self, model_variant="default", load_path="bin/CJK/"):
    #     self.vae.load_weights(load_path + model_variant + "_small_temp_time_vae.h5")
    #     self.encoder.load_weights(load_path + model_variant + "_small_temp_time_encoder.h5")
    #     self.decoder.load_weights(load_path  + model_variant + "_small_temp_time_decoder.h5")

dqn = DQN()
dqn.build_nn_model()
dqn.dqn.summary()
# print(dqn)