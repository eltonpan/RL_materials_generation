import numpy as np

from keras import backend as K
from keras import metrics
from keras.layers import (GRU, Conv1D, Conv2D, Dense, Dropout, Embedding, Flatten,
                          Input, Lambda, RepeatVector, Reshape,
                          TimeDistributed, Masking)
from keras.layers.merge import Concatenate, Subtract
from keras.models import Model
# from keras.optimizers import Adam # Old version in syn_gen_release env
from tensorflow.keras.optimizers import Adam # Updated for dqn env
from keras.callbacks import EarlyStopping
import torch
from torch import nn
from one_hot import _get_target_char_sequence, onehot_target # for testing purposes, to be deleted

# Keras DQN model
class DQN_keras(object):
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

# dqn = DQN_keras()
# dqn.build_nn_model()
# dqn.dqn.summary()

class DQN_pytorch(nn.Module):
  def __init__(self, max_target_length=40,
                     max_step_size = 5,
                     num_elem = 80,
                     num_comp = 10,
                     prec_conv_window = 3,
                     intermediate_dim = 64,
):
    super(DQN_pytorch, self).__init__()
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3)# 1st Conv for s_material
    self.conv2 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3)# 2nd Conv for s_material
    # self.conv3 = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3)# 3rd Conv for s_material
    self.act = nn.LeakyReLU() # Activation
    # self.fc1 = nn.Linear(3706, intermediate_dim) # Dense layer for s_material (for 3 conv)
    self.fc1 = nn.Linear(3996, intermediate_dim) # Dense layer for s_material (for 2 conv)
    # self.fc1 = nn.Linear(4294, intermediate_dim) # Dense layer for s_material (for only 1 conv)
    self.fc2 = nn.Linear(max_step_size, intermediate_dim) # Dense layer for s_step
    self.fc3 = nn.Linear(num_elem, intermediate_dim) # Dense layer for a_elem
    self.fc4 = nn.Linear(num_comp, intermediate_dim) # Dense layer for a_comp
    self.fc5 = nn.Linear(4*intermediate_dim, intermediate_dim) # 1st dense layer for h_combined
    self.fc6 = nn.Linear(intermediate_dim, 1) # Prediction head - 2nd dense layer for h_combined

    # self.fc_flatten_s_material = nn.Linear(40*115,intermediate_dim) # for flattening s_material

  def forward(self, s_material, # torch.Size([batch_size, 40, 115])
                    s_step,     # torch.Size([batch_size, 5])
                    a_elem,     # torch.Size([batch_size, 80])
                    a_comp      # torch.Size([batch_size, 10])
                    ):
    # For s_material
    s_material_before_r = s_material[0]
    s_material = s_material.reshape(s_material.shape[0],1,40,115).float() # Reshape for Conv2d  
    s_material_after_r = s_material[0][0]
    s_material = self.conv1(s_material) # 1st conv
    s_material = self.act(s_material) # ReLU
    s_material = self.conv2(s_material) # 2nd conv
    s_material = self.act(s_material) # ReLU
    # s_material = self.conv3(s_material) # 3rd conv
    # s_material = self.act(s_material) # ReLU
    # print(s_material.shape)
    s_material = s_material.reshape(s_material.shape[0],s_material.shape[-2]*s_material.shape[-1]) # batch size x 2D size after conv layer
    s_material = self.fc1(s_material) # Dense to (64)
    s_material = self.act(s_material) # Activation

    # # For s_material - flatten
    # s_material = torch.flatten(s_material, start_dim = 1, end_dim = -1).float()
    # s_material = self.fc_flatten_s_material(s_material) # Dense to (64)
    # s_material = self.act(s_material) # Activation

    # For s_step
    s_step = self.fc2(s_step.float())  # Dense to (64)
    s_step = self.act(s_step) # Activation

    # For a_elem
    a_elem = self.fc3(a_elem.float())   # Dense to (64)
    a_elem = self.act(a_elem) # Activation

    # For a_comp
    a_comp = self.fc4(a_comp.float())  # Dense to (64)
    a_comp = self.act(a_comp) # Activation

    # Concatenate all hidden and predict Q
    # print('input shapes')
    # print(s_material.shape)
    # print(s_step.shape)
    # print(a_elem.shape)
    # print(a_comp.shape)
    # print('s_material:', torch.mean(s_material))
    # print('s_step:', torch.mean(s_step))
    # print('a_elem:', torch.mean(a_elem))
    # print('a_comp:', torch.mean(a_comp))

    h_combined = torch.cat((s_material, s_step, a_elem, a_comp),1) # Cat to (batch_size, 4*64) hence cat across columns hence index 1
    # print(h_combined.shape)
    h_combined = self.fc5(h_combined) # Dense 1
    h_combined = self.act(h_combined) # Act

    Q_pred = self.fc6(h_combined) # Dense 2 with NO activation for final hidden layer

    return Q_pred

if __name__ == "__main__":
    s_material = torch.tensor(onehot_target('BaTiO3'))
    s_material = s_material.reshape(1, s_material.shape[0], s_material.shape[1])
    print(s_material.shape)

    s_step = torch.zeros(5)
    s_step[2] = 1.
    s_step = s_step.reshape(1, s_step.shape[0])
    print(s_step.shape)

    a_elem = torch.zeros(80)
    a_elem[1] = 1.
    a_elem = a_elem.reshape(1, a_elem.shape[0])
    print(a_elem.shape)

    a_comp = torch.zeros(10)
    a_comp[3] = 1.
    a_comp = a_comp.reshape(1, a_comp.shape[0])
    print(a_comp.shape)

    dqn = DQN_pytorch()
    output = dqn(s_material, s_step, a_elem, a_comp)
    print(output)
    print(output.shape)
