###############################################################################
# Project: Tuga_AlterEgo (Partial implementation of the MIT paper AlterEgo)   #
# Author:   Joao Nuno Carvalho                                                #
# Date:    2018.05.13                                                         #
# License: MIT Open Source License                                            #
# Description: This is a simulator of a partial implementation of the MIT     #
#              paper AlterEgo. The first DSP processing part.                 #
#              With this we show how to reconstruct the signal from the       #
#              sensors in the face of the device user.                        #
###############################################################################


import numpy as np
from random import random, seed
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA
import pickle   # Save to file and read from file.

import TextToElectrodeSignalGenerator as sigGen


class FilterProcessingDSP:

    def __init__(self, sample_rate ):
        self._sampleRate = sample_rate
        self._iirButterBandpass1_3Hz_50Hz__a = None
        self._iirButterBandpass1_3Hz_50Hz__b = None
        self._iirNotch50Hz__a = None
        self._iirNotch50Hz__b = None

    def applySpikeRemoval(self, input_sensor_array ):
        # From the paper.
        # The signal undergoes a representation transformation before
        # being input to the recognition model. We use a running
        # window average to identify and omit single spikes (> 30
        # uV above baseline) in the stream, with amplitudes greater
        # than average values for nearest 4 points before and after.

        res_sensor_array = input_sensor_array.copy()

        threashold = 1.5
        array_len = len(res_sensor_array)
        for i in range(0, array_len ):
            if res_sensor_array[i] > threashold:
                index_prev_2 = i - 2
                index_prev_1 = i - 1
                index_next_1 = i + 1
                index_next_2 = i + 2
                count = 0.0
                moving_average = 0.0
                if index_prev_2 >= 0:
                    if res_sensor_array[index_prev_2] < threashold:
                        moving_average += res_sensor_array[index_prev_2]
                        count += 1
                if index_prev_1 >= 0:
                    if res_sensor_array[index_prev_1] < threashold:
                        moving_average += res_sensor_array[index_prev_1]
                        count += 1
                if index_next_1 < array_len:
                    if res_sensor_array[index_next_1] < threashold:
                        moving_average += res_sensor_array[index_next_1]
                        count += 1
                if index_next_2 < array_len:
                    if res_sensor_array[index_next_2] < threashold:
                        moving_average += res_sensor_array[index_next_2]
                        count += 1
                if count >= 1:
                    moving_average /= count
                    res_sensor_array[i] = moving_average
        return res_sensor_array

    def applyFilter4OrderBut(self, input_sensor_array, design_filter = False ):

        if design_filter == True:
            # From: AlterEgo Paper
            #
            # The signals are fourth order IIR butterworth filtered (1.3 Hz to 50 Hz).
            # The high pass filter is used in order to prevent signal aliasing
            # artifacts. The low pass filter is applied to avoid movement
            # artifacts in the signal. A notch filter is applied at 60 Hz to
            # nullify line interference in hardware. The notch filter is
            # applied, despite the butterworth filter, because of the gentle
            # roll-off attenuation of the latter.

            # Digital filter.
            order_of_filter = 4

            band_pass_min_freq = 1.3 # Hz
            band_pass_min_freq_normalized = band_pass_min_freq / (self._sampleRate / 2.0)     # ex: 0.013 = 1.3 / (200.0 / 2.0)
            band_pass_max_freq = 50.0 # Hz
            band_pass_max_freq_normalized = band_pass_max_freq / (self._sampleRate / 2.0)     # ex: 0.5 = 50.0 / (200.0 / 2.0)
            band_pass_freq_normalized = [ band_pass_min_freq_normalized, band_pass_max_freq_normalized ]

            # b, a = signal.iirfilter(order_of_filter, [50, 200], rs=60, btype='band', analog = True, ftype = 'cheby2')
            b, a = signal.iirfilter(order_of_filter, band_pass_freq_normalized, btype='band', analog=False, ftype='butter')
            w, h = signal.freqs(b, a, 1000)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(w, 20 * np.log10(abs(h)))
            ax.set_xscale('log')
            ax.set_title('Butterworth 4º order bandpass frequency response')
            ax.set_xlabel('Frequency [radians / second]')
            ax.set_ylabel('Amplitude [dB]')
            ax.axis((0.001, 1000, -100, 100))
            ax.grid(which='both', axis='both')
            plt.show()

            self._iirButterBandpass1_3Hz_50Hz__a = a
            self._iirButterBandpass1_3Hz_50Hz__b = b

        # Applies the filter to the buffer, goind forward and backwords,
        # this will remove phase changes and makes the filter more linear in phase.
        #
        # filtered_buter_array_0 = signal.filtfilt(b, a, input_sensor_array)
        filtered_buter_array = signal.filtfilt(self._iirButterBandpass1_3Hz_50Hz__b, self._iirButterBandpass1_3Hz_50Hz__a, input_sensor_array)

        return filtered_buter_array

    def applyFilterNotch50Hz(self, input_sensor_array, design_filter = False ):

        if design_filter == True:
            # From: AlterEgo Paper
            #
            # A notch filter is applied at 60 Hz to
            # nullify line interference in hardware. The notch filter is
            # applied, despite the butterworth filter, because of the gentle
            # roll-off attenuation of the latter.

            fs = self._sampleRate  # Sample frequency (Hz)
            f0 = 50.0   # Frequency to be removed from signal (Hz)
            Q = 30.0  # Quality factor
            w0 = f0 / (fs / 2)  # Normalized Frequency
            # Design notch filter
            b, a = signal.iirnotch(w0, Q)


            # Frequency response
            w, h = signal.freqz(b, a)
            # Generate frequency axis
            freq = w * fs / (2 * np.pi)
            # Plot
            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            ax[0].plot(freq, 20 * np.log10(abs(h)), color='blue')
            ax[0].set_title("Frequency Response")
            ax[0].set_ylabel("Amplitude (dB)", color='blue')
            ax[0].set_xlim([0, 100])
            ax[0].set_ylim([-25, 10])
            ax[0].grid()
            ax[1].plot(freq, np.unwrap(np.angle(h)) * 180 / np.pi, color='green')
            ax[1].set_ylabel("Angle (degrees)", color='green')
            ax[1].set_xlabel("Frequency (Hz)")
            ax[1].set_xlim([0, 100])
            ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
            ax[1].set_ylim([-90, 90])
            ax[1].grid()
            plt.show()

            self._iirNotch50Hz__a = a
            self._iirNotch50Hz__b = b

        # Applies the filter to the buffer, goind forward and backwords,
        # this will remove phase changes and makes the filter more linear in phase.
        #
        # filtered_buter_array_0 = signal.filtfilt(b, a, input_sensor_array)
        filtered_buter_array = signal.filtfilt(self._iirNotch50Hz__b, self._iirNotch50Hz__a, input_sensor_array)

        return filtered_buter_array

    def applyFastICA(self, input_sensor_array_A, input_sensor_array_B, input_sensor_array_C):


        S = np.c_[input_sensor_array_A.copy(), input_sensor_array_B.copy(), input_sensor_array_C.copy()]
        S /= S.std(axis=0)  # Standardize data

        # The signal are already mixed.
        X = S


        # Criation of the mixing matrix, we execute the first time with many data and train to obtain the matrix,
        # then we use always the some matrix.
        my_mixing_matrix = np.array([[6.52578528,  1.33633918, 15.98838091],
                                    [17.20126137,  2.02137347,  0.17508984],
                                    [4.2600602, 15.83815388, 5.56819261]],
                                    dtype=float)

        file_name = "./saved_ICA_matrix.pickle"

        ##############
        # The model is already fited!!!!

        # # Compute ICA  [CALCULO REAL E QUE TEM DE SER FEITO UMA VEZ PARA CADA UTILIZADOR]
        # ica = FastICA(n_components=3, whiten = True )    # w_init =  matrix de mixing.
        #
        # ica.fit(X)
        #
        # # Saves to file.
        # pickle.dump( ica, open( file_name, "wb" ) )

        # Loads from file.
        ica = pickle.load( open( file_name, "rb" ) )

        S_ = ica.transform(X)   # Reconstruct signals

        # S_ = ica.fit_transform(X)  # Fits and Reconstruct signals

        A_ = ica.mixing_  # Get estimated mixing matrix
        print(A_)




        # # # We can `prove` that the ICA model applies by reverting the unmixing.
        # # assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)
        #
        # # #############################################################################
        # # Plot results
        #
        # plt.figure()
        #
        # models = [X, S, S_]
        # names = ['Observations (mixed signal)',
        #          'True Sources',
        #          'ICA recovered signals']
        # colors = ['red', 'steelblue', 'orange']
        #
        # for ii, (model, name) in enumerate(zip(models, names), 1):
        #     plt.subplot(3, 1, ii)
        #     plt.title(name)
        #     for sig, color in zip(model.T, colors):
        #         plt.plot(sig, color=color)
        #
        # plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
        # plt.show()

        # list of 3 splited matrix 1 X N
        out_sensor_array_A = np.zeros(input_sensor_array_A.shape, dtype=np.float32 )
        out_sensor_array_B = np.zeros(input_sensor_array_B.shape, dtype=np.float32)
        out_sensor_array_C = np.zeros(input_sensor_array_C.shape, dtype=np.float32)
        for i in range(0, S_.shape[0]): # 512*3
            out_sensor_array_A[i] = S_[i][0]
            out_sensor_array_B[i] = S_[i][1]
            out_sensor_array_C[i] = S_[i][2]

        #out_sensor_array_A, out_sensor_array_B, out_sensor_array_C = np.split(S_, 3, axis = 1)
        out_sensor_array_A_new = out_sensor_array_B.copy() * -1
        out_sensor_array_B_new = out_sensor_array_C.copy() * -1
        out_sensor_array_C_new = out_sensor_array_A.copy() * -1
        return [out_sensor_array_A_new, out_sensor_array_B_new, out_sensor_array_C_new]


##########
# Test Class of the above class.

class test_FilterProcessingDSP:

    def __init__(self):
        pass

    def run(self):

        # Generate the input signal.
        # TODO: The signal can be generated only once and saved to disk, then loaded from disk.

        print("\n\n")
        print("####################################")
        print("#  TextToElectrodeSignalGenerator  #")
        print("####################################")
        print("\n\n")

        sg = sigGen.TextToElectrodeSignalGenerator( )

        # Nota: Smaller parameters to permite debug, they can be very useful to train because they are so small,
        #       but be aware and alert of the size of the running window in the FFT following fase.
        sg.sampleRate                  = 200    # Samples/sec
        sg.charBaseDuration            = 300.0  # mS
        sg.charDeltaRandVelocity       = 0.0    # Between 0 and 1
        sg.blankCharDuration           = 300.0  # mS
        sg.finalPointSentenceDuration  = 300.0  # mS

        sg.enable50HzNoise             = True
        sg.enableDCComponent           = True
        sg.enableOutOfBandAddInSignal  = True
        sg.enableHeartSignal           = True
        sg.enableSignalSpikesGT30uV    = True
        sg.enableDistantMuscleMovement = False

        sg.printConfigParameters()

        sigGen.TextToElectrodeSignalGenerator.configLowerSignals( sg )

        # # Note: The first T is hear on purpose!
        # input_text = "tbatatinhas salteadas com couve lombarda."
        # array_sensor_0, array_sensor_1, array_sensor_2 = sg.generateSignalsFromText( input_text )
        #
        # index_small_len = int(len(array_sensor_0) / 12000.0)
        # index_array = np.arange(0, index_small_len )

        # Note: The first T is hear on purpose!
        input_text = "ttttt"   # "t b"

        gen_signal_sensor_0_pre,\
        gen_signal_sensor_1_pre,\
        gen_signal_sensor_2_pre,\
        gen_signal_sensor_0,\
        gen_signal_sensor_1,\
        gen_signal_sensor_2,\
        array_sensor_0,\
        array_sensor_1,\
        array_sensor_2 = sg.generateSignalsFromText(input_text)

        # Plot the 3 signals.
        sigGen.TextToElectrodeSignalGenerator.plotSignals(
                        gen_signal_sensor_0_pre,
                        gen_signal_sensor_1_pre,
                        gen_signal_sensor_2_pre,
                        gen_signal_sensor_0,
                        gen_signal_sensor_1,
                        gen_signal_sensor_2,
                        array_sensor_0,
                        array_sensor_1,
                        array_sensor_2)

        # dir_path = "./genFiles/eletrodeGenSignal_"
        # TextToElectrodeSignalGenerator.writeArrayToFile(array_sensor_0, dir_path + "0")
        # TextToElectrodeSignalGenerator.writeArrayToFile(array_sensor_1, dir_path + "1")
        # TextToElectrodeSignalGenerator.writeArrayToFile(array_sensor_2, dir_path + "2")
        #
        # array_sensor_0_from_file = TextToElectrodeSignalGenerator.loadArrayFromFile( dir_path + "0")
        # array_sensor_1_from_file = TextToElectrodeSignalGenerator.loadArrayFromFile( dir_path + "1")
        # array_sensor_2_from_file = TextToElectrodeSignalGenerator.loadArrayFromFile( dir_path + "2")


        ##########
        # Begin of DSP Filtering.


        print("\n\n")
        print("###########################")
        print("#  Filter_processing_DSP  #")
        print("###########################")
        print("\n\n")

        sample_rate = sg.sampleRate
        print("sampleRate: " + sample_rate.__str__())

        f_pro = FilterProcessingDSP( sample_rate = sample_rate)

        # Arrays com as interferências.
        # array_sensor_0
        # array_sensor_1
        # array_sensor_2

        # TODO:
        # In a future online implementation it will have to process the data in blocks for real time processing.

        filtered_output_0 = f_pro.applySpikeRemoval(array_sensor_0)
        filtered_output_1 = f_pro.applySpikeRemoval(array_sensor_1)
        filtered_output_2 = f_pro.applySpikeRemoval(array_sensor_2)

        filtered_output_0 = f_pro.applyFilter4OrderBut(filtered_output_0, design_filter = True)
        filtered_output_1 = f_pro.applyFilter4OrderBut(filtered_output_1)
        filtered_output_2 = f_pro.applyFilter4OrderBut(filtered_output_2)

        filtered_output_0 = f_pro.applyFilterNotch50Hz(filtered_output_0, design_filter = True)
        filtered_output_1 = f_pro.applyFilterNotch50Hz(filtered_output_1)
        filtered_output_2 = f_pro.applyFilterNotch50Hz(filtered_output_2)

        filtered_output_0,\
        filtered_output_1, \
        filtered_output_2 = f_pro.applyFastICA(filtered_output_0, filtered_output_1, filtered_output_2)


        # TODO:
        # Plot graph.

        index_all_len = len(array_sensor_0)
        index_array_all = np.arange(0, index_all_len )
        print("index_all_len:" + index_all_len.__str__())

        plt.figure(1)

        # Original Signals.
        plt.subplot(911)
        plt.plot(index_array_all, gen_signal_sensor_0_pre)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscule_Signal_sensor_0')

        plt.subplot(912)
        plt.plot(index_array_all, gen_signal_sensor_1_pre)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscule_Signal_sensor_1')

        plt.subplot(913)
        plt.plot(index_array_all, gen_signal_sensor_2_pre)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscule_Signal_sensor_2')


        # # Original Signals, mixed linearelly. Interferency between the facial and neck signals.
        # plt.subplot(911)
        # plt.plot(index_array_all, gen_signal_sensor_0)
        # plt.grid(True)
        # plt.ylabel('Amplitude')
        # plt.title('Muscule_Signal_sensor_mixed_inter_muscule_interferance_0')
        #
        # plt.subplot(912)
        # plt.plot(index_array_all, gen_signal_sensor_1)
        # plt.grid(True)
        # plt.ylabel('Amplitude')
        # plt.title('Muscule_Signal_sensor_mixed_inter_muscule_interferance_1')
        #
        # plt.subplot(913)
        # plt.plot(index_array_all, gen_signal_sensor_2)
        # plt.grid(True)
        # plt.ylabel('Amplitude')
        # plt.title('Muscule_Signal_sensor_mixed_inter_muscule_interferance_2')


        # Electrode signal with all interferences.
        plt.subplot(914)
        plt.plot(index_array_all, array_sensor_0)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscule_Signal_sensor_mixed_with_external_interferences_0')

        plt.subplot(915)
        plt.plot(index_array_all, array_sensor_1)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscule_Signal_sensor_mixed_with_external_interferences_1')

        plt.subplot(916)
        plt.plot(index_array_all, array_sensor_2)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscule_Signal_sensor_mixed_with_external_interferences_2')


        # Electrode signal with all interferences.
        plt.subplot(917)
        plt.plot(index_array_all, filtered_output_0)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Spike_removal_filter_pass_band_notch50Hz__0')

        plt.subplot(918)
        plt.plot(index_array_all, filtered_output_1)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Spike_removal_filter_pass_band_notch50Hz__1')

        plt.subplot(919)
        plt.plot(index_array_all, filtered_output_2)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Spike_removal_filter_pass_band_notch50Hz__2')

        plt.show()


if __name__ == "__main__":
    t = test_FilterProcessingDSP()
    t.run()