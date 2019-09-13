###############################################################################
# Project: Tuga_AlterEgo (Partial implementation of the MIT paper AlterEgo)   #
# Author:   Joao Nuno Carvalho                                                #
# Date:    2018.05.13                                                         #
# License: MIT Open Source License                                            #
# Description: This is a simulator of a partial implementation of the MIT     #
#              paper AlterEgo. The first DSP processing part.                 #
#              With this we show how to reconstruct the signal from the       #
#              sensors in the face of the device user.                        #
#                                                                             #
# This class generates test signals of muscle electrodes to simulate the      #
# signals that arrive at the USB port via hardware sensor board. The sensor   #
# board has electrodes, instrumentation ampops, 24 bits ADC and a             # 
# microcontroller to send the signal from SPI of the ADC to serial via USB.   #
#                                                                             #
#  This class receives:                                                       #
#    -an input text,                                                          #
#    -a configuration of the channels and signals                             #
#    -and a char to channel sequence map.                                     #
#                                                                             #
#  You can specify the duration of the signal to be generated.                #
###############################################################################


import numpy as np
from random import random, seed
import matplotlib.pyplot as plt

class TextToElectrodeSignalGenerator:

    ##########
    # Static method's

    @staticmethod
    def writeArrayToFile(array_sensor, path_to_file):
        if not path_to_file.endswith(".npy"):
            # Adds the path to the end.
            path_to_file += ".npy"
        np.save(path_to_file, array_sensor)  # .npy extension is added if not given

    @staticmethod
    def loadArrayFromFile(path_to_file):
        array_sensor = np.load(path_to_file)   # ".npy" extension.
        return array_sensor

    ##########
    # Instance method's

    def __init__(self,
                 sample_rate                      = 250,   # Samples/sec
                 char_base_duration               = 200.0, # mS
                 char_delta_rand_velocity         = 0.3,   # Between 0 and 1
                 blank_char_duration              = 200.0, # mS
                 final_point_sentence_duration    = 500.0, # mS
                 enable_50hz_noise                = False,
                 enable_dc_component              = False,
                 enable_out_of_band_add_in_signal = False,
                 enable_heart_signal              = False,
                 enable_signal_spikes_gt_30uV     = False,
                 enable_distant_muscle_movement   = False ):

        self._dic_char_to_signal = self._fill_char_map_to_signal_combinations()
        self._sensorSignalsFreq = []

        self._sampleRate                  = sample_rate    # Samples/sec

        self._charBaseDuration            = char_base_duration             # ms
        self._charDeltaRandVelocity       = char_delta_rand_velocity
        self._blankCharDuration           = blank_char_duration            # ms
        self._finalPointSentenceDuration  = final_point_sentence_duration  # ms

        self._enable50HzNoise             = enable_50hz_noise
        self._enableDCComponent           = enable_dc_component
        self._enableOutOfBandAddInSignal  = enable_out_of_band_add_in_signal
        self._enableHeartSignal           = enable_heart_signal
        self._enableSignalSpikesGT30uV    = enable_signal_spikes_gt_30uV
        self._enableDistantMuscleMovement = enable_distant_muscle_movement

        # Set initial seed, to have reproducible runs.
        seed(0)


    def _fill_char_map_to_signal_combinations(self):

        dic_char_to_signal = {
            # Char frequency that appears in Portuguese.
            " ": "BLANK_SIGNAL",
            "a": "000",  # 14,63%
            "b": "001",  # 1,04%
            "c": "002",  # 3,88%
            "d": "010",  # 5,01%
            "e": "011",  # 12,57%
            "f": "012",  # 1,02%
            "g": "020",  # 1,30%
            "h": "021",  # 1,28%
            "i": "022",  # 6,18%
            "j": "100",  # 0,40%
            "k": "101",  # 0,02%
            "l": "102",  # 2,78%
            "m": "110",  # 4,74%
            "n": "111",  # 5,05%
            "o": "112",  # 10,73%
            "p": "120",  # 2,52%
            "q": "121",  # 1,20%
            "r": "122",  # 6,53%
            "s": "200",  # 7,81%
            "t": "201",  # 4,34%
            "u": "202",  # 4,63%
            "v": "210",  # 1,67%
            "w": "211",  # 0,01%
            "x": "212",  # 0,21%
            "y": "220",  # 0,01%
            "z": "221",  # 0,47%
            # ".": "222",
            ".": "BLANK_SIGNAL",
            "ç": "300",
            "à": "301",
            "á": "302",
            "é": "310",
            "ã": "311",
            "õ": "312",
            "ê": "320",
            # "!": "321"
            "!": "BLANK_SIGNAL"
        }
        return dic_char_to_signal

    @property
    def sampleRate(self):
        return self._sampleRate

    @sampleRate.setter
    def sampleRate(self, value):
        self._sampleRate = value

    @property
    def charBaseDuration(self):
        # print("Getting value")
        return self._charBaseDuration

    @charBaseDuration.setter
    def charBaseDuration(self, value):
        #if value < -273:
        #    raise ValueError("Temperature below -273 is not possible")
        # print("Setting value")
        self._charBaseDuration = float(value)

    # Note: To use the property inside the class in other method or outside the method use:
    #         self.temperature

    @property
    def charDeltaRandVelocity(self):
        return self._charDeltaRandVelocity

    @charDeltaRandVelocity.setter
    def charDeltaRandVelocity(self, value):
        self._charDeltaRandVelocity = float(value)

    @property
    def blankCharDuration(self):
        return self._blankCharDuration

    @blankCharDuration.setter
    def blankCharDuration(self, value):
        self._blankCharDuration = float(value)

    @property
    def finalPointSentenceDuration(self):
        return self._finalPointSentenceDuration

    @finalPointSentenceDuration.setter
    def finalPointSentenceDuration(self, value):
        self._finalPointSentenceDuration = float(value)

    ###########
    # Enable

    @property
    def enable50HzNoise(self):
        return self._enable50HzNoise

    @enable50HzNoise.setter
    def enable50HzNoise(self, value):
        self._enable50HzNoise = value

    @property
    def enableDCComponent(self):
        return self._enableDCComponent

    @enableDCComponent.setter
    def enableDCComponent(self, value):
        self._enableDCComponent = value

    @property
    def enableOutOfBandAddInSignal(self):
        return self._enableOutOfBandAddInSignal

    @enableOutOfBandAddInSignal.setter
    def enableOutOfBandAddInSignal(self, value):
        self._enableOutOfBandAddInSignal = value

    @property
    def enableHeartSignal(self):
        return self._enableHeartSignal

    @enableHeartSignal.setter
    def enableHeartSignal(self, value):
        self._enableHeartSignal= value

    @property
    def enableSignalSpikesGT30uV(self):
        return self._enableSignalSpikesGT30uV

    @enableSignalSpikesGT30uV.setter
    def enableSignalSpikesGT30uV(self, value):
        self._enableSignalSpikesGT30uV= value

    @property
    def enableDistantMuscleMovement(self):
        return self._enableDistantMuscleMovement

    @enableDistantMuscleMovement.setter
    def enableDistantMuscleMovement(self, value):
        self._enableDistantMuscleMovement= value


    #######
    # Code
    #

    def printConfigParameters(self):
        print("sampleRate: "                 + self.sampleRate.__str__() )
        print("charBaseDuration: "           + self.charBaseDuration.__str__() )
        print("charDeltaRandVelocity: "      + self.charDeltaRandVelocity.__str__() )
        print("blankCharDuration: "          + self.blankCharDuration.__str__() )
        print("finalPointSentenceDuration: " + self.finalPointSentenceDuration.__str__() )
        print("\n")
        print("enable50HzNoise:"             + self.enable50HzNoise.__str__() )
        print("enableDCComponent:"           + self.enableDCComponent.__str__() )
        print("enableOutOfBandAddInSignal:"  + self.enableOutOfBandAddInSignal.__str__() )
        print("enableHeartSignal:"           + self.enableHeartSignal.__str__() )
        print("enableSignalSpikesGT30uV:"    + self.enableSignalSpikesGT30uV.__str__() )
        print("enableDistantMuscleMovement:" + self.enableDistantMuscleMovement.__str__() )

    # sg.addSensorSignalsFreq( sensorID = 0,
    #                          description = "Face right of mouth eletrode",
    #                          num_signals = 3,
    #                          signals = [5.0, 15.0, 25.0] )

    def addSensorSignalsFreq(self, sensorID, description, num_signals, signals ):
        dic = { "sensorID"    : sensorID,
                "description" : description,
                "num_signals" : num_signals,
                "signals"     : signals            # This is a list of floats of frequencies.
               }
        self._sensorSignalsFreq.append(dic)

    def getSensorSignalsFreq(self):
        return self._sensorSignalsFreq

    def generateSignalsFromText(self, input_text ):
        gen_signal_sensor_0_pre, gen_signal_sensor_1_pre, gen_signal_sensor_2_pre,\
        gen_signal_sensor_0, gen_signal_sensor_1, gen_signal_sensor_2 = self._generate3SensorsSignals( input_text )

        array_sensor_0, array_sensor_1, array_sensor_2 = self._addInterferenceSignals(gen_signal_sensor_0,
                                                                                      gen_signal_sensor_1,
                                                                                      gen_signal_sensor_2 )
        ret_list = [ gen_signal_sensor_0_pre, gen_signal_sensor_1_pre, gen_signal_sensor_2_pre,
                     gen_signal_sensor_0, gen_signal_sensor_1, gen_signal_sensor_2,
                     array_sensor_0, array_sensor_1, array_sensor_2 ]
        return ret_list

    def _generate3SensorsSignals(self, input_text):

        # Determine the max_buffer_size to allocate with zeros.
        max_signal_char_lenght = max([self._charBaseDuration,
                                      self._blankCharDuration,
                                      self._finalPointSentenceDuration ])
        max_signal_char_lenght = max_signal_char_lenght + max_signal_char_lenght * self._charDeltaRandVelocity + 0.1
        max_samples_per_char = int( round((max_signal_char_lenght / 1000.0) * self._sampleRate ))
        max_buffer_size = len(input_text) * max_samples_per_char \
                          + len(input_text)     # This last len adition is only for the math rounding error.

        # Buffer allocation.
        gen_signal_sensor_0_pre = np.zeros(max_buffer_size, dtype = np.float32)
        gen_signal_sensor_1_pre = np.zeros(max_buffer_size, dtype = np.float32)
        gen_signal_sensor_2_pre = np.zeros(max_buffer_size, dtype = np.float32)

        gen_signal_sensor_0 = np.zeros(max_buffer_size, dtype = np.float32)
        gen_signal_sensor_1 = np.zeros(max_buffer_size, dtype = np.float32)
        gen_signal_sensor_2 = np.zeros(max_buffer_size, dtype = np.float32)

        # Index where to write next.
        gen_index = 0

        # TODO: Implement!
        #input_text = "texto"

        input_text = input_text.lower()
        for cur_char in input_text:
            cur_char_signal_duration = None
            cur_char_signal_duration_extra_rate =  self._charDeltaRandVelocity * random()
            if cur_char == " ":
                # Empty/space char.
                cur_char_signal_duration = self._blankCharDuration
            elif cur_char == "." or cur_char == "!":
                cur_char_signal_duration = self._finalPointSentenceDuration
            else:
                # Evey other char.
                cur_char_signal_duration = self._charBaseDuration
            # Pass every caracter lenght from mS to Seconds.
            cur_char_signal_duration /= 1000.0
            cur_char_signal_duration =  cur_char_signal_duration \
                                        + cur_char_signal_duration * cur_char_signal_duration_extra_rate
            cur_char_signal_samples =  int(round(cur_char_signal_duration * self._sampleRate))

            gen_char_0_pre = self._generateCharSignal(cur_char, cur_char_signal_duration, cur_char_signal_samples, sensor = 0, flag_print = True)
            gen_char_1_pre = self._generateCharSignal(cur_char, cur_char_signal_duration, cur_char_signal_samples, sensor = 1)
            gen_char_2_pre = self._generateCharSignal(cur_char, cur_char_signal_duration, cur_char_signal_samples, sensor = 2)

            # TODO: Parametrize this values!

            # We are simulating the interference between the 3 muscle signals on the electrodes.
            # gen_char_0 = gen_char_0_pre * 0.8 + gen_char_1_pre * 0.1 + gen_char_2_pre * 0.1
            # gen_char_1 = gen_char_0_pre * 0.1 + gen_char_1_pre * 0.8 + gen_char_2_pre * 0.1
            # gen_char_2 = gen_char_0_pre * 0.1 + gen_char_1_pre * 0.1 + gen_char_2_pre * 0.8

            gen_char_0 = gen_char_0_pre * 0.7 + gen_char_1_pre * 0.15 + gen_char_2_pre * 0.15
            gen_char_1 = gen_char_0_pre * 0.15 + gen_char_1_pre * 0.7 + gen_char_2_pre * 0.15
            gen_char_2 = gen_char_0_pre * 0.15 + gen_char_1_pre * 0.15 + gen_char_2_pre * 0.8


            # Appends the char signal samples to the end of each buffer 0, 1, and 2.
            start_index = gen_index
            end_index   = gen_index + cur_char_signal_samples

            print("\n")
            print("gen_index: " + gen_index.__str__())
            print("start_index: " + start_index.__str__())
            print("end_index: " + end_index.__str__())

            gen_signal_sensor_0_pre[start_index : end_index] = gen_char_0_pre[0 : cur_char_signal_samples].copy()
            gen_signal_sensor_1_pre[start_index : end_index] = gen_char_1_pre[0 : cur_char_signal_samples].copy()
            gen_signal_sensor_2_pre[start_index : end_index] = gen_char_2_pre[0 : cur_char_signal_samples].copy()

            gen_signal_sensor_0[start_index : end_index] = gen_char_0[0 : cur_char_signal_samples].copy()
            gen_signal_sensor_1[start_index : end_index] = gen_char_1[0 : cur_char_signal_samples].copy()
            gen_signal_sensor_2[start_index : end_index] = gen_char_2[0 : cur_char_signal_samples].copy()
            gen_index = end_index


        gen_signal_sensor_0_pre_out = gen_signal_sensor_0_pre[0 : gen_index].copy()
        gen_signal_sensor_1_pre_out = gen_signal_sensor_1_pre[0 : gen_index].copy()
        gen_signal_sensor_2_pre_out = gen_signal_sensor_2_pre[0 : gen_index].copy()

        gen_signal_sensor_0_out = gen_signal_sensor_0[0 : gen_index].copy()
        gen_signal_sensor_1_out = gen_signal_sensor_1[0 : gen_index].copy()
        gen_signal_sensor_2_out = gen_signal_sensor_2[0 : gen_index].copy()

        return [gen_signal_sensor_0_pre_out, gen_signal_sensor_1_pre_out, gen_signal_sensor_2_pre_out,
                gen_signal_sensor_0_out, gen_signal_sensor_1_out, gen_signal_sensor_2_out]

    def _generateCharSignal(self, cur_char, cur_char_signal_duration, cur_char_signal_samples, sensor, flag_print = False):
        # Allocate the buffer array for a single char signal.

        signal_mask = self._dic_char_to_signal[cur_char]
        if flag_print == True:
            print("{} - {}".format(cur_char, signal_mask))

        if signal_mask == "BLANK_SIGNAL":
            char_signal_array = np.zeros((cur_char_signal_samples,), dtype=np.float32)
            return char_signal_array

        sig_num = int(signal_mask[2 - sensor])

        # self._sensorSignalsFreq[sensor]["signals"][sig_num]

        # amplitude = 1.0  # range [0.0, 1.0]
        # fs        = self._sampleRate  # sampling rate, Hz, must be integer
        # f         = self._sensorSignalsFreq[sensor]["signals"][sig_num]  # ex: 5.0  sine frequency, Hz, has to be float
        # # samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)
        #
        # # generate samples, note conversion to float32 array
        # char_signal_array = (amplitude * np.sin(2 * np.pi * np.arange(cur_char_signal_samples) * f / fs)).astype(np.float32)

        array_number_of_samples = cur_char_signal_samples
        signal_frequency        = self._sensorSignalsFreq[sensor]["signals"][sig_num]  # ex: 5.0  sine frequency, Hz, has to be float
        sampling_rate           = self._sampleRate  # sampling rate, Hz, must be integer
        amplitude               = 1.0  # range [0.0, 1.0]
        char_signal_array = self._generateSinSignalArray( array_number_of_samples, signal_frequency, sampling_rate, amplitude)

        return char_signal_array

    def _generateSinSignalArray(self, array_number_of_samples, signal_frequency, sampling_rate = None, amplitude = 1.0 ):
        # amplitude = 1.0  # range [0.0, 1.0]
        # fs        = self._sampleRate  # sampling rate, Hz, must be integer
        # f         = 5.0  # ex: 5.0  sine frequency, Hz, has to be float
        #
        # generate samples, note conversion to float32 array
        # samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

        if sampling_rate is None:
            sampling_rate = self._sampleRate
        signal_array = (amplitude * np.sin(2 * np.pi * np.arange(array_number_of_samples) * signal_frequency / sampling_rate)).astype(np.float32)
        return signal_array

    def _addInterferenceSignals(self, input_signal_array_0, input_signal_array_1, input_signal_array_2 ):

        # _sampleRate

        # The 3 input signals have the same length.
        tot_num_of_samples = len(input_signal_array_0)

        # Initializes a zeroed array with the same size.
        temp_array = None

        total_of_signals = 0.0

        if self._enable50HzNoise == True:
            temp_array = self._gen50HzNoise(tot_num_of_samples)
            total_of_signals += 1
        if self._enableDCComponent == True:
            temp_array += self._genDCComponent(tot_num_of_samples)
            total_of_signals += 1
        if self._enableOutOfBandAddInSignal == True:
            temp_array += self._genOutOfBandAddInSignal(tot_num_of_samples)
            total_of_signals += 1
        if self._enableHeartSignal == True:
            temp_array += self._genHeartSignal(tot_num_of_samples)
            total_of_signals += 1
        if self._enableSignalSpikesGT30uV == True:
            temp_array += self._genSignalSpikesGT30uV(tot_num_of_samples)
            total_of_signals += 1
        if self._enableDistantMuscleMovement == True:
            temp_2_array = self._genDistantMuscleMovement(tot_num_of_samples)
            if temp_2_array is not None:
                temp_array += temp_2_array
                total_of_signals += 1

        if temp_array is not None and total_of_signals != 0.0:
            temp_array /= total_of_signals

        ret_array_0 = input_signal_array_0.copy()
        ret_array_1 = input_signal_array_1.copy()
        ret_array_2 = input_signal_array_2.copy()

        if temp_array is not None:
            ret_array_0 += temp_array
            ret_array_1 += temp_array
            ret_array_2 += temp_array

            ret_array_0 /= 2.0
            ret_array_1 /= 2.0
            ret_array_2 /= 2.0

        # TODO: Normalize the signals between 0 and 1.

        return [ret_array_0, ret_array_1, ret_array_2]


    def _gen50HzNoise(self, tot_num_of_samples):
        # Generation os a 50Hz electrical wire signal interference.
        array_number_of_samples = tot_num_of_samples
        signal_frequency        = 50.0  # ex: 5.0  sine frequency, Hz, has to be float
        sampling_rate           = self._sampleRate  # sampling rate, Hz, must be integer
        amplitude               = 1.0  # range [0.0, 1.0]
        signal_array = self._generateSinSignalArray( array_number_of_samples, signal_frequency, sampling_rate, amplitude)
        return signal_array

    def _genDCComponent(self, tot_num_of_samples):
        # Add's a DC component to the signal.
        # This will have to be filtered out in the processing stages, when the filters are applied.
        # Create's the sample buffer.
        gen_signal = np.zeros(tot_num_of_samples, dtype=np.float32)
        gen_signal += 1.0
        return gen_signal

    def _genOutOfBandAddInSignal(self, tot_num_of_samples):
        # Generation os a 65Hz signal interference upper band signal.
        array_number_of_samples = tot_num_of_samples
        signal_frequency        = 65.0  # ex: 5.0  sine frequency, Hz, has to be float
        sampling_rate           = self._sampleRate  # sampling rate, Hz, must be integer
        amplitude               = 1.0  # range [0.0, 1.0]
        signal_array = self._generateSinSignalArray( array_number_of_samples, signal_frequency, sampling_rate, amplitude)
        return signal_array

    def _genHeartSignal(self, tot_num_of_samples):
        # Generation os a 0.15Hz normal heart rate frequency interference.
        array_number_of_samples = tot_num_of_samples
        signal_frequency        = 0.15  # ex: 5.0  sine frequency, Hz, has to be float
        sampling_rate           = self._sampleRate  # sampling rate, Hz, must be integer
        amplitude               = 1.0  # range [0.0, 1.0]
        signal_array = self._generateSinSignalArray( array_number_of_samples, signal_frequency, sampling_rate, amplitude)
        return signal_array

    def _genSignalSpikesGT30uV(self, tot_num_of_samples):
        # Create's the sample buffer.
        gen_signal = np.zeros(tot_num_of_samples, dtype=np.float32)
        for index in range(0, tot_num_of_samples, 20):
            gen_signal[index] += 15.0
        return gen_signal

    def _genDistantMuscleMovement(self, tot_num_of_samples):
        # Create's the sample buffer.

        # TODO: I have to think better about this one!!!
        #       Don't have a real ideia of how to simulate this one,
        #       I have to read the book of Medical Instrumentation,
        #       but it didn't arrive yet.

        return None

    def _genGausianNoise(self, tot_num_of_samples):
        # Create's the sample buffer.

        # TODO: Implment!!!
        #       This should be added to each signal sensor.

        return None

    @staticmethod
    def plotSignals(gen_signal_sensor_0_pre,
                    gen_signal_sensor_1_pre,
                    gen_signal_sensor_2_pre,
                    gen_signal_sensor_0,
                    gen_signal_sensor_1,
                    gen_signal_sensor_2,
                    array_sensor_0,
                    array_sensor_1,
                    array_sensor_2):

        print("\n")
        index_small_len = int(len(array_sensor_0) / 10.0)
        index_array_small = np.arange(0, index_small_len)
        print("index_small_len:" + index_small_len.__str__())

        index_all_len = len(array_sensor_0)
        index_array_all = np.arange(0, index_all_len )
        print("index_all_len:" + index_all_len.__str__())

        plt.figure(1)

        # # Signals zoomed in.
        # plt.subplot(911)
        # plt.plot(index_array_small, array_sensor_0[0 : index_small_len])
        # plt.grid(True)
        # plt.ylabel('Amplitude')
        # plt.title('array_sensor_0')
        #
        # plt.subplot(912)
        # plt.plot(index_array_small, array_sensor_1[0 : index_small_len])
        # plt.grid(True)
        # plt.ylabel('Amplitude')
        # plt.title('array_sensor_1')
        #
        # plt.subplot(913)
        # plt.plot(index_array_small, array_sensor_2[0 : index_small_len])
        # plt.grid(True)
        # plt.ylabel('Amplitude')
        # plt.title('array_sensor_2')


        # Original Signals.
        plt.subplot(911)
        plt.plot(index_array_all, gen_signal_sensor_0_pre)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscle_Signal_sensor_0')

        plt.subplot(912)
        plt.plot(index_array_all, gen_signal_sensor_1_pre)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscle_Signal_sensor_1')

        plt.subplot(913)
        plt.plot(index_array_all, gen_signal_sensor_2_pre)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscle_Signal_sensor_2')


        # Original Signals, mixed linearelly. Interferency between the facial and neck signals.
        plt.subplot(914)
        plt.plot(index_array_all, gen_signal_sensor_0)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscle_Signal_sensor_mixed_inter_muscle_interference_0')

        plt.subplot(915)
        plt.plot(index_array_all, gen_signal_sensor_1)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscle_Signal_sensor_mixed_inter_muscle_interference_1')

        plt.subplot(916)
        plt.plot(index_array_all, gen_signal_sensor_2)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscle_Signal_sensor_mixed_inter_muscle_interference_2')


        # Electrode signal with all interferences.
        plt.subplot(917)
        plt.plot(index_array_all, array_sensor_0)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscle_Signal_sensor_mixed_with_external_interferences_0')

        plt.subplot(918)
        plt.plot(index_array_all, array_sensor_1)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscle_Signal_sensor_mixed_with_external_interferences_1')

        plt.subplot(919)
        plt.plot(index_array_all, array_sensor_2)
        plt.grid(True)
        plt.ylabel('Amplitude')
        plt.title('Muscle_Signal_sensor_mixed_with_external_interferences_2')

        plt.show()


    @staticmethod
    def configLowerSignals(signal_gen_object):
        sg = signal_gen_object  # This is an object of this class initialized by a constructor.

        sg.addSensorSignalsFreq( sensorID = 0,
                                 description = "Face right of mouth electrode",
                                 num_signals = 3,
                                 signals = [5.0, 15.0, 25.0] )

        sg.addSensorSignalsFreq( sensorID = 1,
                                 description = "Below mouth electrode",
                                 num_signals = 2,
                                 signals = [5.0, 15.0, 25.0] )

        sg.addSensorSignalsFreq( sensorID = 2,
                                 description = "Neck electrode",
                                 num_signals = 4,
                                 signals = [5.0, 15.0, 25.0, 35.0] )

        print("\n")
        print( sg.getSensorSignalsFreq( ) )
        print("\n")



##########
# Test Class of the above class.

class test_TextToElectrodeSignalGenerator:

    def __init__(self):
        pass

    def run(self):
        sg = TextToElectrodeSignalGenerator( sample_rate = 250,
                                             char_base_duration = 200.0,
                                             char_delta_rand_velocity = 0.3,
                                             blank_char_duration = 200.0,
                                             final_point_sentence_duration = 500.0 )

        print("sampleRate: " + sg.sampleRate.__str__() )
        sg.sampleRate = 300
        print("sampleRate: " + sg.sampleRate.__str__() )

        print("charBaseDuration: " + sg.charBaseDuration.__str__() )
        sg.charBaseDuration = 300
        print("charBaseDuration: " + sg.charBaseDuration.__str__() )

        print("charDeltaRandVelocity: " + sg.charDeltaRandVelocity.__str__() )
        sg.charDeltaRandVelocity = 0.4
        print("charDeltaRandVelocity: " + sg.charDeltaRandVelocity.__str__() )

        print("blankCharDuration: " + sg.blankCharDuration.__str__() )
        sg.blankCharDuration = 0.4
        print("blankCharDuration: " + sg.blankCharDuration.__str__() )

        print("finalPointSentenceDuration: " + sg.finalPointSentenceDuration.__str__() )
        sg.finalPointSentenceDuration = 0.4
        print("finalPointSentenceDuration: " + sg.finalPointSentenceDuration.__str__() )


        # Nota: Smaller parameters for debug proposes, they can be usefull when training because they are small,
        #       but we have to be alert to the size of the running window in the FFT phase ( in front).
        sg.sampleRate                 = 200   # Samples/sec
        sg.charBaseDuration           = 300.0  # mS
        sg.charDeltaRandVelocity      = 0.0   # Between 0 and 1
        sg.blankCharDuration          = 300.0  # mS
        sg.finalPointSentenceDuration = 300.0  # mS

        print("sampleRate: "                 + sg.sampleRate.__str__() )
        print("charBaseDuration: "           + sg.charBaseDuration.__str__() )
        print("charDeltaRandVelocity: "      + sg.charDeltaRandVelocity.__str__() )
        print("blankCharDuration: "          + sg.blankCharDuration.__str__() )
        print("finalPointSentenceDuration: " + sg.finalPointSentenceDuration.__str__() )

        print("\n")
        print("enable50HzNoise:"             + sg.enable50HzNoise.__str__() )
        print("enableDCComponent:"           + sg.enableDCComponent.__str__() )
        print("enableOutOfBandAddInSignal:"  + sg.enableOutOfBandAddInSignal.__str__() )
        print("enableHeartSignal:"           + sg.enableHeartSignal.__str__() )
        print("enableSignalSpikesGT30uV:"    + sg.enableSignalSpikesGT30uV.__str__() )
        print("enableDistantMuscleMovement:" + sg.enableDistantMuscleMovement.__str__() )

        sg.enable50HzNoise             = True
        sg.enableDCComponent           = True
        sg.enableOutOfBandAddInSignal  = True
        sg.enableHeartSignal           = True
        sg.enableSignalSpikesGT30uV    = True
        sg.enableDistantMuscleMovement = True

        print("\n")
        print("enable50HzNoise:"             + sg.enable50HzNoise.__str__() )
        print("enableDCComponent:"           + sg.enableDCComponent.__str__() )
        print("enableOutOfBandAddInSignal:"  + sg.enableOutOfBandAddInSignal.__str__() )
        print("enableHeartSignal:"           + sg.enableHeartSignal.__str__() )
        print("enableSignalSpikesGT30uV:"    + sg.enableSignalSpikesGT30uV.__str__() )
        print("enableDistantMuscleMovement:" + sg.enableDistantMuscleMovement.__str__() )

        # sg.enable50HzNoise             = False
        # sg.enableDCComponent           = False
        # sg.enableOutOfBandAddInSignal  = False
        # sg.enableHeartSignal           = False
        # sg.enableSignalSpikesGT30uV    = False
        # sg.enableDistantMuscleMovement = False
        #
        # print("\n")
        # print("enable50HzNoise:"             + sg.enable50HzNoise.__str__() )
        # print("enableDCComponent:"           + sg.enableDCComponent.__str__() )
        # print("enableOutOfBandAddInSignal:"  + sg.enableOutOfBandAddInSignal.__str__() )
        # print("enableHeartSignal:"           + sg.enableHeartSignal.__str__() )
        # print("enableSignalSpikesGT30uV:"    + sg.enableSignalSpikesGT30uV.__str__() )
        # print("enableDistantMuscleMovement:" + sg.enableDistantMuscleMovement.__str__() )

        TextToElectrodeSignalGenerator.configLowerSignals( sg )


        # # Note: The first T is hear on purpose!
        # input_text = "tbatatinhas salteadas com couve lombarda."
        # array_sensor_0, array_sensor_1, array_sensor_2 = sg.generateSignalsFromText( input_text )
        #
        # index_small_len = int(len(array_sensor_0) / 12000.0)
        # index_array = np.arange(0, index_small_len )

        # Note: The first T is hear on purpose!
        input_text = "t b"
        gen_signal_sensor_0_pre, gen_signal_sensor_1_pre, gen_signal_sensor_2_pre, \
        gen_signal_sensor_0, gen_signal_sensor_1, gen_signal_sensor_2,\
        array_sensor_0, array_sensor_1, array_sensor_2 = sg.generateSignalsFromText(input_text)

        # Plot the 3 signals.
        TextToElectrodeSignalGenerator.plotSignals(
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
        #
        # if (    array_sensor_0 == array_sensor_0_from_file
        #     and array_sensor_1 == array_sensor_1_from_file
        #     and array_sensor_2 == array_sensor_2_from_file ):
        #     print("The files written and load from disk are equal!")
        # else:
        #     print("Error: The files written and load from disk are not equal!")


if __name__ == "__main__":
    t = test_TextToElectrodeSignalGenerator()
    t.run()