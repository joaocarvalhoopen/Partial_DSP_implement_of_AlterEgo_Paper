# Partial DSP implementation of AlterEgo Paper

## Description
This is a simulator of a partial implementation of the MIT paper [AlterEgo: A Personalized Wearable Silent Speech Interface](https://www.media.mit.edu/projects/alterego/overview/) . More precisely, the first DSP processing part, including FastICA. <br>
With this we show how to reconstruct the signals from the sensors in the face of the user wearing the device. We take into account some usual signal interferences. <br>

## Results - See the similarities between the different input signals and the output signals 

### Input signal
![Input signal](./Tuga_AlterEgo_0_input_signal.png?raw=true "Input signal") <br>

### Band pass filter
![Band pass filter](./Tuga_AlterEgo_1_band_pass_filter.png?raw=true "Band pass filter") <br>

### Notch filter
![Notch filter](./Tuga_AlterEgo_2_notch_filter.png?raw=true "Notch filter") <br>

### Reconstructed signal with FastICA
![Reconstructed signal](./Tuga_AlterEgo_3_reconstructed_signal.png?raw=true "Reconstructed signal") <br>

## References
* [MIT Project AlterEgo](https://www.media.mit.edu/projects/alterego/overview/) <br>
* [Paper AlterEgo: A Personalized Wearable Silent Speech Interface](https://www.media.mit.edu/projects/alterego/publications/)

## License
MIT Open Source License

## Have fun!
Best regards, <br>
Joao Nuno Carvalho