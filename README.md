# FtRosa


"Easy way to extract audio features using librosa"

**You can extract every musical audio features featured in [Librosa](https://github.com/librosa/librosa) in one function.**



## What is musical audio features?

- "We use the term 'feature' to refer to **a numerical (scalar, vector, or matrix) or nominal description of the music item** under consideration, typically being the result of a feature extraction process. Feature extractors aim at **transforming the raw data representing the music item into more descriptive representations**, ideally describing musical aspects as perceived by humans, for instance, related to instrumentation, rhythmic structure, melody, or harmony." - Knees, Peter, and Markus Schedl. Music similarity and retrieval: an introduction to audio-and web-based strategies. Vol. 9. Heidelberg: Springer, 2016.


# Features

## Audio features

- Spectogram Features
    
    - Spectrogram features
        - **Spectral centroid**

        - **Spectral bandwidth** of p'th-order

        - **Spectral contrast**

        - **Spectral flatness**

        - **Spectral Roll-off frequency**

        - **Polynomial features**: Coefficients of fitting an nth-order polynomial to the columns of a spectrogram
        
    - Mel-scaled spectrogram
        - **Mel-frequency cepstral coefficients (MFCCs)**
        
  - Chromagram Features

    - **Chromagram STFT**: A chromagram from a waveform or power spectrogram

    - **Chromagram Constant-Q (CQT)**

    - **Chroma Energy Normalized (CENS)**

    - **Tonnetz**: The tonal centroid features (tonnetz)
    
- Energy features

    - **RMS**: Root-mean-square (RMS) value for each frame, either from the audio samples `y` or from a spectrogram `S`

    - **Zero crossing rate:** The zero-crossing rate of an audio time series
    
    - **Onset Strength**
    
- Rhythm features
    
    - **Tempo**: BPM (beats per minute)
    
## Decomposition
- Median-filtering **harmonic-percussive source separation** (HPSS)

## Feature statistics for a feature vector over time

- Mean
- Standard Deviation
- Skewness
- Kurtosis
- Minimum Value
- Maximum Value

---

## Feature Extraction

1. First, an audio is decomposed to a harmonic part and a percussive part. Then we have three audio(signal) time series `y`, `y_harm`, `y_perc`.

2. For each time series (`y`, `y_harm`, `y_perc`), we compute the features introduced above. You can choose to extract features from only `y`, or from `y_harm`, `y_perc` separately. And for chroma features the default option is `y_harm`, and for rhythm featurs the default option is `y_perc`.

3. A feature can be a scalar or a vector. In the case of the feature vectors, we generate mean, standard deviaton, skewness, kurtosis, max, and min value of each feature vector. You can also choose specific stats manually.

---

# Visualization Example

For full description using audible example: [Notebook](https://colab.research.google.com/github/jo-cho/ftrosa/blob/main/Visualization%20Notebook.ipynb)

## Audio

```
- Digital signals information - 
 audio time series length: 661500 
 sampling rate: 22050 
 audio length: 30.0 seconds
 ```
 
 ![image](https://user-images.githubusercontent.com/52461409/218402741-0f0bcdfa-5137-4acc-9f73-5d5106e6faec.png)

## Harmonic-Percussive Source Separation

![image](https://user-images.githubusercontent.com/52461409/218402783-9b0fd49f-a6ba-45bc-9f2d-17a37d91dbe1.png)

## Spectral Features

![image](https://user-images.githubusercontent.com/52461409/218402894-95fedc4c-4537-4552-a510-cbcaa8c7141d.png)

![image](https://user-images.githubusercontent.com/52461409/218402904-56ba6556-289f-408b-a9c7-af177eb96e9b.png)

![image](https://user-images.githubusercontent.com/52461409/218402913-f2fd73aa-964e-45c7-b80d-6f2c3803f761.png)

![image](https://user-images.githubusercontent.com/52461409/218402932-78c23547-1f00-4e17-b886-ddf6b1d97dd7.png)

![image](https://user-images.githubusercontent.com/52461409/218403317-1c48a4ad-2db7-410c-a7ad-c4fdd5f963aa.png)

![image](https://user-images.githubusercontent.com/52461409/218403334-e39c1bc4-9d8c-46db-ac23-00fee87c9bd2.png)


![image](https://user-images.githubusercontent.com/52461409/218403368-db3071d1-f546-4e86-b163-a6b6241dca34.png)
![image](https://user-images.githubusercontent.com/52461409/218403494-7f01c59f-2d44-40eb-9dfb-3558df3c7f80.png)

## Chroma Features

![image](https://user-images.githubusercontent.com/52461409/218403544-83c8eafd-a6a7-4734-8f7b-af578a0c7feb.png)
![image](https://user-images.githubusercontent.com/52461409/218403566-f0644099-742f-422f-be6e-24580b88641f.png)
![image](https://user-images.githubusercontent.com/52461409/218403594-2c6083de-a596-4760-a8c9-6d142b4d54dd.png)
![image](https://user-images.githubusercontent.com/52461409/218403604-58688e21-cbae-48ac-b568-57d13f798682.png)

![image](https://user-images.githubusercontent.com/52461409/218403631-ddc86828-7116-46ce-9fc6-b6937cf61cc9.png)


## Energy Features

![image](https://user-images.githubusercontent.com/52461409/218403661-5d85fd53-806f-47e5-831f-ebf884244f37.png)
![image](https://user-images.githubusercontent.com/52461409/218403670-c64b650f-7d38-420e-96db-17217b605159.png)

## Rhythm Features

![image](https://user-images.githubusercontent.com/52461409/218403708-0b91a419-0324-404a-a324-b713fd228ddc.png)

![image](https://user-images.githubusercontent.com/52461409/218403742-a6cdc103-9839-41e0-9e15-b86b5e7c50ae.png)

![image](https://user-images.githubusercontent.com/52461409/218403761-35600b76-2183-47a4-ac2e-eb57398772c7.png)
![image](https://user-images.githubusercontent.com/52461409/218403785-37f6fdb0-754b-4e14-85ab-76150af012cf.png)

---

# Features Table

```python
from ftrosa import get_all_musical_features

path_audio = "data/example.wav" # path of an audio
song_name = "example audio"
```

with the default option
```python
get_all_musical_features(path_audio, song_name)
 ```
or you can manually choose the parameters
```python
get_all_musical_features(
    path_audio,
    song_name,
    stats=None,
    duration=30,
    start=10,
    from_harm_perc=False,
    chroma_harm=True,
    bpm_perc=True,
    sr=22050,
    hpr_margin=1.5,
    chroma_method_list=['cqt'],
    n_contrast_bands=4,
    n_mfcc=12,
    start_bpms=[60, 120])
 ```
returns


![image](https://user-images.githubusercontent.com/52461409/218405681-13e95fc8-f023-4712-9888-1a6b24f4b8db.png)




--- 
[Librosa citations](https://zenodo.org/record/7618817#.Y-n1tHZByUk)
