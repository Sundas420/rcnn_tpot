from Main.PyEMD import EMD
import os
from pyAudioAnalysis import ShortTermFeatures
import scipy.io
from scipy.ndimage import gaussian_filter
import numpy as np
import librosa
from scipy import signal, stats
from scipy.fft import fft
import pywt
from Main import AMS
def preprocessing():
    def EMD_(X):
        f = []
        emd = EMD()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                feat1 = emd(X[i][j])
                f.append(np.sum(feat1))
        feat = np.mean(f)
        return feat
    def dwt_(x):
        coeffs2 = pywt.dwt2(x, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2
        return np.mean(LL)

    def log_bandpower(data, window_sec=None, relative=False):
        lbp = []
        # for i in range(len(data)):
        sf = 100
        band = [0.5, 4]
        """Compute the average power of the signal x in a specific frequency band.
    
        Parameters
        ----------
        data : 1d-array
            Input signal in the time-domain.
        sf : float
            Sampling frequency of the data.
        band : list
            Lower and upper frequencies of the band of interest.
        window_sec : float
            Length of each window in seconds.
            If None, window_sec = (1 / min(band)) * 2
        relative : boolean
            If True, return the relative power (= divided by the total power of the signal).
            If False (default), return the absolute power.
    
        Return
        ------
        bp : float
            Absolute or relative band power.
        """
        from scipy.signal import welch
        from scipy.integrate import simps
        band = np.asarray(band)
        low, high = band

        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        # Compute the modified periodogram (Welch)
        freqs, psd = welch(data, sf, nperseg=nperseg)

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]

        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using Simpson's rule.
        bp = simps(psd[idx_band], dx=freq_res)

        if relative:
            bp /= simps(psd, dx=freq_res)
        lbp.append(np.log(bp))
        return np.array(lbp),bp

    feat_all, lab = [],[]
    path = r'E:\Deepika\Paper_works\Sundas Almas (237915) - Paper 1 (Class I)\237915_sa\Data'
    list = os.listdir(path)
    for i in range(len(list)):
        print(i)
        path2 = path + '/' + list[i]
        list2 = os.listdir(path2)
        for j in range(len(list2)):
            path3 = path2 + '/' + list2[j]
            input = scipy.io.loadmat(path3)
            data = input['data_struct'][0,0]['CrossM']
            data = data.astype('float')
            age = input['data_struct'][0,0]['age']
            sr = input['data_struct'][0, 0]['srate']
            # for k in range(data.shape[0]):
            #     for l in range(data.shape[1]):
            #         data_ = data[k, l, :]
    #--------------------------------------Preprocessing--------------------------------------

            preprocessed_data = gaussian_filter(data,1)

    #--------------------------------------spectral-based----------------------------------------

            fs_ = 100
            x = np.resize(preprocessed_data,(128,))
            ams_feature = AMS.ams_extractor(x, fs_, int(fs_ * 0.02), int(fs_ * 0.01), 128, 8, 1, 'hanning', 'v1')
            ams_feature = AMS.uniformize_matrix(ams_feature)
            aa = np.nan_to_num(ams_feature)
            f1 = np.mean(aa)                                                        #AMS

    #------------------------------------ frequency-based ----------------------------------------

            sr_ = 10000
            f2 = librosa.onset.onset_strength(y=np.float32(preprocessed_data), sr=sr,
                                                     aggregate=np.median,
                                                     fmax=16000, n_mels=256)      #Spectural flux
            f2= np.mean(f2)

            f3 = librosa.feature.tonnetz(y=np.float32(preprocessed_data), sr=sr_)  #tonal power ratio
            f3= np.mean(f3)

            if i>2:
                f4 = librosa.feature.spectral_centroid(y=np.float32(preprocessed_data), sr=100)  # Spectral Centroid
            else:
                f4 = librosa.feature.spectral_centroid(y=np.float32(preprocessed_data), sr=sr)
            f4= np.mean(f4)

            z = np.resize(preprocessed_data, (1000,))
            F, f_names = ShortTermFeatures.feature_extraction(z, sr_, 0.050*sr_, 0.025*sr_) # Spectral_spread
            f5 = F[4 ,:]
            f5 = np.mean(f5)

            fs = 1000.0
            (f, f6) = scipy.signal.periodogram(preprocessed_data, fs, scaling='density') #power spectral density
            f6 = np.mean(f6)

            f7,_ = log_bandpower(x)  #logarithmic band power
            f7 = np.mean(f7)

    #----------------------------------------- statistical features --------------------------------------------

            f8 = np.mean(preprocessed_data)                                     # mean
            f9 = np.median(preprocessed_data)                                   # median
            f10 = np.std(preprocessed_data)                                     # standard deviation
            f11 = np.mean(stats.kurtosis(preprocessed_data))                    # kurtosis
            f12 = np.mean(stats.skew(preprocessed_data))                        # skew
            f13 = np.mean(stats.entropy(preprocessed_data))                     # entropy
            f13 = np.nan_to_num(f13,neginf=88)

    #-------------------------------------- frequency domain features ------------------------------------------

            f14 = fft(preprocessed_data)                                        # fourier transform
            f14 = np.mean(f14.astype(float))
            bp_,f15 = log_bandpower(x)                                          # band power
            f15 = np.mean(f15)
            f16 = EMD_(preprocessed_data)                                       # EMD
            f17 = dwt_(preprocessed_data)                                       # DWT

            feat = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16]
            feat_all.append(feat)
            lab.append(age[0][0])
            np.savetxt("Feature2.csv", feat_all, delimiter=',', fmt="%s")
            np.savetxt("Label2.csv", lab, delimiter=',', fmt="%s")

preprocessing()
