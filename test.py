from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne

raw = read_raw_edf("C:\\Users\\Administrator\\Downloads\\mesa-sleep-0001.edf", preload=False)

raw.plot_psd(fmax=50)
