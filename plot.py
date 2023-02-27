import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

wave_read = wave.open('a.wav', 'rb')
dtype = f'int{wave_read.getsampwidth()*8}'
frame_total = wave_read.getnframes()
channels = wave_read.getnchannels()
framerate = wave_read.getframerate()
interval = 100 # t
nyquist = framerate//interval//2+1
amplitude = lambda x: np.abs(np.fft.fft(x))[:nyquist]
linspace = np.linspace(0, framerate//interval//2, nyquist) * interval

# padding
padding = np.zeros((framerate-(frame_total%framerate), channels), dtype=dtype)

# read all data and reshape
audio_data = np.frombuffer(wave_read.readframes(frame_total), dtype=dtype).reshape(-1, channels)
audio_data = np.append(audio_data, padding, axis=0).reshape(-1, framerate//interval, channels)
audio_data = audio_data.astype(np.float64)

left_data = audio_data[:, :, 0]
right_data = audio_data[:, :, 1]
left_amp = np.apply_along_axis(amplitude, 1, left_data)
right_amp = np.apply_along_axis(amplitude, 1, right_data)

fig = plt.figure(figsize=(19.2, 10.8))
plt.xlabel('Frequency', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.xticks(linspace)
plt.xlim(20, 2500)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 17

ims = []
for amp in np.dstack((left_amp, right_amp)):
    left_plot = plt.plot(linspace, amp[:, 0], color='b')
    right_plot = plt.plot(linspace, amp[:, 1], color='r')

    ims.append(left_plot+right_plot)

ani = animation.ArtistAnimation(fig, ims, interval=1000//interval)
ani.save('anim.mp4', writer="ffmpeg", progress_callback=lambda i, n: print(f'{i}'))
