import numpy as np
import matplotlib.pyplot as plt

from numpy import sin, cos

import sys
sys.path.append("./")
import omemd
import os

import numpy as np


#----------- create signal ------------------------
def sawtootWave(f, t):
    T = 1/f
    tn = (t%T)/T
    return 2*tn - 1


np.random.seed(12345)
dt = 0.025#0.025#0.025#0.01
t0 = 0#1/512#1174*dt
maxt = 100#100#1e4#90#50#2009*dt#50
antT = int((maxt-t0)/dt) + 1;
pi = np.pi
X = []
underlingTruth = []

for i in range(antT):
    t = dt*i + t0
    p = 2*pi*t
    X.append([0.7*sin(2*p) + 0.3*sawtootWave(6,t), 0.5*np.sin(0.5*p) + 0.5*sawtootWave(6,t)+0.1, 0.7*np.sin(0.5*p) + 0.3*sin(2*p) + 0.001*t-0.05])
X = np.array(X)
# ---------------- settings -------------------------------
antNoiceChannels = 0
antChannels = len(X[0])
antDirections   = 64
windowSize      = int(10/dt)
stepSize        = int(5/dt)
freqDriftRate   = 0.5
newIMF_freqThreshold = 0.6

# ------------------ running algorithm -----------------------
algo = omemd.OMEMD(antChannels, antNoiceChannels, antDirections, windowSize, stepSize, freqDriftRate, dt, newIMF_freqThreshold)
#algo.addData(X);
chunckSize = 100
print("xshape=", X.shape)
for i in range(len(X)//chunckSize):
    print(i*chunckSize)
    algo.addData(X[chunckSize*i:chunckSize*(i+1),:])

imfs = algo.getIMFS()

print(imfs.shape)
print(X.shape)
print(windowSize, stepSize)

# ------------------- plotting data --------------------------
fig, axs = plt.subplots(len(imfs)+1,antChannels, sharex=True, sharey=True, figsize=(28, 16))

t = [0]
tBar = [0]
first = True

for channelI, channel in enumerate(X.T):
    axs[0][channelI].plot(np.arange(len(channel))*dt, channel, label = "signal")#, linestyle=linStyles[dirI], color=colors[dirI], label=name, linewidth = lineWidth[dirI])

for imfI, imf in enumerate(imfs):
    tTmp = np.array([i*dt for i in range(len(imf))])
    if len(tTmp) > len(t):
        t = tTmp
    for channelI, channel in enumerate(imf.T):
        axs[imfI+1][channelI].plot(tTmp, channel, label = "imf"+str(imfI))#, linestyle=linStyles[dirI], color=colors[dirI], label=name, linewidth = lineWidth[dirI])

for ax in axs:
    ax[0].legend(loc='upper left')
    for a in ax:
        a.axvspan(0, t[windowSize-stepSize], color='gray', alpha=0.3)
        a.axvspan(t[-(windowSize-stepSize)], t[-1], color="gray", alpha=0.3)
#for i in range(len(axs[0,:])):
#    axs[0][i].set_title(channelsStr[i])
plt.tight_layout()
plt.xlim(t[0], t[-1])
#plt.xlim(4.5, 5.5)
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()








