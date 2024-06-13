
import numpy as np
from math import pi,sqrt,sin,cos, log2
from scipy.fft import fft, fftfreq
import sys
sys.path.append("./MEMD-Python--master")
from MEMD_all import local_peaks, hamm, nth_prime, memd

import time

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if phi < 0:
        phi = 2*np.pi + phi
    return(rho, phi)
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)
def getCenterOfMassFFT(x, antNotNoiceChannels):
    if len(x.shape) != 2:
        print("Warning: getCenterOfMass should have 2 dimentions, x[t][c]", x.shape)
        x = np.array([x]).T
        print("new X shape", x.shape)
    rf = np.zeros(len(fft(x[:,0]))//2)
    for i in range(antNotNoiceChannels):
        rf = rf + np.power(np.abs(fft(x[:,i])[:len(rf)]), 2)
    deg = 2*np.pi*np.arange(1.*len(rf))/(len(rf)-1)
    xf, yf = pol2cart(rf, deg)
    #yf = rf * np.sin(deg)
    #print("xf",xf)
    #print("yf",yf)
    centerX = sum(rf*xf)/(2*sum(rf))
    centerY = sum(rf*yf)/(2*sum(rf))
    centerR, centerDeg = cart2pol(centerX, centerY)
    return centerDeg, centerR

def get_sorted_indices(lst):
    # Enumerate the list to keep track of original indices
    enumerated_list = list(enumerate(lst))

    # Sort the enumerated list based on the values
    sorted_enumerated_list = sorted(enumerated_list, key=lambda x: x[1])

    # Extract the new indices
    sorted_indices = [index for index, value in sorted_enumerated_list]
    
    return sorted_indices


class OMEMD:
    printing = False
    antChannels = 0
    antNoiceChannels = 0
    antDirections = 0
    windowLength = 0
    stepSize = 0
    freqTolerense = []
    tau = 4
    notFinnishedStart = 0
    windowFunc = []
    directions = []
    originalSignal = []
    IMFS = []
    imfMainFreq = []
    IMFS_start = []
    IMFS_itCount = []
    freqDriftRate = 0.3
    newIMFthresh = 1#0.2*np.pi
    newIMFminR = 1e-1
    rest = []
    zeros = np.array([])
    degToFreq = 0

    def _imfSequence(self, localIMFS):
        #centMassFreq, centMassR = np.array([getCenterOfMassFFT(imf, self.antChannels - self.antNoiceChannels) for imf in localIMFS])
        centMass = np.array([getCenterOfMassFFT(imf, self.antChannels - self.antNoiceChannels) for imf in localIMFS])
        centMassFreq = np.array([log2(m) for m in centMass[:,0]])
        centMassR = centMass[:,1]
        if len(self.imfMainFreq) == 0:
            sortI = get_sorted_indices(centMassFreq)[::-1]
            sequence = sortI.copy()
            antGlobalIMF = len(sortI)
            for i in range(1, len(sortI)):
                if (i + 1 == len(sortI) or abs(centMassFreq[sortI[i]] - centMassFreq[sortI[i-1]]) < abs(centMassFreq[sortI[i]] - centMassFreq[sortI[i+1]])) and abs(centMassFreq[sortI[i]] - centMassFreq[sortI[i-1]]) < self.newIMFthresh: # if the closest center of mass have higher frequency then this senter of mass, and if the difference is lower then the threshold
                        for j in range(len(sequence)):
                            if sequence[j] > sequence[i]:
                                sequence[j] = sequence[j] - 1
                        sequence[i] = sequence[i] - 1
                        antGlobalIMF = antGlobalIMF -1

            self.IMFS = []
            self.imfMainFreq = []
            for i in range(0, antGlobalIMF):
                self.IMFS.append(np.zeros(localIMFS[0].shape,dtype=float))
                self.IMFS_start.append(0)
                self.IMFS_itCount.append(1)
                self.imfMainFreq.append(0)
                antLocalInGlobal = 0
                for j in range(len(sequence)):
                    if sequence[j] == i:
                        self.imfMainFreq[-1] = self.imfMainFreq[-1] + centMassFreq[sortI[j]]
                        antLocalInGlobal = antLocalInGlobal + 1
                self.imfMainFreq[-1] = self.imfMainFreq[-1] / antLocalInGlobal

            sequence = np.array(sequence)           
            return sequence
        else:
            sequence = np.zeros(len(localIMFS), dtype = float)
            for i in range(len(localIMFS)):
                for j in range(len(self.imfMainFreq)):
                    if abs(self.imfMainFreq[int(sequence[i])] - centMassFreq[i]) > abs(self.imfMainFreq[j] - centMassFreq[i]):
                        sequence[i] = j
            #print("centMassR    = ", centMassR, "\tcentMassFreq = ", self.degToFreq*(2**(np.array(centMassFreq, dtype=float))), "\tglobalMainFreq = ", self.degToFreq*(2**(np.array(self.imfMainFreq, dtype=float))), "\tsequence = ", sequence) 
            for i in range(len(centMassFreq)):
                if abs(centMassFreq[i] - self.imfMainFreq[int(sequence[i])]) < self.newIMFthresh or centMassR[i] < self.newIMFminR:
                    self.IMFS_itCount[int(sequence[i])] = self.IMFS_itCount[int(sequence[i])] + 1
                    if self.freqDriftRate == "Avreage":
                        self.imfMainFreq[int(sequence[i])] = (1 - (1/self.IMFS_itCount[int(sequence[i])])) * self.imfMainFreq[int(sequence[i])] + (1/self.IMFS_itCount[int(sequence[i])]) * centMassFreq[i]
                    else:
                        self.imfMainFreq[int(sequence[i])] = (1 - self.freqDriftRate) * self.imfMainFreq[int(sequence[i])] + self.freqDriftRate * centMassFreq[i]
                else:
                    seqOld = sequence[i]
                    if centMassFreq[i] > self.imfMainFreq[int(sequence[i])]:
                        sequence[i] = sequence[i]-0.5
                        if seqOld != 0:
                            self.imfMainFreq  = self.imfMainFreq[:int(seqOld)] + [centMassFreq[i]] + self.imfMainFreq[int(seqOld):]
                            self.IMFS_itCount = self.IMFS_itCount[:int(seqOld)]+ [1] + self.IMFS_itCount[int(seqOld):]
                        else:
                            self.imfMainFreq =                                   [centMassFreq[i]] + self.imfMainFreq
                            self.IMFS_itCount =                                  [1] + self.IMFS_itCount[int(seqOld):]

                    else:
                        sequence[i] = sequence[i]+0.5
                        self.imfMainFreq  = self.imfMainFreq[:int(seqOld+1)] + [centMassFreq[i]] + self.imfMainFreq[int(seqOld+1):]
                        self.IMFS_itCount = self.IMFS_itCount[:int(seqOld+1)]+ [1] + self.IMFS_itCount[int(seqOld+1):]

                    for j in range(i+1,len(centMassFreq)):
                        if sequence[j] > seqOld or (sequence[j] == seqOld and centMassFreq[j] < centMassFreq[i]):
                            sequence[j] =sequence[j] + 1

            #print("sequence", sequence)
            return sequence

            

    def addData(self, data):
        try:
            if len(self.originalSignal) == 0:
                self.originalSignal = data
            else:
                self.originalSignal = np.concatenate((self.originalSignal, data))
            start_time = time.time()
            while (self.notFinnishedStart + self.windowLength < len(self.originalSignal)):
                #print("start", self.notFinnishedStart, self.windowLength, len(self.originalSignal))#, "\n imf_Start=", self.IMFS_start)
                subVec = self.originalSignal[self.notFinnishedStart:self.notFinnishedStart+self.windowLength]
                #print("shapeSubvec", subVec.shape)
                localIMFS = np.transpose(memd(subVec, self.antDirections), (0,2,1))
                localIMFsequence = self._imfSequence(localIMFS[:-1])
                for i in range(len(localIMFS[:-1])): 
                    #self.plot(localIMFS[i], "localIMF_"+str(i)+"_"+str(self.notFinnishedStart)+"_"+str(localIMFsequence[i]), self.notFinnishedStart, "local_")
                    if (localIMFsequence[i]%1 == 0.5) :
                        #print("------- is 0.5 --", self.notFinnishedStart)
                        #if self.notFinnishedStart == 0 and localIMFsequence[i] == -0.5:
                        #        self.IMFS = [self.windowFunction[:, np.newaxis] * localIMFS[i]]
                        #else:
                        #print(np.concatenate((np.zeros((self.notFinnishedStart, self.antChannels), dtype = float), localIMFS[i]), axis=0))
                        #print(len(self.IMFS), np.concatenate((np.zeros((self.notFinnishedStart, self.antChannels), dtype = float), localIMFS[i]), axis=0).shape)
                        imf   = self.windowFunction[:, np.newaxis] * localIMFS[i]
                        index = int(localIMFsequence[i]+0.5)
                        #self.IMFS = np.concatenate((self.IMFS[:index], [imf], self.IMFS[index:]),axis=0)
                        self.IMFS       = self.IMFS[:index]       + [imf] +                    self.IMFS[index:]
                        self.IMFS_start = self.IMFS_start[:index] + [self.notFinnishedStart] + self.IMFS_start[index:]
                        #for j in range(len(localIMFsequence[i+1:])):
                        #    if (localIMFsequence[j] > localIMFsequence[i]):
                        #        localIMFsequence[j] = localIMFsequence[j] + 1
                    else:
                        pad_width = [
                            (0, max(0, self.notFinnishedStart + self.windowLength - self.IMFS[int(localIMFsequence[i])].shape[0] - self.IMFS_start[int(localIMFsequence[i])] )),  # Pad rows
                            (0, 0)  # No padding on columns
                        ]
                        fromT = self.notFinnishedStart - self.IMFS_start[int(localIMFsequence[i])]
                        toT   = self.notFinnishedStart - self.IMFS_start[int(localIMFsequence[i])] + self.windowLength
                        self.IMFS[int(localIMFsequence[i])] = np.pad(self.IMFS[int(localIMFsequence[i])], pad_width, mode='constant', constant_values=0)
                        self.IMFS[int(localIMFsequence[i])][fromT:toT] = self.IMFS[int(localIMFsequence[i])][fromT:toT] + self.windowFunction[:, np.newaxis] * localIMFS[i]
                self.notFinnishedStart = self.notFinnishedStart + self.stepSize
                #print("global IMFs target freqs = ", self.degToFreq*(2**(np.array(self.imfMainFreq, dtype=float))))
                #if (self.notFinnishedStart > 1900):
                #    break
            #self._appendRest(0, data)
            print("--- %s seconds online MEMD---" % (time.time() - start_time))
        
        except:
             # Print the exception
            import traceback
            traceback.print_exc()
            print("going to save the IMFS")
            self.test()
            exit()
        #self.test()
        
    def __init__(self, antChannel, antNoiceChannels, antDirection, windowLength, stepSize, frequencyDriftRate = 0.3, dt = 1, freqThreshold = -1): # frequency Drift rate has to be < 1
        self.antChannels = antChannel
        self.antNoiceChannels = antNoiceChannels
        self.antDirections = antDirection
        self.windowLength = windowLength
        self.stepSize = stepSize
        self.freqDriftRate = frequencyDriftRate
        self.degToFreq = (fftfreq(windowLength, dt)[:windowLength//2][-1])/(2*np.pi)
        if freqThreshold != -1:
            self.newIMFthresh = freqThreshold
       #     print("threshold, old", self.newIMFthresh, self.newIMFthresh*self.degToFreq, "new", freqThreshold/self.degToFreq, freqThreshold)
       #     self.newIMFthresh = freqThreshold/self.degToFreq

        notNormalicedWindowFunct = np.array([np.exp(-((2*self.tau*(t-windowLength/2)/windowLength)**2)/2) - np.exp(-((self.tau)**2)/2) for t in range(windowLength)])
        sumWindowFunction = np.zeros(3*windowLength+stepSize, float)
        for T in range(0, 2*self.windowLength+1, stepSize):
            for t in range(T, T+windowLength):
                sumWindowFunction[t] = sumWindowFunction[t] + notNormalicedWindowFunct[t-T] 
        self.windowFunction = np.array([notNormalicedWindowFunct[t]/sumWindowFunction[self.windowLength+t] for t in range(self.windowLength)], float)


        self.zeros = np.array([[0 for _ in range(self.antChannels)]], dtype=float)

        #--------get direction vectors---------
        
        # Initializations for Hammersley function
        base = []
        base.append(-self.antDirections)

        if self.antChannels==3:
            base.append(2)
            seq = np.zeros((self.antDirections,self.antChannels-1))
            for it in range(0,self.antChannels-1):
                seq[:,it] = hamm(self.antDirections,base[it])
        else:
            #Prime numbers for Hammersley sequence
            prm = nth_prime(self.antChannels-1)
            for itr in range(1,self.antChannels):
                base.append(prm[itr-1])
            seq = np.zeros((self.antDirections,self.antChannels))
            for it in range(0,self.antChannels):
                seq[:,it] = hamm(self.antDirections,base[it])

        dir_vec = np.zeros((self.antChannels, 1))
        for it in range(0,self.antDirections):
            #print(tmpIMFCount, tmpShiftCount, it)
            if self.antChannels !=3:     # Multivariate signal (for self.antChannels ~=3) with hammersley sequence
                #Linear normalisation of hammersley sequence in the range of -1.00 - 1.00
                b=2*seq[it,:]-1 
                
                # Find angles corresponding to tNever Too Latehe normalised sequence
                tht = np.arctan2(np.sqrt(np.flipud(np.cumsum(b[:0:-1]**2)))\
                                 ,b[:self.antChannels-1]).transpose()
                
                # Find coordinates of unit direction vectors on n-sphere
                dir_vec[:,0] = np.cumprod(np.concatenate(([1],np.sin(tht))))
                dir_vec[:self.antChannels-1,0] =  np.cos(tht)*dir_vec[:self.antChannels-1,0]
                
                self.directions.append(dir_vec[:,0].copy())
                
            else:     # Trivariate signal with hammersley sequence
                # Linear normalisation of hammersley sequence in the range of -1.0 - 1.0
                tt = 2*seq[it,0]-1
                if tt>1:
                    tt=1
                elif tt<-1:
                    tt=-1         
                
                # Normalize angle from 0 - 2*pi
                phirad = seq[it,1]*2*pi
                st = sqrt(1.0-tt*tt)

                dir_vec[0]=st*cos(phirad)
                dir_vec[1]=st*sin(phirad)
                dir_vec[2]=tt
                
                self.directions.append(dir_vec[:,0].copy())
                

        self.directions = np.array(self.directions)    

    def getIMFS(self):
        ret = self.IMFS.copy()
        size = min(len(self.IMFS_start), len(self.IMFS))
        print("shape", self.IMFS[0].shape, "start", self.IMFS_start)
        for i in range(size):
            pad_width = [
                (self.IMFS_start[i], max(0, self.notFinnishedStart - self.IMFS[i].shape[0] - self.IMFS_start[i] )),  # Pad rows
                (0, 0)  # No padding on columns
            ]
            print(ret[i].shape)
            ret[i] = np.pad(ret[i], pad_width)
            print("shape getIMFS", i, ret[i].shape)
        return np.array(ret)


        
         


