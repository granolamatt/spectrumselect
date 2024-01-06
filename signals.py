import pygame
import numpy as np
import matplotlib.pyplot as plt

class burst_object:
    def __init__(self, length):
        self.length = length
        self.fmat = self.rrc_filter((np.arange(length) - 32)/2) / length
        self.fmat[::2] *= -1
        self.fmat = np.fft.fft(self.fmat, n=self.length)

    def makeBurst(self, snr=10):
        data = np.random.randint(0,2,self.length)
        ssyms = data.astype(np.float32)
        rb = ssyms[::2] + 1j*ssyms[1::2]
        rb = 2.0 * (rb - (0.5+0.5j))
        syms = np.zeros(self.length, dtype=np.complex64)
        syms[::2] = rb
        self.fmat /= 2*20*np.log10(self.length) / (snr - 3)
        fsyms = np.fft.fft(syms, n=self.length)
        return self.fmat*fsyms

    def rrc_filter(self, tmat, beta=0.35/2):
        t = tmat - 1e-9
        sin_arg = np.sin(np.pi * t * (1.0 - beta))
        cos_arg = 4.0 * beta * t * np.cos(np.pi * t * (1 + beta))
        denom = np.pi * t * (1.0 - 16.0 * (beta * t) * (beta * t))
        return (sin_arg + cos_arg) / denom


class SignalSelection:
    def __init__(self,startpos, horizontal_location):
        self.startpos = horizontal_location
        self.endpos = horizontal_location
        self.endit = False
    
    def render(self, screen, step):
        self.startpos -= step
        if self.endit:
            self.endpos -= step
        horsize = (self.endpos - self.startpos)
        rect = pygame.Rect(self.startpos, 0, horsize, 600)
        pygame.draw.rect(screen, (255, 255, 255), rect)
        if self.endpos < 0:
            return True
        return False
    
    def end(self):
        self.endit = True

    def calcScore(self, bins):
        timesig = np.fft.ifft(bins,n=len(bins)*2)
        start = len(timesig)//32
        freqsig = np.fft.fft(timesig*np.conj(timesig))[start:len(timesig)//2]
        baud = len(timesig)/(np.argmax(np.abs(freqsig)) + start) / 2
        if baud > 3 or baud < 1:
            return -1
        
        score = 10 - np.abs(1.3 - baud)
        print("Score is", score)
        

        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.plot(np.abs(bins))
        # plt.subplot(2,1,2)
        # plt.plot(np.abs(freqsig))
        # plt.title(str(baud))
        # plt.show()
        return score

    def __eq__(self,other):
        return self.startpos == other.startpos and self.endpos == other.endpos


class SignalGenerator:
    def __init__(self, screen):
        self.screen = screen
        self.N = 1024*1024
        self.sigbuffer = np.random.normal(0,1/self.N,self.N) + 1j*np.random.normal(0,1/self.N,self.N)
        self.pos = self.N//2
        self.selections = []
        self.stepsize = 128
        self.avesize = 64
        self.score = 0
        self.deaths = 0

    def setScore(self):
        font = pygame.font.Font('freesansbold.ttf', 32)
        green = (0,255,0)
        blue = (0,0,255)
        text = font.render(f'score {np.round(self.score,decimals=1)} lives {3 - self.deaths}', True, green, blue)
        textRect = text.get_rect()
        textRect.center = (800,100)
        self.screen.blit(text, textRect)
        # textRect.center = (X // 2, Y // 2)

    def makeSpectrum(self):
        self.sigbuffer[:self.N//2] = self.sigbuffer[self.N//2:]
        self.sigbuffer[self.N//2:] = np.random.normal(0,1/self.N,self.N//2) + 1j*np.random.normal(0,1/self.N,self.N//2)
        # self.sigbuffer = np.zeros(N, dtype=np.complex64)
        num = np.random.randint(1,20)
        print("Making",num,"signals")
        for cnt in range(num):
            offset = np.random.randint(20,self.N//4) * 2
            siglen = np.random.randint(20,self.N//64) * 2
            bo = burst_object(siglen)
            samps = bo.makeBurst()
            if self.N//2+len(samps)+offset < len(self.sigbuffer):
                self.sigbuffer[self.N//2 + offset:self.N//2+len(samps)+offset] += samps
            
    def setStart(self):
        self.selections.append(SignalSelection(self.pos,self.agent_horizontal_location))

    def setEnd(self):
        for sel in self.selections:
            sel.end()

    def aveData(self, samps):
        mag = samps.reshape((-1,self.avesize))
        mag = -40*np.log10(np.average(mag, axis=1)) + 100
        # print("max",np.max(mag),"min",np.min(mag))
        return mag

    def reset(self):
        self.sigbuffer = np.random.normal(0,1/self.N,self.N) + 1j*np.random.normal(0,1/self.N,self.N)
        self.makeSpectrum()
        self.pos = self.N//2
        self.state = self.sigbuffer[self.pos:self.pos+self.avesize*1024]

    def render(self):
        stepmult = self.stepsize//self.avesize
        rem_list = []
        for sel in self.selections:
            rem = sel.render(self.screen,stepmult)
            if rem:
                rem_list.append(sel)
        for sel in rem_list:
            ss = int(self.pos + sel.startpos*self.avesize)
            ee = int(self.pos + sel.endpos*self.avesize)
            ns = sel.calcScore(self.sigbuffer[ss:ee])
            if ns < 0:
                self.deaths += 1
            else:
                self.score += ns
            self.selections.remove(sel)
        mag = self.aveData(np.abs(self.state))
        for x in range(1,len(mag)):
            start = mag[x-1]
            stop = mag[x]
            pygame.draw.line(self.screen, (255,0,0), (x,start), (x,stop))
        self.setScore()
        

    def step(self):
        self.pos += self.stepsize
        self.state = self.sigbuffer[self.pos:self.pos+self.avesize*1024]
        if self.pos > self.N//2:
            self.pos -= self.N//2
            self.makeSpectrum()