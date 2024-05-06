from scipy.fft import rfft, rfftfreq
import numpy as np
from matplotlib import pyplot as plt
import time, cv2
try:
    import ray # <-- For multiprocessing
    from utils import log, remaining_iter_ms
except ModuleNotFoundError:
    import os.path, sys
    sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))
    from utils import log

"""(A) MULTIPROCESSING-READY FUNCTION -----------------------------------------------------"""

""" THREAD 2 - EVENT DETECTION """
@ray.remote
def T2_detect(rate, info, remote=True):
    log('START', format=('purple','bold'))
    #remote = get_caller_name()=='worker'
    def info_get(variable:str, key=''):
        return ray.get(info.get.remote(variable, key)) if remote else info.get(variable, key)
    def info_set(variable:str, newdata, overwrite=False):
        return info.set.remote(variable, newdata, overwrite) if remote else info.set(variable, newdata, overwrite)
        
    Ts = 1/rate
    """ Spectral analysis"""
    det = SpectralAnalysis(variables = info_get('params', 'buff_vars'),
                        window_size=info_get('params', 'n'), 
                        spectrogram_length=info_get('params', 'n_s'),
                        sampling_rate=rate, Alims=(100, 1000), fig_shape=(720,200))
    while True:
        t0 = time.time()
        det.update_spectrogram(wave=info_get('meas_hist'), rate=info_get('params', 'Hz'))
        """Ensure desired sampling rate"""
        if cv2.waitKey(remaining_iter_ms(Ts, t0)) == 27: break  
    cv2.destroyAllWindows()


"""(B) MAIN CLASS -------------------------------------------------------------------------"""
class SpectralAnalysis:
    """ Class to perform online spectral analysis of datastream (wave). """
    def __init__(self, variables, window_size:int, spectrogram_length:int, sampling_rate, 
                Alims=(1e-20, 1e4), fig_shape=(720,100), db=True, normalized=True):
        """
            - variables -> list of strings with the names of the input variables
            - window_size -> buffer length for the measured variable (number of timesteps considered for frequency decomposition).
            - spectrogram_length -> length of the spectrogram time axis.
            - sampling_rate -> In Hz
            - Alims -> Range of signal amplitudes to be observed (saturation lims).
            - fig_shape -> Desired shape for the plots
        """
        self.vars = variables
        self.n = window_size
        self.n_s = spectrogram_length
        self.sr = sampling_rate
        self.m = self.n//2 + 1 # number of bins
        self.maxf = self.m/(self.n/self.sr)
        min_value = -1 if normalized else 1e-20
        self.spectrogram = init_hist(np.ones((self.m, len(variables)))*min_value, self.n_s)
        self.t = time.time()
        self.Alims = np.array(Alims)
        self.db = db
        self.normalized = normalized
        self.figshape = np.array(fig_shape)
        self.fig = plt.figure(figsize=self.figshape/100)
        if not self.normalized: log('WARNING. Plot may not show correctly because data is not being normalized.', format='yellow')

    def update_spectrogram(self, wave, rate, display=True):
        """ Online computation and plotting of data spectrogram """
        """ Shape: (time, freq. bins (m), vars) """
        assert len(wave)==self.n, "Length of input data does not correspond to desired window size."
        self.wr = rate
        if abs(self.sr-self.wr)>8: log("[Warning] sampling time difference too large.", format=('yellow'))
        self.yf, self.y90 = compute_frequency_magnitude(wave, self.Alims, self.db, self.normalized, perc=90)
        self.spectrogram = rollback_hist(self.yf, self.spectrogram)
        if display: self.plot_spectrogram()
        return self.spectrogram
        
    def plot_spectrogram(self, spectrogram=None, figshape=None, line=True, anomalies=None, monit_window=None, threshold=float('inf')):
        """ Real time plotting of spectrogram: freq (y axis), time (x axis), amplitudes (colormap) """
        def draw_f_ax(img, step=2, ticks=[]):
            if not ticks: ticks=np.arange(step,self.maxf,step)
            scale = figshape[1]/self.m
            locs = ((ticks/self.maxf)*self.m*scale).astype(int)
            for f,x in zip(ticks,locs):
                linelen = int(self.n_s*scale) if line else 10
                cv2.line(img, (0,int(x)), (linelen, int(x)), (100,)*3)
                cv2.putText(img, f"{f:4.1f} Hz", (5,int(x)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255),1,2)
            return img
        if figshape is None: figshape = self.figshape
        if spectrogram is None: spectrogram = self.spectrogram
        spectrogram = (spectrogram+1)/2.0 if self.normalized else self.spectrogram
        img = surf_plot(np.clip(spectrogram,0,1), scale=255, figshape=self.figshape, overlap=anomalies, overlap_pad=monit_window, th=threshold)
        img = draw_f_ax(img)
        t1 = time.time()
        # if hasattr(self, 'y90'):
        #     txt = f"f90 = {self.y90:.3f}"
        #     ts,_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        #     cv2.putText(img, txt, (img.shape[1]-ts[0]-4, img.shape[0]-ts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255),1,2)
        cv2.putText(img, f"{1/(t1-self.t):.0f}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0),1,2)
        self.t = t1
        cv2.imshow(f"Spectrogram {self.vars}", img)

    def plot_bode(self):
        """ Real time plotting of bode diagram. freq (x axis), amplitudes (y axis)"""
        plt.clf()
        plt.plot(rfftfreq(self.n, 1/self.sr), self.yf)
        plt.pause(0.001)


"""(C) AUX FUNCTIONS ------------------------------------------------------------------------"""

""" Timeseries handling """
def init_hist(init_value, history_length):
    return np.tile(init_value[None], (history_length,)+(1,)*init_value.ndim)

def rollback_hist(new_value, history): # rolling axis is first one!
    history = np.roll(history, -1, axis=0)
    history[-1] = new_value
    return history

def surf_plot(matrices, scale, figshape=None, colormap=cv2.COLORMAP_VIRIDIS, overlap=None, overlap_pad=None, show=False, th=float('inf')):
    def add_overlap(base, overlap, mask=None):
        return (overlap*mask + base*(1-mask)).astype('uint8')
        
    if not isinstance(matrices, (list, tuple)): matrices=(matrices,)
    viz = []
    # Setup overlap
    if overlap is not None:
        pad = np.array(overlap_pad if overlap_pad is not None else ((0,0),(0,0)))
        if tuple(overlap.shape[:2]+np.sum(pad, axis=1))!=matrices[0].shape[:2]: 
            log(f'Overlap shape {overlap.shape} + pad {list(pad)} does not match base frame shape {matrices[0].shape}. Skipping.', format='yellow')
            overlap=None
        else:   
            score = np.sum(overlap)/overlap.shape[0]
            overlap = np.pad(overlap, [pad[0], pad[1], (0,0)], mode='constant')
            mask = (overlap).transpose(1,0,2).astype('float32') 
            overlap = np.clip(mask*scale, 0, 255).astype('uint8')
            overlap = cv2.applyColorMap(overlap, cv2.COLORMAP_JET) if (colormap and overlap.shape[-1]==1) else matrix
            corners = pad[:,0] * np.array(figshape)/np.array(matrices[0].shape[:2])
            corners = (corners.astype(int), np.array(figshape)-1)
    # Plot each image
    for matrix in matrices:
        """Surface colormap plot of matrix"""
        matrix = np.clip(matrix.transpose(1,0,2)*scale, 0, 255).astype('uint8')
        matrix = cv2.applyColorMap(matrix, colormap) if (colormap and matrix.shape[-1]==1) else matrix
        # Add the overlap
        if overlap is not None:
            matrix = add_overlap(matrix, overlap, mask)
        if figshape is not None:
            matrix = resize(matrix, dim=np.flip(figshape), inter=cv2.INTER_LINEAR)
        if overlap is not None:
            width = 1+int(score>th)
            color = (80,127,255) if score>th else (170,232,238)
            matrix = cv2.rectangle(matrix, corners[0], corners[1], color, width)
            txt, size = f"{score:.3f}", 0.4
            ts,_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, size, width)
            matrix = cv2.putText(matrix, txt, (corners[0][0]-ts[0]-5, corners[1][1]-ts[1]-5), cv2.FONT_HERSHEY_SIMPLEX,size,color,width,2)
        # Draw separation line
        matrix = np.concatenate((matrix, 255*np.ones((1,)+matrix.shape[1:], dtype=np.uint8)), axis=0)
        viz.append(matrix)
    if show:
        cv2.imshow(f"{show}", np.concatenate(viz, axis=0))
        cv2.waitKey()
        cv2.destroyAllWindows()
    return matrix


def compute_frequency_magnitude(wave, lims, to_db = True, normalized=True, perc=None):
    """ Fourier transform and postprocessing """
    yf = np.abs(rfft(wave, axis=0))
    if perc is not None: yfp = np.percentile(yf, 90)
    if to_db:
        lims = 20 * np.log10(lims)
        with np.errstate(divide='ignore', invalid='ignore'):
            yf = 20 * np.log10(yf)
    if normalized:
        yf = 2.0*(yf-lims[0])/(lims[1]-lims[0])-1.0
    if perc is not None: return yf, yfp
    else:                return yf

def resize(image, dim=None, ratio=1, inter = cv2.INTER_LINEAR):
    (h, w) = image.shape[:2]
    if dim is None:
        if ratio!=1: dim = (int(w * ratio), int(h * ratio))
        else:        dim = None
    else:
        dim = tuple(np.flip(dim))
        if dim == (w, h): dim = None
    if dim is not None: return cv2.resize(image, dim, interpolation = inter)
    else              : return image


if __name__ == '__main__':

    det = SpectralAnalysis(variables = ['Tz'],
                        window_size=100, spectrogram_length=250,
                        sampling_rate=20, Alims=(100, 1000), fig_shape=(720,200))
    det.plot_spectrogram()
    cv2.waitKey()
    print(det.spectrogram.shape)



    