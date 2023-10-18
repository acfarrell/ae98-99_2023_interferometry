from common_imports import *
from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.feature import canny
from skimage.restoration import unwrap_phase
from spurs_unwrap import unwrap
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline

def get_phase(data, refs, bw_x=25, bw_y=None, dc = 5, spurs_unwrap=False):
    '''
    Get phase map from interferogram for a given data image and list of reference images
    
    Returns the phase map and a boolean mask that excludes pixels within one period 
    of the edge of the image, since these regions are subject to poor behavior due to 
    discontinuities in the FFT. The mask is used primarily for plotting convenience.
    '''
    h, w = data.shape
    if bw_y is None:
        bw_y = bw_x
    #window = 1.#* np.hanning(w)[None,:]#np.hanning(h)[:,None]
    edge_mask = np.ones_like(data) 

    edge_width = 100#bw_x * 2


    edge_mask[:edge_width, :]= (sin( pi / edge_width/2 * abs(np.arange(edge_width)))**2)[:,None]
    edge_mask[-edge_width:, :]= (sin( pi / edge_width/2 * abs(edge_width - np.arange(edge_width)))**2)[:,None]
    edge_mask[:, :edge_width]*= (sin( pi / edge_width/2 * abs(np.arange(edge_width)))**2)[None,:]
    edge_mask[:,-edge_width:]*= (sin( pi / edge_width/2 * abs(edge_width - np.arange(edge_width)))**2)[None, :]

    ref_ffts = []
    kcs_x = []
    kcs_y = []
    for ref in refs:
        ref_fft = fftshift(fft2((ref) * edge_mask))
        # remove dc offset so it doesn't interfere with the carrier frequency
        ref_fft[h//2-dc :h//2+dc+1, w//2-dc:w//2+dc+1] = 0
        
        # Determine carrier frequency
        k_cy, k_cx = np.unravel_index(np.argmax((abs((ref_fft[:,:w//2])))), ref_fft[:,:w//2].shape)
        kcs_x.append(k_cx)
        kcs_y.append(k_cy)

        ref_ffts.append(ref_fft)
    k_cx = int(np.mean(kcs_x))
    k_cy = int(np.mean(kcs_y))
    Y, X = np.indices((h,w))

    # make a mask that excludes the edges of the images within one carrier frequency 
    # used in plotting   
    valid = np.zeros((h,w), dtype=bool)
    edge_x = abs(w//2 - k_cx)
    edge_y = abs(h//2 - k_cy)
    valid[edge_y:-edge_y, edge_x:-edge_x] = True
    
    # Make a gaussian filter around the carrier frequency
    filter = exp(-((X - k_cx)**2/bw_x**2 + (Y-k_cy)**2/bw_y**2)/2)

    data_fft = fftshift(fft2(data * edge_mask))

    data_fft[h//2-dc :h//2+dc+1, w//2-dc:w//2+dc+1] = 0

    # Shift the carrier frequency to the origin in frequency space
    shifted_fft = (np.roll(np.roll(filter * data_fft, -k_cx, axis=1),-k_cy, axis=0))

    
    
    # Invert the FFT and retreive the phase
    data_pha = (-log(ifft2(shifted_fft))/1j).real
    ref_phas = []
    for ref_fft in ref_ffts:
        shifted_ref_fft = (np.roll(np.roll(filter * ref_fft, -k_cx, axis=1),-k_cy, axis=0))
        ref_phas.append((-log(ifft2(shifted_ref_fft))/1j).real)
        
    # unwrap the phase and remove the averaged background phase
    if not spurs_unwrap:
        return unwrap_phase(data_pha)- np.mean([unwrap_phase(ref_pha) for ref_pha in ref_phas], axis=0), valid
    return unwrap(data_pha, p=.5) - np.mean([unwrap(ref_pha, p=.5) for ref_pha in ref_phas], axis=0), valid

def get_density(pha, M, method='basex', center=None, reg = None, symmetric = True):
    '''
    Calculate the plasma density from a given phase map (in radians)
    
    Abel inversion methods are:
        'basex':    basis set expansion, fast but can produce negative densities
        'daun':     Daun algorithm, regularized so no negative densities, but slow
    '''
    assert method in ['basex', 'daun'], "Supported Abel inversion methods are \'basex\' and \'daun\'"
    h, w = pha.shape
    if_pix_size = 5.86 * um / M     # interferometer pixel size 
    lambda_l = 800 * u.nm           # probe wavelength
    w_0 = 2 * pi * (c / lambda_l).to(1/s)                   # probe frequency
    n_c = m_e * eps0 * (2 * np.pi * c / e / lambda_l)**2    # probe critical density

    alpha = (c/w_0/if_pix_size).to(dl)
    
    if symmetric:
        sym_ax = 0
    else:
        sym_ax = None
    
    if center is None:
        center = detect_center(pha)[0]
    
    if method == 'daun':
        inverted = abel.transform.Transform(pha.T, 
                                            origin = (w/2., center), 
                                            symmetry_axis=sym_ax, 
                                            direction='inverse', 
                                            method='daun', 
                                            transform_options=dict({ 'reg':'nonneg', }),
                                            center_options=dict({'crop':'maintain_data'})).transform.T

    elif method == 'basex':
        if reg is None:
            reg = 300
        inverted = abel.transform.Transform(pha.T, 
                                            origin = (w/2., center), 
                                            symmetry_axis=sym_ax, 
                                            direction='inverse', 
                                            method='basex', 
                                            transform_options=dict({ 'reg':reg, 'correction':True}),
                                            center_options=dict({'crop':'maintain_data'})).transform.T
    
    # calculate the density from the Abel inverted phase
    density = (n_c * (1-(1 - alpha * inverted)**2)).to(1/cm**3)

    # correct for PyAbel's image centering, so output is the same size as the input phase
    if w % 2 == 0:
        # centering will add an empty column on the end if the input has even width
        density = density[:,:-1]
    
    # roll the image back so the center is in the same place as the input image, crop excess
    density = density[density.shape[0]//2 - center:][:h]
    if density.shape[0] == h - 1:
        density = np.pad(density.to_value(1/cm**3), ((0,1),(0,0)), ) * 1/cm**3
    
    if method =='basex':
        # remove negative densities returned by basex inversion
        density[density.value < 0] = 0 * density.unit
        
    return density

def gaussian(x, A, x0, sigma, offset):
    return A * np.exp(- (x - x0)**2 / (2 * sigma)**2) + offset

def detect_center(phase, center_idx = None, plot = False):
    h, w = phase.shape
    y_pix = np.arange(h)
    
    if center_idx is None:
        # Find columns of the image where there is significant phase shift
        plasma_region = np.any(phase >  2 * np.nanstd(phase) +  np.nanmean(phase), axis=0)
        
        # exclude the edges where phase reconstruction can misbehave
        plasma_region[:100] = False
        plasma_region[w-100:] = False
        
        # average along x-axis
        phase_yproj = np.mean(phase[:,plasma_region], axis = 1)
        
        # fit a gaussian
        popt, pcov = curve_fit(gaussian, y_pix,phase_yproj , p0=[phase_yproj.max(), np.argmax(phase_yproj), 10,phase_yproj.min()])

        center = popt[1]
        sigma = popt[2]
        offset = popt[3]
        print(f'Center = {center:.1f}, Sigma = {sigma:.1f}, Offset = {offset:.2f}')
        center = int(np.round(center))
    else:
        center = center_idx
        
    if plot:
        fig, ax = plt.subplots()
        ax.plot(phase_yproj, label='Data')
        ax.plot(gaussian(y_pix, *popt), '--', label='Gaussian Fit')
        ax.axvline(center, ls='--', color='k', lw=0.75)
        ax.set_xlabel('Transverse Axis (pixels)')
        ax.set_ylabel('Mean $\\varphi$ (radians)')
        ax.set_title('Detected Plasma Center')
        ax.legend()
        plt.show()
        
    return center, offset

def equalize(data,sig = 7, dist=5):
    data_eq = cv2.GaussianBlur(data, (sig,sig),0)
    env_lo = np.zeros(data_eq.shape)
    env_hi = np.zeros(data_eq.shape)
    for i,row in enumerate(data_eq):
        peaks, _ = find_peaks(row, distance=dist)
        vals, _ = find_peaks(-row, distance=dist)
        env_hi[i] = CubicSpline(peaks, row[peaks])(np.arange(row.size))
        env_lo[i] = CubicSpline(vals, row[vals])(np.arange(row.size))

    data_eq = 2 * (data_eq-env_lo)/(env_hi-env_lo) - 1
    data_eq[data_eq > 1] = 1
    data_eq[data_eq < -1] = -1
    return data_eq

def get_if_shot(data_dir, shot_number):
    date = int(data_dir.split('/202308')[1][:2])
    if date >= 10:
        
        data_fname = glob.glob(data_dir + f'{shot_number}_Image*.tiff')[0]
        data = np.array(Image.open(data_fname), dtype=float)

        ref_dir = data_dir + 'background/'
        files = glob.glob(ref_dir + '*.tiff')
        files.sort(key = lambda x: os.path.getmtime(x))

        data_time = os.path.getmtime(data_fname)
        ref_times = np.array([os.path.getmtime(fname) for fname in files])-data_time
        ref_idx = np.argmax(ref_times > 0)-1 # index of last reference shot before the data was taken

        # We took more than one reference shot, so also grab any reference files created around the same time as the last one
        ref_fnames = [files[idx] for idx in np.argwhere(abs(ref_times - ref_times[ref_idx]) < 100 ).flatten()]
        refs = [np.array(Image.open(fname), dtype=float) for fname in ref_fnames]
        return data, refs
    
def get_tilt_angle(ref):
    masked = (cv2.threshold((ref-ref.min())/(ref.max() - ref.min()), 0.5, 1, cv2.THRESH_BINARY)[1] * 255).astype(np.uint8)

    tested_angles = np.linspace(-np.deg2rad(3), np.deg2rad(3), 500, endpoint=False)

    h, theta, d = hough_line(canny(masked, sigma=3), theta=tested_angles)

    angles=[]
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        angles.append(angle)
    tilt_angle = np.rad2deg(np.nanmean(angles))
    return tilt_angle
