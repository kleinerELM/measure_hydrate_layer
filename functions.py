#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, cv2, multiprocessing, time, sys, random, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import median_filter
from skimage import exposure
import pandas as pd
import scipy.optimize

# import other libaries by kleinerELM
home_dir = os.path.dirname(os.path.realpath(__file__))

ts_path = os.path.dirname( home_dir ) + os.sep + 'tiff_scaling' + os.sep
ts_file = 'extract_tiff_scaling'
if ( os.path.isdir( ts_path ) and os.path.isfile( ts_path + ts_file + '.py' ) or os.path.isfile( home_dir + ts_file + '.py' ) ):
    if ( os.path.isdir( ts_path ) ): sys.path.insert( 1, ts_path )
    import extract_tiff_scaling as es
else:
    print( 'missing ' + ts_path + ts_file + '.py!' )
    print( 'download from https://github.com/kleinerELM/tiff_scaling' )
    sys.exit()

def get_random_image_crop( shape, scaling, w_crop, h_crop = 0):
    """
    Calculate the random (square) crop of a larger image.

    Parameters
    ----------
    shape : int
        iterator position for the particle
    scaling : dict
        the scaling dictionary (as defined in ``extract_tiff_scaling``)
    w_crop : int
        Width of the cropped image (in real dimensions like nm or µm as defined in ``scaling``)
    h_crop : int (default = 0)
        Height of the cropped image (in real dimensions like nm or µm as defined in ``scaling``). Default value results in ``h_crop = w_crop``

    Returns
    -------
    tuple of int
        x_min, x_max, y_min, y_max, x_crop, y_crop
    """

    if h_crop == 0: h_crop = w_crop

    x_crop = int(w_crop/scaling['x'])
    if x_crop > shape[0]:
        x_crop = shape[0]
        x_min  = 0
    else:
        x_min = random.randint(0, shape[0]-x_crop)
    x_max = x_min+x_crop

    y_crop = int(h_crop/scaling['y'])
    if y_crop > shape[1]:
        y_crop = shape[1]
        y_min  = 0
    else:
        y_min = random.randint(0, shape[0]-y_crop)
    y_max = y_min+y_crop

    return x_min, x_max, y_min, y_max, x_crop, y_crop

def get_hist(ax):
    n,bins = [],[]
    for rect in ax.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n.append(y1-y0)
        bins.append(x0) # left edge of each bin
    bins.append(x1)     # also get right edge of the last bin

    return n, bins

def denoiseNLMCV2( image, h=15, templateWindowSize=7, searchWindowSize=23 ):
    denoised = np.zeros(image.shape, np.uint8) # empty image
    cv2.fastNlMeansDenoising( image, denoised, float( h ), templateWindowSize, searchWindowSize )

    return denoised

class FitCurveProfile:
    fit_funktion_TeX = r'a \cdot e^{-0.5 \cdot (\frac{log(x-d)-b}{c})^2}'
    parameter_string = "a = {:.3f},\nb = {:.3f},\nc = {:.3f},\nd = {:.3f}"
    f_data           = []
    fx_data          = []
    min_x            = 0

    def fit_function_exp(self, x, a, b, c, d):
        if(x-d <= 0).all() : d = 0
        return a * np.exp(-0.5 * ((np.log(x-d)-b)/c)**2) # -d

    def __init__(self, x_data=[], y_data=[], ax=False, min_x = 0.0, bar_pos = 'r', p0=None, bounds = None, verbose=False):
        # get data directly from a histogram
        if ax != False: y_data, x_data = get_hist(ax)

        # reprocess the bin values (x-positions)
        if bar_pos == 'l':   x_data = x_data[0:len(x_data)-1]
        elif bar_pos == 'c':
            for i in range(len(x_data)-1):
                x_data[i] = (x_data[i]+x_data[i+1])/2
            x_data = x_data[0:len(x_data)-1]
        else:
            bar_pos = 'r'
            x_data = x_data[1:len(x_data)]

        self.y_data = np.array(y_data)
        self.x_data = np.array(x_data)
        self.min_x = min_x

        # crop data if requested (remove values below min_x)
        if min_x > self.x_data.min():
            p = np.where(self.x_data >= min_x)[0].min()   # min position in list
            if p < len(self.x_data)-1:
                self.x_data = self.x_data[p:]
                self.y_data = self.y_data[p:]

        # fit the data
        self.fit_data(p0, bounds)

        if verbose:
            self.plot_function()
            #self.plot_data()

    def plot_function(self):
        print( 'a * e^(-0.5 * ((log(x-d)-b)/c)^2)' )
        print( self.parameter_string )

    def fit_data(self, p0=None, bounds=None):
        if p0 == None:
            p0 = (1.0, 0.5, 0.5, 0)#self.min_x-.001
        if bounds == None:
            bounds = ((-5, -10, 0.001, -1), (5, 10, 10, .99*self.min_x) )
        self.params, self.cv = scipy.optimize.curve_fit(self.fit_function_exp, self.x_data, self.y_data, p0, bounds=bounds, maxfev=10000)
        self.parameter_string = self.parameter_string.format(self.params[0], self.params[1], self.params[2], self.params[3])

    def get_f_data(self):
        if len(self.f_data) == 0:
            self.f_data = []
            for i in range(len(self.x_data)):
                self.f_data.append( self.fit_function_exp( self.x_data[i], self.params[0], self.params[1], self.params[2], self.params[3] ) )
            self.f_data = np.array(self.f_data)
            self.fx_data = self.x_data
        return self.fx_data, self.f_data

    def plot_data(self):
        if len(self.f_data) == 0: self.get_f_data()

        ax = plt.axes()
        ax.plot(self.x_data, self.y_data, label='raw data')
        ax.plot(self.x_data, self.f_data, label='fit')
        ax.set_title('fit of the hydrate rim thickness distribution')
        ax.set_ylabel('grey value')
        ax.set_ylim((0, 1))
        ax.set_xlim((0, self.x_data.max()))
        ax.set_xlabel('circumference position')
        ax.text(1,0.85, r"$%s$" %(self.fit_funktion_TeX),fontsize=20,color="gray")
        ax.text(1,0.45, self.parameter_string, fontsize=15,color="gray")
        ax.legend()
        plt.show()

def process_particle( k, particle_raw, particle_rgb, c_rgb_alite, c_rgb_pores, scaling = 1.0, show_image = True, save_image = True, border_range = 2 ):
    """
    The main function to process the size of the hydration fringe.

    Parameters
    ----------
    k : int
        iterator position for the particle
    particle_raw : np.array of uint8
        cropped raw image around the particle
    particle_rgb :
        cropped segmented image around the particle
    c_rgb_alite : tuple
        color value of alite
    c_rgb_pores : tuple
        color value of pores
    show_image : bool


    Returns
    -------
    dict
        returns the results which are required by ``Image_Processor.merge_particle_result()``
    """
    df_hydrate_fringes_columns = ['particle', 'len_px', 'h_start', 'h_stop', 'boundary'] #, 'len', 'adjacent'
    df_hydrate_fringes = pd.DataFrame([], columns = df_hydrate_fringes_columns)
    polar_raw = particle_raw
    polar     = []
    unploar   = []
    z         = 1

    # the particle image should be square
    if particle_raw.shape[0] != particle_raw.shape[1]:
        print("particle #{:4d} has to be ignored. Not square (Too close to border)! [{}x{} px]".format(k, particle_raw.shape[0],particle_raw.shape[1]))
    else:

        # perimeter length of the particle
        u  = int( particle_raw.shape[0]/2 )
        z = 2*u
        degree = 360
        # polar transform the cropped particle image
        polar_raw = cv2.warpPolar(particle_rgb, (u,degree),(u, u), u, flags=cv2.WARP_FILL_OUTLIERS)
        # process the hydrate fringe
        polar = polar_raw.copy()
        df_hydrate_fringes_list = np.zeros((degree, 5), dtype=np.int16)
        for i in range(degree):
            h_start  = -1
            h_stop   = -1
            adjacent = -1 #-1: nothing, 0: pores, 2: alite
            length   = 0
            for j in range(u):
                # find the start point of the hydrate fringe (not the defined alite color anymore!)
                if not np.array_equal(polar[i][j], c_rgb_alite) and h_start < 0:
                    h_start = j

                # find the end of the hydrate fringe (switch to pore or alite color)
                elif ( (np.array_equal(polar[i][j], c_rgb_alite) or
                        np.array_equal(polar[i][j], c_rgb_pores) ) and
                        h_start >= 0 and
                        h_stop < 0 ):
                    h_stop = j
                    adjacent = 2 if np.array_equal(polar[i][j], c_rgb_alite) else 0
                    # do not continue to search for phase changes to save time
                    break

            if h_stop < 0:      h_stop = h_start

            # hydrate_start should never be < 0!
            if h_start < 0:     print(i, h_start, 'ERROR!!')
            elif adjacent == 0: length = h_stop - h_start

            boundary = -1 if length == 0 else 0 # identify ignored areas

            df_hydrate_fringes_list[i] = np.array([k, length, h_start, h_stop, boundary])


        for i in range(degree):
            #boundary = df_hydrate_fringes_list[i,4] #.loc[i, 'boundary']

            if df_hydrate_fringes_list[i,4] == 0: #boundary == 0:
                for l in range(border_range):
                    p = i+l # position in positive direction
                    n = i-l # position in negative direction

                    if   n < 0:  n = degree+(n)
                    elif p >= degree: p = p-degree

                    if ( (df_hydrate_fringes_list[p, 1] == 0) or
                         (df_hydrate_fringes_list[n, 1] == 0) ):
                        df_hydrate_fringes_list[i,4] = l+1
                        break #  the lowest number

        df_hydrate_fringes = pd.DataFrame(df_hydrate_fringes_list,columns=df_hydrate_fringes_columns)

        if show_image or save_image:
            # color definitions for the image
            c_ignore      = (100,100,100)
            c_stop        = (  0,  0,  0)
            c_ignore_hydr = (100,150,100)

            # re-color the images
            for i, r in df_hydrate_fringes.iterrows():
                if r['boundary'] < 0:
                    polar[i][r['h_stop']:u] = c_ignore_hydr if r['h_start'] == r['h_stop'] else c_ignore
                if r['boundary'] == 0: polar[i][r['h_stop']:u] = c_stop
                elif r['boundary'] > 0:
                    for l in range(1,border_range+1):
                        if r['boundary'] == l:
                            d = int(100/(border_range+1))*(l)
                            polar[i][r['h_start']:r['h_stop']] = (c_ignore_hydr[0]-d, c_ignore_hydr[1]-d, c_ignore_hydr[2]-d)
                            polar[i][r['h_stop'] :u] = c_ignore

        # remove all measurements close to a boundary, with a zero-measurement and reset the index
        df_hydrate_fringes = df_hydrate_fringes[(df_hydrate_fringes.len_px != 0) & (df_hydrate_fringes['boundary'] == 0)].reset_index()
        df_hydrate_fringes['len'] = df_hydrate_fringes['len_px'].apply( lambda x: x*scaling )

        # re-transform the final polar image
        unploar = cv2.warpPolar(polar, (z,z),(u,u), u, cv2.WARP_INVERSE_MAP)

    len_df_hf = len(df_hydrate_fringes)
    show_image = show_image if len(polar) > 0 else False
    return {'id':                 k,
            'df_hydrate_fringes': df_hydrate_fringes,
            'measurements':       len_df_hf,
            #'len_min':            df_hydrate_fringes['len'].min(),
            #'len_max':            df_hydrate_fringes['len'].max(),
            'len_mean':           df_hydrate_fringes['len'].mean(),
            'len_std':            df_hydrate_fringes['len'].std(),
            #'len_median':         df_hydrate_fringes['len'].median(),
            'measure_percent':    0 if z <= 1 else len_df_hf/z,
            'polar_raw':          polar_raw,
            'polar':              polar,
            'unploar':            unploar,
            'show_image':         show_image,
            'z':                  z }

def plot_2d_result(particle_df, hydrate_rim_df, specimen, label, age, scale, unit, min_val, width_f = 1/8, smooting_f = 1, max_dia = 9, max_dia_rim = 7):
	from matplotlib.gridspec import GridSpec
	fig, ax = plt.subplots( 2, 3 )
	plt.close()

	grain_diameters = np.linspace(0,max_dia, num=int(max_dia/width_f))

	f = 1#math.floor(max_dia_rim / (scale*100))
	if f <= 1: f = 1
	max_rim_pos = 0
	bins = [0]
	i=0
	while max_rim_pos < (max_dia_rim*1.1)/scale: # recalculate µm to px
		max_rim_pos += 1
		i += 1
		if i%f==0: bins.append(max_rim_pos)
	bins = np.array(bins)

	bin_centres = ((bins[:-1] + bins[1:]) / 2)
	data3D = np.zeros((len(grain_diameters), len(bin_centres)))

	df = pd.merge(hydrate_rim_df, particle_df[['area', 'diameter', 'perimeter', 'circularity', 'measure_percent']], left_on='particle', right_index=True)

	measurements_per_dia = np.zeros((len(grain_diameters)))
	#particles = np.zeros((len(grain_diameters)))
	for i, dia in enumerate( grain_diameters):
		df_filtered = df[(df.diameter < dia+(width_f*smooting_f)) & (df.diameter > dia)] # & (df['measure_percent'] > 0.25)]
		if len(df_filtered) > 0:
			count, division = np.histogram(df_filtered['len_px'], bins=bins)
			measurements_per_dia[i] = len(df_filtered)
			count = count/count.max()
			data3D[i] = count

	Z = np.array(data3D)
	X, Y = np.meshgrid(bin_centres*scale, grain_diameters) #reintroduce scaling

	S = Z.sum(0)
	for idx, x in enumerate(X[0]):
		if x > min_val:
			#x_min_idx = idx
			break
	S = S[:-1]/S[idx:].max()

	FCP = FitCurveProfile( x_data=X[0], y_data=S, min_x = min_val, bar_pos = 'c', p0=(1.0, 0.5, 0.5, 0) )
	x_data, fit_data = FCP.get_f_data()
	FCP.plot_function()
	f_max = 0
	x_max_pos = 0
	for k,f in enumerate(fit_data):
		if f_max < f:
			f_max = f
			x_max_pos = k

	fig = plt.figure()
	gs = GridSpec(nrows=2, ncols=3, width_ratios=[1, 0.16, 0.07], height_ratios=[0.2, 1], wspace=0.025, hspace=0.025)

	ax[0,0] = fig.add_subplot(gs[0,0])
	ax[0,0].set_yscale('log')
	ax[0,0].hist(grain_diameters[:-1]-grain_diameters[1]/2, grain_diameters-grain_diameters[1]/2, weights=measurements_per_dia[:-1], linewidth=0, color='grey')
	ax[0,0].set_xlim([0,max_dia - .2])
	ax[0,0].set_ylabel( "log\nfrequency" )
	ax[0,0].set_yticks([])
	ax[0,0].set_xticks([])
	ax[0,0].tick_params(axis="x", labelbottom=0)
	ax[0,0].grid(visible=False, which='both', axis='y')
	ax[0,0].spines[['top', 'right']].set_visible(False)


	ax[0,1] = fig.add_subplot(gs[0,1:2])
	ax[0,1].axis("off")
	ax[0,1].text(1.2, 0.3, "{}, {} d".format(label, age), horizontalalignment='center', fontsize=16)


	ax[1,0] = fig.add_subplot(gs[1,0])
	plot = ax[1,0].pcolormesh(Y, X, Z, vmin=0, vmax=1.0, cmap='ocean_r')
	ax[1,0].set_ylabel( "hydrate layer thickness in µm", fontsize=12 )
	ax[1,0].set_ylim([0,max_dia_rim - .2])
	ax[1,0].set_xlabel( "grain diameter in µm", fontsize=12 )
	ax[1,0].set_xlim([0,max_dia - .2])
	ax[0,0].grid(visible=False, which='both', axis='x')


	ax[1,1] = fig.add_subplot(gs[1,1])
	ax[1,1].spines[['top','right']].set_visible(False)
	ax[1,1].hist(X[0][:-1]-X[0][1]/2, X[0]-X[0][1]/2, weights=S, linewidth=0, orientation="horizontal", color='orange')
	ax[1,1].plot(  fit_data/fit_data.max(), x_data, label='fit', color='r')
	ax[1,1].hlines(y=x_data[x_max_pos], xmin=0, xmax=1, linewidth=1, color='r')
	#ax[1,1].plot(S, X[0])
	ax[1,1].set_xlabel( "linear\nfrequency" )
	ax[1,1].set_ylim([0,max_dia_rim - .2])
	ax[1,1].set_yticks([])
	ax[1,1].set_xticks([])
	ax[1,1].text(0.2*S.max(), 5.2, "$l_{}$ =\n{:.2f} {}".format('{max}', x_data[x_max_pos], unit), horizontalalignment='left', fontsize=8)#, rotation='vertical')


	ax[1,2] = fig.add_subplot(gs[1,2])
	#ax[1,2].axis("off")
	cbar = fig.colorbar(plot,cax=ax[1,2])
	cbar.set_label("normalized frequency", fontsize=12)


	plt.savefig( "{} {} 2d rim histogram.pdf".format(specimen, age) )
	plt.show()

	return x_data[x_max_pos]

class Image_Processor:
    label           = 'unnamed'
    scaling         = { 'x' : 1, 'y' : 1, 'unit' : 'nm' }
    f_scaling       = 1000 # usually the files are given in nm - with this factor the base unit is changes e.g. to µm (1000)
    unit            = 'µm'
    preview_size    = 30 # in µm
    preview_size_px =  0 # in px
    min_grain_dia   =  0 # in px
    max_grain_dia   =  0 # in px
    min_grain_area  =  0 # in px
    max_grain_area  =  0 # in px

    c_rgba_pores    = ( 1, 0, 0, 1)
    c_rgba_hydrates = ( 0, 1, 0, 1)
    c_rgba_alite    = ( 0, 0, 1, 1)

    c_rgb_pores     = (255, 0,   0)
    c_rgb_hydrates  = (0, 255,   0)
    c_rgb_alite     = (0,   0, 255)

    coreCount       = multiprocessing.cpu_count()
    processCount    = coreCount-1 if coreCount > 1 else 1
    if processCount > 3: processCount -= 1 # better leave 2 cores for high-core-counts

    file_path       = ''
    file_dir        = ''
    file_name       = ''

    # just to initialize the image variables and for reference. Dimensions are placeholders
    w               = 10
    h               = 10
    img             = np.empty([w, h]) # just to initialize the variable
    removed_pores   = np.empty([w, h]) # just to initialize the variable
    thresh_rgb      = np.ones((w, h, 3), np.uint8)
    multiplied_img  = np.empty((w, h, 3), np.uint8)

    # result lists / dictionaries
    selected_contours         = []
    h_fringe_len              = {}

    display_max_result_images = 5
    save_images               = False
    show_images               = False

    df_hydrate_fringes  = pd.DataFrame([], columns = ['particle', 'len', 'adjacent', 'len_px', 'boundary'])
    df_particles_columns = ['contour', 'cx', 'cy', 'area', 'perimeter', 'circularity', 'len_mean', 'len_std', 'z', 'measurements', 'measure_percent'] # , 'len_min', 'len_max', 'len_median'
    df_particles        = pd.DataFrame([], columns = df_particles_columns)

    def __init__(self,
                 file_path,                      # full path to dataset eg. 'C:\data\dataset.tif'
                 settings = {
                    'label'               : 'unnamed',
                    'age'                 : '0 d',
                    't_pores'             : 100,
                    't_alite'             : 200,
                    'min_grain_dia'       : 0.35,  # diameter in µm #int(   1000/2.9) #0.4µm²
                    'max_grain_dia'       : 3.00,  # diameter in µm #int(  50000/2.9) #
                    'min_circularity'     : 0.60,  # 0.0-1.0, where 1.0 is a perfect circle
                    'enhance_hist'        : True,
                    'denoise'             : True,
                    'denoising_algorithm' : 'nlm',
                    'min_rim'             : 0.5,   # ignore measurements below this value in µm
                    'max_rim'             : 7.0,   # ignore measurements above this value in µm
                    'reduce'              : False
                 },
                 scaling = None,
                 show_images = False,
                 save_images = False
                ):

        if os.path.isfile(file_path):
            self.settings  = settings
            self.label     = settings['label']
            self.min_rim   = settings['min_rim']
            self.max_rim   = settings['max_rim']
            self.file_path = file_path
            self.file_dir  = os.path.dirname(  file_path )
            self.file_name = os.path.basename( file_path ).split('.', 1)[0]

            if save_images:
                self.save_images = save_images
                self.result_image_dir = self.file_dir + os.sep + 'result_images'
                if not os.path.exists( self.result_image_dir ):
                    os.makedirs( self.result_image_dir )
            self.show_images = show_images

            self.load_image( file_path, scaling )
            if show_images or save_images: self.plot_raw_image( save=save_images )

            self.save_metadata()

            self.get_cropped_example()
        else:
            raise Exception( '{} does not exist!'.format(file_path) )

    def get_scaling(self, file_path, scaling = None):

        if scaling == None:
            # autodetect the scaling (works for FEI/thermofischer SEM images or scaled images from ImageJ/Fiji)
            UC = es.unit()
            scaling = es.autodetectScaling( os.path.basename(file_path), self.file_dir)

            # in most cases it makes more sense to change the scaling to µm.
            _, scaling['unit'] = UC.make_length_readable( scaling['x']*self.f_scaling, scaling['unit'])
            scaling['x'] = scaling['x']/self.f_scaling
            scaling['y'] = scaling['y']/self.f_scaling

        self.scaling = scaling
        self.unit    = scaling['unit']

    def load_image(self, file_path, scaling = None):
        self.img = cv2.imread( file_path, cv2.IMREAD_GRAYSCALE )
        self.w = self.img.shape[0]
        self.h = self.img.shape[1]

        self.get_scaling(file_path, scaling)

        print( "Loaded an image with {}x{} px".format(self.w, self.h))
        print( " - scaling: {:.4f} {} / px".format(self.scaling['x'], self.unit))
        print( " - size:    {:.1f} x {:.1f} {} = {:.2f} mm²".format(self.w*self.scaling['x'], self.w*self.scaling['y'], self.unit, self.w*self.scaling['x']*self.w*self.scaling['y']/1000/1000))

        # if the resolution is below 0.05 µm rescale the image
        if self.settings['reduce'] and (self.scaling['x'] * 2) < 0.1:
            f = math.floor( 0.1 / self.scaling['x'] )
            self.w = int(self.w/f)
            self.h = int(self.h/f)
            self.img = cv2.resize(self.img, (self.w, self.h), interpolation= cv2.INTER_LINEAR)
            self.scaling['x'] = self.scaling['x'] * f
            self.scaling['y'] = self.scaling['y'] * f
            print(" - resized image to {}x{} px (f: 1/{}, scaling: {:.4f} {} / px)".format( self.w, self.h, f, self.scaling['x'], self.unit ))

        self.min_grain_dia  = self.settings['min_grain_dia'] / self.scaling['x']
        self.max_grain_dia  = self.settings['max_grain_dia'] / self.scaling['x']
        self.min_grain_area = int( math.pi * ((self.min_grain_dia/2)**2) )
        self.max_grain_area = int( math.pi * ((self.max_grain_dia/2)**2) )

        print( "\n - Selected color thresholds: 0-{:d} for pores {:d}-{:d} for hydrates and {:d}-255 for alite".format(self.settings['t_pores'], self.settings['t_pores']+1, self.settings['t_alite']-1, self.settings['t_alite'])  )

        print( " - Analyzing gains with areas from {:.3f} {}² up to {:.2f} {}²".format(self.min_grain_area*(self.scaling['x']**2), self.unit, self.max_grain_area*(self.scaling['x']**2), self.unit) )

        print( " - Analyzing gains with diameters from {:.2f} {} up to {:.1f} {}".format(self.settings['min_grain_dia'], self.unit, self.settings['max_grain_dia'], self.unit) )

    def save_metadata( self ):
        self.settings['file_path'] = self.file_path
        self.settings['w_px'] = self.w
        self.settings['h_px'] = self.h
        self.settings['unit'] = self.unit
        self.settings['s_x'] = self.scaling['x']
        self.settings['s_y'] = self.scaling['y']

        meta_path = home_dir + os.sep + 'last_processed_{}.meta'.format(self.settings['specimen'])
        if os.path.isfile(meta_path):
            df_meta = pd.read_csv(meta_path, index_col=0)
            df_meta = df_meta.loc[:, ~df_meta.columns.str.contains('^Unnamed')]
            # remove line if image file already exists
            df_meta = df_meta[df_meta.file_path != self.settings['file_path']]
            df_meta = pd.concat([df_meta, pd.DataFrame([self.settings])], ignore_index=True)
            df_meta.reset_index(drop=True, inplace=True)
        else:
            df_meta = pd.DataFrame([self.settings])

        df_meta.to_csv(meta_path)
        print("\nmetadata saved to", meta_path)

    def get_cropped_example(self, preview_size = 20 ):
        if preview_size == None: preview_size = self.preview_size
        self.x_min, self.x_max, self.y_min, self.y_max, self.preview_size_px, _ = get_random_image_crop(
            self.img.shape,
            self.scaling,
            preview_size
        )
        self.crop = self.img[self.x_min:self.x_max, self.y_min:self.y_max]

    def filter_image(self):
        alg = self.settings['denoising_algorithm']
        if self.settings['enhance_hist'] or self.settings['denoise']:
            filter_n = 'enh_' if self.settings['enhance_hist'] else ''
            if self.settings['denoise']: filter_n = filter_n + alg
            filtered_file_path = '{}/{}_{}.tif'.format( self.file_dir, self.file_name, filter_n )

        if os.path.exists(filtered_file_path):
            print('Loading pre-filtered image')
            self.filtered = cv2.imread( filtered_file_path, cv2.IMREAD_GRAYSCALE )
        else:
            # preprocess the image
            self.filtered = exposure.adjust_log( self.img ) if self.settings['enhance_hist'] else self.img

            if self.settings['denoise']:
                if alg == 'nlm':
                    print( "Denoising image using Non Local Means" )
                    self.filtered = denoiseNLMCV2( self.filtered, h=18 )
                elif alg == 'bilateral':
                    print( "Denoising image using Bilateral filter" )
                    self.filtered = denoiseBilateralCV2( self.filtered, d=15 )

            if self.settings['enhance_hist'] or self.settings['denoise']:
                print( 'saving to', filtered_file_path )
                cv2.imwrite( filtered_file_path, self.filtered )

        if self.settings['enhance_hist'] or self.settings['denoise']:
            self.plot_filtered_image(self.save_images)

    def set_thresholds(self):

        # get pores / alite masks
        _, thresh_pores = cv2.threshold(self.filtered, self.settings["t_pores"], 255, cv2.THRESH_BINARY_INV)
        _, thresh_alite = cv2.threshold(self.filtered, self.settings["t_alite"], 255, cv2.THRESH_BINARY)

        # fill holes in alite mask
        bench_time = time.time()
        self.unfiltered_contours, _ = cv2.findContours(thresh_alite, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        thresh_alite = np.zeros((thresh_pores.shape[0], thresh_pores.shape[1]), np.uint8)

        # draw the contours on an empty canvas and fill the contours
        contour_cnt = len(self.unfiltered_contours)
        for i in range(contour_cnt):
            if i%int(contour_cnt/20) == 0 and i > 0: print( '{:3.0f}% done'.format( i/len(self.unfiltered_contours)*100 ))
            thresh_alite = cv2.drawContours(thresh_alite, self.unfiltered_contours, i, (255), -1)
        self.unfiltered_contours, self.unfiltered_hierarchy = cv2.findContours(thresh_alite, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print( "fill holes took {:.0f} s".format(time.time() - bench_time) )

        # hydrates
        thresh_hydrates = ((thresh_pores+thresh_alite)-255)*255

        # combine to 3 phase rgb image
        self.thresh_rgb = np.ones((thresh_pores.shape[0],thresh_pores.shape[1],3), np.uint8)

        #define colors for the phase map (maximum 3)
        self.thresh_rgb[:,:,0] = thresh_pores
        self.thresh_rgb[:,:,1] = thresh_hydrates
        self.thresh_rgb[:,:,2] = thresh_alite

        self.multiplied_img = cv2.addWeighted( cv2.cvtColor(self.filtered, cv2.COLOR_GRAY2RGB), 0.85, self.thresh_rgb, 0.15, 0.0)

        # pores are completely black
        self.removed_pores = cv2.bitwise_and(self.filtered, self.filtered, mask=cv2.bitwise_not(thresh_pores))

        t = self.thresh_rgb.astype(bool)
        f_percent = 100/(self.w * self.h)
        self.pore_area    = t[:,:,0].sum()
        self.hydrate_area = t[:,:,1].sum()
        self.clinker_area = t[:,:,2].sum()
        print('pores: {:.1f}%, hydrates: {:.1f}%, {}: {:.1f}%'.format(f_percent * self.pore_area, f_percent*self.hydrate_area, self.settings['label'], f_percent*self.clinker_area))

        if self.save_images:

            t_rgb_file_path   = self.file_dir + os.sep + self.file_name + "_t_rgb.tif"
            print( 'saving to', t_rgb_file_path )
            cv2.imwrite( t_rgb_file_path, self.thresh_rgb )

            t_multi_file_path = self.file_dir + os.sep + self.file_name + "_t_multi.tif"
            print( 'saving to', t_multi_file_path )
            cv2.imwrite( t_multi_file_path, self.multiplied_img )

            t_rp_file_path    = self.file_dir + os.sep + self.file_name + "_t_rp.tif"
            print( 'saving to', t_rp_file_path )
            cv2.imwrite( t_rp_file_path, self.removed_pores )


    def filter_contours(self, verbose = True):
        self.img_alite_contours = np.zeros((self.w,self.h,3), np.uint8)
        contour_cnt = len(self.unfiltered_contours)
        for i in range(contour_cnt):
            if i%int(contour_cnt/20) == 0 and i > 0: print( '{:3.0f}% done'.format( i/len(self.unfiltered_contours)*100 ))
            M = cv2.moments(self.unfiltered_contours[i])
            area = M['m00']
            if area > self.min_grain_area and area < self.max_grain_area:
                perimeter = cv2.arcLength(self.unfiltered_contours[i],True)
                circularity = 4*math.pi*(area/(perimeter**2))
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                # test if the particle has the correct circularity
                circ_compliant = circularity > self.settings['min_circularity']
                # test if the particle and its perimeter won't touch the image border
                peri_compliant =( cx-perimeter > 0 and
                                  cy-perimeter > 0 and
                                  cx+perimeter < self.w and
                                  cy+perimeter < self.h )
                if circ_compliant and peri_compliant:
                    data = [self.unfiltered_contours[i], cx, cy, area, perimeter, circularity]
                    for i in range(len(self.df_particles_columns)-len(data)): data.append(0) # fill empty columns
                    self.selected_contours.append(data)

                    color = (random.randint(100,255), random.randint(100,255), random.randint(100,255))
                    cv2.drawContours(self.img_alite_contours, self.unfiltered_contours, i, color, cv2.FILLED, cv2.LINE_8, self.unfiltered_hierarchy, 0)

                    if verbose: print('area: {:6.1f}, perim.: {:5.1f}, circ.: {:.2f}, center: ({:5d},{:5d})'.format(area, perimeter, circularity, cx, cy) )

        self.df_particles = pd.DataFrame(self.selected_contours, columns=self.df_particles_columns)

        # calculate diameter and re-apply scaling
        self.df_particles['diameter']  = self.df_particles.area.apply( lambda x: math.sqrt(x / math.pi)*2 * self.scaling['x'] )
        self.df_particles['area']      = self.df_particles.area.apply( lambda x: x * self.scaling['x']**2 )
        self.df_particles['perimeter'] = self.df_particles.perimeter.apply( lambda x: x * self.scaling['x'] )

    def get_particle_crops(self, contour):
        # simplify var names
        cx = contour[2]	    # x coordinate of the center point in the raw image
        cy = contour[1]	    # y coordinate of the center point in the raw image
        r = math.sqrt(contour[3]/math.pi)  # correct for the mean particle radius
        #u  = radius if radius >0 else contour[4] # rounded perimeter length
        u  = int( self.max_rim / self.scaling['x'] + r)

        # crop the image to the image section containing the particle
        # using the perimeter length u as window size
        #particle_raw = self.removed_pores[ cx-u : cx+u, cy-u : cy+u ]
        particle_raw = self.img       [ cx-u : cx+u, cy-u : cy+u ]
        particle_rgb = self.thresh_rgb[ cx-u : cx+u, cy-u : cy+u ]

        return particle_raw, particle_rgb

    def merge_particle_result( self, r ):
        k                         = r['id']
        show_image                = r['show_image']

        df_hf                     = r['df_hydrate_fringes']
        ## apply scaling
        #df_hf                     = df_hf.loc[df_hf['boundary'] == 0].reset_index()
        #df_hf['len']              = df_hf['len_px'].apply( lambda x: x*self.scaling['x'] )

        self.df_hydrate_fringes   = pd.concat([ self.df_hydrate_fringes, df_hf ], ignore_index=True)

        particle_cnt = len(self.df_particles)
        if k % math.ceil(particle_cnt/20) == 0 and k > 0:
            elapsed_time = time.time() - self.bench_time # in s
            #time_per_particle = elapsed_time/k
            estimated_time = elapsed_time/k*particle_cnt # in s #len(self.selected_contours)
            print( "{:3.0f}% done, estimated time: {:3.1f} min, remaining time: {:3.1f} min".format(k/particle_cnt*100, estimated_time/60, (estimated_time-elapsed_time)/60 ))
            self.show_processing_output = False
            show_image = False

        row = self.df_particles.iloc[k].copy()
        #self.df_particles.loc[k, 'len_min']         = r['len_min'] #df_hf['len'].min()
        #self.df_particles.loc[k, 'len_max']         = r['len_max'] #df_hf['len'].max()
        row['len_mean']        = r['len_mean'] #df_hf['len'].mean()
        row['len_std']         = r['len_std'] #df_hf['len'].std()
        #row['len_median']      = r['len_median'] #df_hf['len'].median()
        row['measurements']    = r['measurements'] #len_df_hf             # usable measurements
        row['z']               = r['z']                 # total measurements
        row['measure_percent'] = r['measure_percent'] #0 if r['z'] <= 1 else len_df_hf/r['z']
        self.df_particles.iloc[k] = row
        # debugging output
        if self.save_images or show_image or self.show_processing_output:
            is_border_particle = (len(r['polar']) < 1)
            has_measurements   = r['measurements'] > 0

            result_file_name = self.file_dir + os.sep + self.file_name + "_p{}".format(k)
            if has_measurements: df_hf.to_feather( result_file_name+'.feather' )

            title = "particle #{:4d} - {:4d} measurements, mean rim width: {:4.1f} ± {:4.1f} {}".format(
                k,
                r['measurements'],
                #r['len_min'],    r['len_max'], self.unit,
                r['len_mean'],   df_hf['len'].std(), self.unit,
                #r['len_median'], self.unit
            ) if has_measurements else 'particle #{:4d} -   no measurements!'.format( k )

            if (self.save_images or show_image) and not is_border_particle:
                particle_raw, particle_rgb = self.get_particle_crops(self.selected_contours[k])
                self.plot_particle_measurement(
                    title,
                    self.file_name + "_p{}".format(k),
                    particle_raw,
                    particle_rgb,
                    r['polar_raw'],
                    r['polar'],
                    r['unploar']
                )

                # avoid cluttered jupyter file
                if not show_image:
                    plt.close()
                    if self.show_processing_output: print(title)
                else:
                    plt.show()
            else:
                if self.show_processing_output and not is_border_particle: print(title)

    #border_range - remove x measurements next to a invalid measurement
    def process_particles(self, multithreading = True, processCount = 0, border_range = 3, benchmark_particles_max = 15, load_saved=True, verbose = False):
        displayed_images        = 0
        self.show_processing_output = True #enable processing output for the first particles

        # load dataset if already processed. Time saving to play with the
        dataset_file_name1 = self.file_dir + os.sep + self.file_name + "_hyd-rim"
        dataset_file_name2 = self.file_dir + os.sep + self.file_name + "_particles"
        if load_saved and (os.path.isfile(dataset_file_name1+'.csv') or os.path.isfile(dataset_file_name1+'.feather')) and (os.path.isfile(dataset_file_name2+'.csv') or os.path.isfile(dataset_file_name2+'.feather')):
            print('loading existing dataset.')
            if os.path.isfile(dataset_file_name1+'.feather'):
                self.df_hydrate_fringes = pd.read_feather(dataset_file_name1+'.feather')
                self.df_particles       = pd.read_feather(dataset_file_name2+'.feather')
            else:
                self.df_hydrate_fringes = pd.read_csv(dataset_file_name1+'.csv')
                self.df_particles       = pd.read_csv(dataset_file_name2+'.csv')
            print('loaded {} measurements'.format(len(self.df_hydrate_fringes)))
        else:
            # check for and start multithreading
            if not (processCount > 0 and processCount < self.processCount): processCount = self.processCount
            if self.processCount < 2: multithreading = False
            if multithreading:
                print( 'Using {} processes to measure the hydrate fringes.'.format(processCount) )
                pool = multiprocessing.Pool( processCount )

            min_dim  = int( self.max_rim / self.scaling['x'] )*2
            self.bench_time = time.time()
            ignored_particles = 0
            for k in range(len(self.selected_contours)):
                if k >= benchmark_particles_max: self.show_processing_output = verbose

                show_image = (self.display_max_result_images > displayed_images ) & (k < benchmark_particles_max)
                particle_raw, particle_rgb = self.get_particle_crops( self.selected_contours[k] )
                if particle_raw.shape[0] >= min_dim and particle_raw.shape[0] == particle_raw.shape[1]:
                    if not multithreading or k < benchmark_particles_max:
                        # single threaded run, showing output (for debugging)
                        result = process_particle( k, particle_raw, particle_rgb, self.c_rgb_alite, self.c_rgb_pores, self.scaling['x'], show_image, self.save_images, border_range )
                        self.merge_particle_result( result )

                        if show_image and len(result['df_hydrate_fringes']) > 0: displayed_images += 1

                    else:
                        # multithreaded run
                        if k <= benchmark_particles_max: print('-'*20)
                        pool.apply_async( process_particle, args=( k, particle_raw, particle_rgb, self.c_rgb_alite, self.c_rgb_pores, self.scaling['x'], show_image, self.save_images, border_range ), callback=self.merge_particle_result)
                else:
                    ignored_particles += 1

            # re-merge threads
            if multithreading:
                pool.close() # close the process pool
                pool.join()  # wait for all tasks to finish

            if ignored_particles > 0: print( 'ignored {} of {} particles, since they were too close to the border'.format( ignored_particles, len(self.selected_contours) ) )
            self.df_hydrate_fringes.to_feather( dataset_file_name1+'.feather' )
            c = self.df_particles.columns.to_list()
            c.pop(0)
            self.df_particles[c].to_feather( dataset_file_name2+'.feather' )


    def fit_distributions( self, min_rim=None, max_dia_rim=None, p0=None, bounds=None, verbose=False ):
        if min_rim == None: min_rim = self.min_rim
        if max_dia_rim == None: max_dia_rim = self.max_rim

        self.fits = []
        df = self.df_hydrate_fringes[ ( self.df_hydrate_fringes.len < max_dia_rim ) ]
        df = pd.merge(df, self.df_particles[['area', 'diameter', 'perimeter', 'circularity', 'measure_percent']], left_on='particle', right_index=True)

        fig, ax = plt.subplots(math.ceil( int(self.settings['max_grain_dia']) / 3 ), 3, figsize=( 18, 17 ))
        fig.suptitle( "Trying to fit the hydrate rim histograms from {} to {} {}.\n\n".format( min_rim, max_dia_rim, self.unit ), fontsize=16 )

        e = 0
        f = 0
        for i in range( int(self.settings['max_grain_dia'])):
            df_filtered = df[(df.diameter < i+1) & (df.diameter > i) & (df['measure_percent'] > 0.25)]
            if len(df_filtered) > 0:
                measurement_cnt = len(df_filtered)

                ax[e][f].set_title(  "hydrate rim thickness (particle diameter {} < x < {} {})\n of {} Measurements".format(i, i+1, self.unit, measurement_cnt) )
                df_filtered.len.hist(bins=100, ax=ax[e][f], density=True, label='hydrate rim thickness')

                FCP = FitCurveProfile( ax=ax[e][f], min_x = min_rim, bar_pos = 'c', p0=p0, bounds=bounds, verbose=verbose )
                x_data, fit_data = FCP.get_f_data()
                ax[e][f].plot(x_data, fit_data, label='fit')

                df.len.hist(bins=100, ax=ax[e][f], density=True, label='hydrate rim thickness (full dataset)', alpha=0.4)
                ax[e][f].set_xlabel( "hydrate rim thickness in {}".format( self.unit ) )
                ax[e][f].set_ylabel( "frequency" )
                ax[e][f].set_xlim([0, max_dia_rim])
                ax[e][f].set_ylim([0, 1])
                ax[e][f].legend()
                ax[e][f].text(max_dia_rim/3, 0.6 , r"$%s$" %(FCP.fit_funktion_TeX), fontsize=20)
                ax[e][f].text(max_dia_rim/3, 0.25, FCP.parameter_string, fontsize=15)

                m = max(enumerate(fit_data),key=lambda x: x[1])[0]
                max_rim = 0
                if m > 0:
                    max_rim = x_data[m]
                    if FCP.y_data[m] < 0.05:
                        print( 'particle diameter {} < x < {} {}: maximum frequency seems unplausible: {:.3f} at {:.3f} {}'.format(i, i+1, self.unit, FCP.y_data[m], x_data[m], self.unit) )
                        max_rim = 0
                    else:
                        print( 'particle diameter {} < x < {} {}: rim thickness {:.3f} {}.'.format( i, i+1, self.unit, x_data[m], self.unit ) )
                else:
                    print( 'particle diameter {} < x < {} {}: no maximum in value range'.format(i, i+1, self.unit) )

                self.fits.append(
                    {'dia_min': i,
                        'dia_max': i+1,
                        'a':       FCP.params[0],
                        'b':       FCP.params[1],
                        'c':       FCP.params[2],
                        'd':       FCP.params[3],
                        'x_data':  x_data,
                        'f_data':  fit_data,
                        'max_rim': max_rim,
                        'measurement_cnt': measurement_cnt
                    }
                )
            else: print("No measurements for particles with a diameter {} < x < {} µm".format(i, i+1, len(df_filtered)))

            f += 1
            if f > 2:
                f  = 0
                e += 1
        plt.tight_layout()
        plt.show()

        df_filtered = df[(df.diameter > self.settings['max_grain_dia']) & (df['measure_percent'] > 0.25)]
        if len(df_filtered) > 0:
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            fig.suptitle( "hydrate rim thickness (particles diameter > {} µm)\n{} Measurements".format(self.settings['max_grain_dia'], len(df_filtered)), fontsize=12 )
            df.len.hist(bins=100, ax=ax, density=True)
            df[(df.diameter > self.settings['max_grain_dia']) ].len.hist(bins=100, ax=ax, density=True, alpha=0.7)
            ax.set_xlabel( "hydrate rim thickness in {}".format( self.unit ) )
            ax.set_ylabel( "frequency" )
            ax.set_xlim([0, max_rim])
            ax.set_ylim([0, 1])
            plt.show()
        else: print("No measurements for particles with a diameter > {} µm".format(self.settings['max_grain_dia']))

    #######################
    # plotting functions
    #######################

    def get_image_axis_labels(self, ax):
        ticks    = ax.get_xticks()
        form = '%d' if ticks.max() > 10 else '%.1f'
        labels = [ form % e for e in ticks * self.scaling['x'] ]

        return ticks, labels

    def process_full_statistics( self, max_rim = None ):
        if max_rim == None: max_rim = self.max_rim

        print( '{} measurements'.format( len(self.df_hydrate_fringes) ) )

        f = self.df_hydrate_fringes[ ( self.df_hydrate_fringes.len < max_rim )]
        filtered = f

        fig, ax = plt.subplots(1, 3, figsize=( 18, 6 ))
        fig.suptitle( "Basic statistics of hydrate rim measurements.", fontsize=16 )

        i = 0
        ax[i].set_title(   "Histogram of the hydrate rim thickness" )
        filtered.len.hist(bins=100, ax=ax[i], density=True)
        ax[i].set_xlabel( "hydrate rim thickness in {}".format( self.unit ) )
        ax[i].set_ylabel( "frequency" )
        ax[i].set_xlim([0, max_rim])

        i += 1
        ax[i].set_title( "Histogram of the particle-diameter distribution" )
        self.df_particles.diameter.hist(bins=50, ax=ax[i], density=True)
        ax[i].set_xlim([0, math.sqrt(self.max_grain_area / math.pi)*2* self.scaling['x']])
        ax[i].set_ylim([0, 1])
        ax[i].set_xlabel( "diameter in {}".format( self.unit ) )
        ax[i].set_ylabel( "frequency" )

        i += 1
        ax[i].set_title(   "Histogram of the particle-area distribution" )
        self.df_particles.area.hist(bins=50, ax=ax[i], density=True)
        ax[i].set_xlim([0, self.max_grain_area* self.scaling['x']**2])
        ax[i].set_ylim([0, 1])
        ax[i].set_xlabel( "area in {}²".format( self.unit ) )
        ax[i].set_ylabel( "frequency" )
        plt.show()

    def plot_raw_image( self, save=False ):
        fig, ax = plt.subplots(1, 2, figsize=( 18, 8 ))
        fig.suptitle( "Loaded image ({})".format(self.label), fontsize=16 )

        i = 0
        ax[i].imshow( self.img, cmap='gray' )
        ax[i].set_title(  "Image {}".format( self.file_name ) )
        ax[i].set_xlabel( "x in {}".format( self.unit ) )
        ax[i].set_ylabel( "y in {}".format( self.unit ) )
        tick_pos, tick_labels = self.get_image_axis_labels( ax[i] )
        ax[i].set_xticks(tick_pos, tick_labels )
        ax[i].set_yticks(tick_pos, tick_labels )
        ax[i].set_xlim([0, self.h-1])
        ax[i].set_ylim([self.w-1, 0])

        i += 1
        ax[i].hist(self.img.ravel(),256,[0,256])
        ax[i].set_title(  "Histogram {}".format( self.file_name ) )
        ax[i].set_xlim([0,256])
        ax[i].set_yticks( [], [] )
        ax[i].set_xlabel( "grey value" )
        ax[i].set_ylabel( "frequency" )

        if save: plt.savefig(self.file_dir + os.sep + self.file_name + "_raw-histogram.png")

    def plot_filtered_image( self, save=False ):
        fig, ax = plt.subplots(1, 3, figsize=( 18, 5 ))
        fig.suptitle( "Denoising result ({})".format(self.label), fontsize=16 )

        i = 0
        ax[i].imshow( self.crop, cmap='gray' )
        ax[i].set_title(  "raw image" )
        ax[i].set_xlabel( "x in {}".format( self.unit )  )
        ax[i].set_ylabel( "y in {}".format( self.unit )  )
        tick_pos, tick_labels = self.get_image_axis_labels( ax[i] )
        ax[i].set_xticks(tick_pos, tick_labels )
        ax[i].set_yticks(tick_pos, tick_labels )
        ax[i].set_xlim([0,self.preview_size_px])
        ax[i].set_ylim([self.preview_size_px,0])

        i += 1
        ax[i].imshow( self.filtered[self.x_min:self.x_max, self.y_min:self.y_max] , cmap='gray' )
        ax[i].set_title(  "denoised image" )
        ax[i].set_xlabel( "x in {}".format( self.unit )  )
        ax[i].set_xticks(tick_pos, tick_labels )
        ax[i].set_yticks( [], [] )
        ax[i].set_xlim([0,self.preview_size_px])
        ax[i].set_ylim([self.preview_size_px,0])

        i += 1
        ax[i].hist( self.img.ravel(),256,[0,256])
        ax[i].hist( self.filtered.ravel(),256,[0,256])
        ax[i].set_title(  "histogram" )
        ax[i].set_xlim([0,256])
        ax[i].set_xlabel( "grey value" )
        ax[i].set_ylabel( "frequency" )
        height = self.w * self.h
        ax[i].axvline(self.settings["t_pores"], 0, height, color='red')
        ax[i].axvline(self.settings["t_alite"], 0, height, color='green')

        if save: plt.savefig(self.file_dir + os.sep + self.file_name + "_filtered-histogram.png")

    def plot_thresh_result( self, save=False ):
        fig, ax = plt.subplots(1, 3, figsize=( 18, 5 ))
        fig.suptitle( "Threshold segmentation result ({})".format(self.label), fontsize=16 )

        i = 0
        ax[i].imshow( self.thresh_rgb[self.x_min:self.x_max, self.y_min:self.y_max] )
        ax[i].set_title(  "Phase map (red=pores, green=hydrate, blue = C3S)" )
        ax[i].set_xlabel( "x in {}".format( self.unit ) )
        ax[i].set_ylabel( "y in {}".format( self.unit ) )
        tick_pos, tick_labels = self.get_image_axis_labels( ax[i] )
        ax[i].set_xticks(tick_pos, tick_labels )
        ax[i].set_yticks(tick_pos, tick_labels )
        ax[i].set_xlim([0,self.preview_size_px])
        ax[i].set_ylim([self.preview_size_px,0])

        i += 1
        ax[i].imshow( self.multiplied_img[self.x_min:self.x_max, self.y_min:self.y_max] )
        ax[i].set_title(  "Phase map multiplied with filtered image".format( self.file_name ) )
        ax[i].set_xlabel( "x in {}".format( self.unit ) )
        ax[i].set_xticks(tick_pos, tick_labels )
        ax[i].set_yticks( [], [] )
        ax[i].set_xlim([0, self.preview_size_px])
        ax[i].set_ylim([self.preview_size_px, 0])

        i += 1
        ravel = self.filtered.ravel()
        ax[i].hist(ravel, self.settings["t_pores"], [0, self.settings["t_pores"]], color=self.c_rgba_pores)
        ax[i].hist(ravel, self.settings["t_alite"]-self.settings["t_pores"], [self.settings["t_pores"],self.settings["t_alite"]], color=self.c_rgba_hydrates)
        ax[i].hist(ravel, 256-self.settings["t_alite"], [self.settings["t_alite"], 256], color=self.c_rgba_alite)
        #ax.hist(filtered.ravel(),256,[0,256])
        ax[i].set_title(  "Histogram of denoised and equalized image" )
        ax[i].set_xlim([0,256])
        ax[i].set_xlabel( "grey value" )
        ax[i].set_ylabel( "frequency"  )
        height = self.w * self.h
        ax[i].axvline(self.settings["t_pores"], 0, height, color='red')
        ax[i].axvline(self.settings["t_alite"], 0, height, color='green')

        if save: plt.savefig(self.file_dir + os.sep + self.file_name + "_thresholds.png")

    def plot_selected_contours( self ):

        fig, ax = plt.subplots(1, 2, figsize=( 18, 8 ))
        fig.suptitle( "Selected {} particles ({})".format(len(self.selected_contours), self.label), fontsize=16 )

        i = 0
        ax[i].imshow( cv2.addWeighted( cv2.cvtColor(self.removed_pores, cv2.COLOR_GRAY2RGB), 0.5, self.img_alite_contours, 0.5, 0.0) )
        ax[i].set_title(  "filtered image with selected particles" )
        ax[i].set_xlabel( "x in {}".format( self.unit ) )
        ax[i].set_ylabel( "y in {}".format( self.unit ) )
        tick_pos, tick_labels = self.get_image_axis_labels( ax[i] )
        ax[i].set_xticks(tick_pos, tick_labels )
        ax[i].set_yticks(tick_pos, tick_labels )
        ax[i].set_xlim([0,self.h-1])
        ax[i].set_ylim([self.w-1,0])

        i += 1
        ax[i].imshow( self.img_alite_contours )
        ax[i].set_title(  "selected particles" )
        ax[i].set_xlabel( "x in {}".format( self.unit ) )
        #ax[i].set_ylabel( "y in {}".format( self.unit ) )
        ax[i].set_xticks(tick_pos, tick_labels )
        ax[i].set_yticks( [], [] )
        ax[i].set_xlim([0,self.h-1])
        ax[i].set_ylim([self.w-1,0])

        plt.show()

    def plot_particle_measurement( self, title, result_file_name, particle_raw, particle_rgb, polar_raw, polar, unploar ):
        fig, ax = plt.subplots(1, 5, figsize=( 25, 5 ))
        fig.suptitle( title, fontsize=16 )

        # perimeter length of the particle
        u  = int( particle_raw.shape[0]/2 )

        mask = np.zeros(unploar.shape[:2], dtype="uint8")
        cv2.circle(mask, (u, u), u-1, 255, -1)
        unploar[mask == 0] = (255, 255, 255)

        rim = math.ceil(self.max_rim)
        r_labels = np.array( range(rim+1 ) )
        r_ticks  = np.floor( r_labels /self.scaling['x'])

        x_labels_u = [rim*-1, 0, rim] #np.concatenate((np.flip(r_labels*-1)[:-1], r_labels ), axis=0)
        x_ticks_u  = [0, u, particle_raw.shape[0]-1] #np.floor( r_labels /self.scaling['x'])
        #print(r_labels,r_ticks, x_labels_u, x_ticks_u)
        i = 0
        ax[i].imshow( particle_raw, cmap="gray" )
        #ax[i].set_title(  "particle in the raw image" )
        ax[i].set_xlabel( "x in {}".format( self.unit ) )
        ax[i].set_ylabel( "y in {}".format( self.unit ) )
        #tick_pos, tick_labels = self.get_image_axis_labels( ax[i] )
        ax[i].set_xticks( x_ticks_u, x_labels_u )
        ax[i].set_yticks( x_ticks_u, x_labels_u )
        ax[i].set_xlim([0,particle_raw.shape[1]-1])
        ax[i].set_ylim([particle_raw.shape[0]-1,0])

        i += 1
        ax[i].imshow( particle_rgb )
        #ax[i].set_title(  "particle in the segmented image" )
        ax[i].set_xlabel( "x in {}".format( self.unit ) )
        ax[i].set_ylabel( "y in {}".format( self.unit ) )
        ax[i].set_xticks( x_ticks_u, x_labels_u )
        ax[i].set_yticks( [], [] )
        ax[i].set_xlim([0,particle_rgb.shape[1]-1])
        ax[i].set_ylim([particle_rgb.shape[0]-1,0])
        print(polar.shape[0])
        i += 1
        ax[i].imshow( polar_raw )
        #ax[i].set_title(  "polar transformed segmented image" )
        ax[i].set_xlabel( "distance in {}".format( self.unit ) )
        ax[i].set_ylabel( "angle" )
        ax[i].set_yticks([0, 90, 180, 270, 359])
        ax[i].set_xticks( r_ticks, r_labels )
        ax[i].set_xlim([0, polar_raw.shape[1]-1])
        ax[i].set_ylim([0, polar_raw.shape[0]-1])

        i += 1
        ax[i].imshow( polar )
        #ax[i].set_title(  "polar transformed image of the extracted grain" )
        ax[i].set_xlabel( "distance in {}".format( self.unit ) )
        ax[i].set_xticks( r_ticks, r_labels )
        ax[i].set_ylabel( "angle" )
        ax[i].set_yticks([0, 90, 180, 270, 359])
        ax[i].set_xlim([0,polar.shape[1]-1])
        ax[i].set_ylim([0,polar.shape[0]-1])

        i += 1
        ax[i].imshow( unploar )
        #ax[i].set_title(  "extracted grain with hydrate fringe" )
        ax[i].set_xlabel( "x in {}".format( self.unit ) )
        ax[i].set_ylabel( "y in {}".format( self.unit ) )
        ax[i].set_xticks( x_ticks_u, x_labels_u )
        ax[i].set_yticks( x_ticks_u, x_labels_u )
        ax[i].set_xlim([0,unploar.shape[1]-1])
        ax[i].set_ylim([unploar.shape[0]-1,0])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)

        if self.save_images: plt.savefig( self.result_image_dir + os.sep + result_file_name + ".svg" )

    def plot_hydrate_rim_thicknesses( self, min_rim=None, max_rim=None, y_limit=[0,5] ):
        if min_rim == None: min_rim = self.min_rim
        if max_rim == None: max_rim = self.max_rim

        sns.set_style("whitegrid")

        max_measurement_cnt = 0
        max_rims = []
        for i,f in enumerate(self.fits):
            if max_measurement_cnt < f['measurement_cnt']: max_measurement_cnt = f['measurement_cnt']
            max_rims.append(f['max_rim'])

        df = self.df_hydrate_fringes[ ( self.df_hydrate_fringes.len < max_rim ) ]
        df = pd.merge(df, self.df_particles[['area', 'diameter', 'perimeter', 'circularity', 'measure_percent']], left_on='particle', right_index=True)
        f_data_l  = []
        for j in range( int(self.settings['max_grain_dia'])):
            df_filtered = df[(df.diameter < j+1) & (df.diameter > j) & (df.measure_percent > 0.25) & (df.len > min_rim)]
            if len(df_filtered) > 0:
                f_data_l.append(df_filtered.len.to_list())

        for j,f in enumerate(self.fits):
            self.fits[j]['max_rim'] = max(enumerate(f['f_data']),key=lambda x: x[1])[0]

        ### 3D-Plot of fit functions
        fig = plt.figure(figsize=( 5, 5.5 ))
        fig.suptitle( "Calculated hydrate rim of\n{}".format(self.label), fontsize=16 )
        ax = fig.add_subplot(projection='3d')

        for i,f in enumerate(self.fits):
            # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
            ax.plot( f['x_data'], f['f_data'], zs=f['dia_max'], zdir='y', alpha=f['measurement_cnt']/max_measurement_cnt, color = 'blue')#, color = colors[i])

        ax.set_xlabel( "hydrate rim thickness in {}".format( self.unit ) )
        ax.set_ylabel( "grain diameter in {}".format( self.unit ) )
        ax.set_zlabel( "frequency" )
        ax.set_xlim([0, max_rim])

        plt.tight_layout()
        plt.show()
        plt.close()


        ### 2D plots of fit functions and maxima of hydrate fringe thickness
        fig, ax = plt.subplots(1, 5, figsize=( 30, 6 ))
        fig.suptitle( "Calculated hydrate rim of {}".format(self.label), fontsize=16 )

        i = 0
        for j,f in enumerate(self.fits):
            self.fits[j]['max_rim'] = max(enumerate(f['f_data']),key=lambda x: x[1])[0]
            ax[i].plot( f['x_data'], f['f_data'], alpha=f['measurement_cnt']/max_measurement_cnt, color = 'blue')

        ax[i].set_xlabel( "hydrate rim thickness in {}".format( self.unit ) )
        ax[i].set_ylabel( "frequency" )
        ax[i].set_xlim([0, max_rim/2])

        i += 1
        ax[i].plot( range(1, len(self.fits)+1), max_rims, '-x', linewidth=1 )
        ax[i].set_xlabel( "grain diameter in {}".format( self.unit ) )
        ax[i].set_ylabel( "hydrate rim thickness in {}".format( self.unit ) )
        ax[i].set_ylim(y_limit)

        i += 1
        ax[i].plot( range(1, len(self.fits)+1), max_rims, '-x', linewidth=1 )
        ax[i].boxplot(f_data_l,
                    #positions=[2, 4, 6],
                    widths=.5,
                    patch_artist=True,
                    #usermedians=max_rims,
                    showmeans=False,
                    showfliers=False,
                    meanline=False,
                    #medianprops={"color": "white", "linewidth": 0.5},
                    #boxprops={"facecolor": "C0", "edgecolor": "white",
                    #          "linewidth": 0.5},
                    #whiskerprops={"color": "C0", "linewidth": 1.5},
                    #capprops={"color": "C0", "linewidth": 1.5}
                    )
        ax[i].set_xlabel( "grain diameter in {}".format( self.unit ) )
        ax[i].set_ylabel( "hydrate rim thickness in {}".format( self.unit ) )
        ax[i].set_ylim(y_limit)

        i += 1
        sns.violinplot(data=f_data_l, scale="count",inner=None, linewidth=0.1, cut=0, ax=ax[i] )

        ax[i].plot( range(0, len(self.fits)), max_rims, '-x', linewidth=1, color='black' )
        ax[i].set_xlabel( "grain diameter in {}".format( self.unit ) )
        ax[i].set_xticks( range(len(self.fits)), range(1,len(self.fits)+1) )
        ax[i].set_ylabel( "hydrate rim thickness in {}".format( self.unit ) )
        ax[i].set_ylim(y_limit)

        i += 1
        sns.violinplot(data=f_data_l, linewidth=0.5, cut=0, ax=ax[i])

        ax[i].plot( range(len(self.fits)), max_rims, '-x', linewidth=1, color='black' )
        ax[i].set_xlabel( "grain diameter in {}".format( self.unit ) )
        ax[i].set_xticks( range(len(self.fits)), range(1,len(self.fits)+1) )
        ax[i].set_ylabel( "hydrate rim thickness in {}".format( self.unit ) )
        ax[i].set_ylim(y_limit)

        plt.show()

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    #remove root windows
    root = tk.Tk()
    root.withdraw()

    home_dir = os.path.dirname(os.path.realpath(__file__))

    tiff_file = filedialog.askopenfilename(initialdir = home_dir,title = "Select 8-Bit SEM-BSE image",filetypes = (("Tif file","*.tif"),("Tiff file","*.tiff")))

    CT = Image_Processor(tiff_file)

    print('done...')