# -*- coding: utf-8 -*-
"""
@author: David Magalhaes Sousa
@email: davidmagalhaessousa@gmail.com
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats
from operator import itemgetter

bandgaps = None # Global variable to guarantee that the user has the result available in case the function was not attributed to a variable when executed
def bandgap_diffuse_reflectance(csv : str, dif_points : int = 100, dif_smooth_window : int = 20):
	
	global bandgaps
	bandgaps = []
	
	def find_nearest(array, value):
		array = np.asarray(array)
		idx = (np.abs(array - value)).argmin()
		return array[idx], idx
	
	def drawLine2P(x,y,xlims,color):
		xrange = np.arange(xlims[0],xlims[1],0.1)
		A = np.vstack([x, np.ones(len(x))]).T
		k, b = np.linalg.lstsq(A, y, rcond=None)[0]
		plt.plot(xrange, k*xrange + b, color)
	
	def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
		px = ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
		py = ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
		return [px, py]
	
	# Import data from csv file
	data = pd.read_csv(csv, header=None)
	nm = data[0].values
	dr = data[1].values
	ev = 1240/nm
	evd = ev
	evi = ev
	del data # Memory management
	
	# Tauc transformations for direct (dt) and indirect (it) plots
	dt = (((1-dr/100)**2/(2*dr/100))*evd)**2
	it = (((1-dr/100)**2/(2*dr/100))*evi)**0.5  
	
	# Show the user the plots to help pin point the upper and lower slopes
	plt.plot(evd, dt, "k")
	plt.title("Direct")
	plt.show()
	plt.plot(evi, it, "k")
	plt.title("Indirect")
	plt.show()
	
	# Slice the data to make the script faster by asking the user for the range where the band gap is
	udrange = float(input("Where is the upper limit of the range to determine the direct bandgap? (value in eV and do not enter any value or text if to use the full higher energy spectrum)\n") or np.max(evd))
	ldrange = float(input("Where is the lower limit of the range to determine the direct bandgap? (value in eV and do not enter any value or text if to use the full lower energy spectrum)\n") or np.min(evd))
	uirange = float(input("Where is the upper limit of the range to determine the indirect bandgap? (value in eV and do not enter any value or text if to use the full higher energy spectrum)\n") or np.max(evi))
	lirange = float(input("Where is the lower limit of the range to determine the indirect bandgap? (value in eV and do not enter any value or text if to use the full lower energy spectrum)\n") or np.min(evd))
	
	if udrange != None:
		dt = dt[:find_nearest(ev,udrange)[1]]
		evd = evd[:find_nearest(ev,udrange)[1]]
	if ldrange != None:
		dt = dt[find_nearest(ev,ldrange)[1]:]
		evd = evd[find_nearest(ev,ldrange)[1]:]
	if uirange != None:
		it = it[:find_nearest(ev,uirange)[1]]
		evi = evi[:find_nearest(ev,uirange)[1]]
	if lirange != None:
		it = it[find_nearest(ev,lirange)[1]:]
		evi = evi[find_nearest(ev,lirange)[1]:]
	
	plt.plot(evd, dt, "k")
	normdifd = (-0.5+(np.diff(dt)-np.min(np.diff(dt)))/(np.max(np.diff(dt))-np.min(np.diff(dt))))*2
	cumsum_vec = np.cumsum(np.insert(normdifd, [0]*dif_smooth_window, 0)) 
	normdifd = (cumsum_vec[dif_smooth_window:] - cumsum_vec[:-dif_smooth_window]) / dif_smooth_window
	plt.plot(evd[1:], normdifd*np.max(dt)/2+np.max(dt)/2, "r", alpha=0.3)
	plt.title("Direct")
	plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
	plt.gca().xaxis.grid(True)
	plt.show()
	plt.plot(evi, it, "k")
	normdifi = (-0.5+(np.diff(it)-np.min(np.diff(it)))/(np.max(np.diff(it))-np.min(np.diff(it))))*2
	cumsum_vec = np.cumsum(np.insert(normdifi, [0]*dif_smooth_window, 0)) 
	normdifi = (cumsum_vec[dif_smooth_window:] - cumsum_vec[:-dif_smooth_window]) / dif_smooth_window
	plt.plot(evi[1:], normdifi*np.max(it)/2+np.max(it)/2, "r", alpha=0.3)
	plt.title("Indirect")
	plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.1))
	plt.gca().xaxis.grid(True)
	plt.show()
	
	# Ask user where the upper and lower slopes to obtain the bandgap are
	udb = float(input("Where is the upper slope to determine the direct bandgap?\nWarning: please choose a value away from too much noise. (value in eV and 0 if to skip)\n" or 0))
	if udb != 0:
		udb_v = find_nearest(evd, udb)
		udb_i = udb_v[1] if udb_v[1] < len(dt)-30 else len(dt)-30
		udb_v = udb_v[0]
		ldb = float(input("Where is the lower slope to determine the direct bandgap? (value in eV and it's the flat line to serve as baseline')\n"))
		ldb_v = find_nearest(evd, ldb)
		ldb_i = ldb_v[1] if ldb_v[1] > 30 else 30
		ldb_v = ldb_v[0]
	else:
		udb = None
		ldb = None
	
	uib = float(input("Where is the upper slope to determine the indirect bandgap?\nWarning: please choose a value away from too much noise. (value in eV and 0 if to skip)\n" or 0))
	if uib != 0:
		uib_v = find_nearest(evi, uib)
		uib_i = uib_v[1] if uib_v[1] <= len(it)-31 else len(it)-31
		uib_v = uib_v[0]
		lib = float(input("Where is the lower slope to determine the indirect bandgap? (value in eV and it's the flat line to serve as baseline')\n"))
		lib_v = find_nearest(evi, lib)
		lib_i = lib_v[1] if lib_v[1] >= 30 else 30
		lib_v = lib_v[0]
	else:
		uib = None
		lib = None
	
	# Define a function to plot the spectrum and the linear fits and calculate the intersection between the two fits
	def plotter(array_pair, index_l, index_u, title, d_or_i, plot=True, xlims = [0, 10]):
		ev = d_or_i[0]
		d_or_i = d_or_i[1]
		evl0 = ev[array_pair[0][index_l][1][0]]
		evl1 = ev[array_pair[0][index_l][1][1]]
		dl0 = d_or_i[array_pair[0][index_l][1][0]]
		dl1 = d_or_i[array_pair[0][index_l][1][1]]
		evu0 = ev[array_pair[1][index_u][1][0]]
		evu1 = ev[array_pair[1][index_u][1][1]]
		du0 = d_or_i[array_pair[1][index_u][1][0]]
		du1 = d_or_i[array_pair[1][index_u][1][1]]
		bandgap = findIntersection(evl0,dl0,evl1,dl1,evu0,du0,evu1,du1)
		if plot:
			plt.plot(ev, d_or_i, zorder=0)
			plt.scatter((evl0,evl1), (dl0,dl1), c="g", zorder=1)
			plt.scatter((evu0,evu1), (du0,du1), c="r", zorder=1)
			plt.text(evl0, dl0, str(int(evl0*100)/100), horizontalalignment='right')
			plt.text(evl1, dl1, str(int(evl1*100)/100), horizontalalignment='right')
			plt.text(evu0, du0, str(int(evu0*100)/100), horizontalalignment='right')
			plt.text(evu1, du1, str(int(evu1*100)/100), horizontalalignment='right')
			plt.text(1.05*np.min(ev), np.max(d_or_i), "Points upper: " + str(du[0][1][1]-du[0][1][0]), horizontalalignment='left')
			plt.text(1.05*np.min(ev), 0.88*np.max(d_or_i), "Points lower: " + str(dl[0][1][1]-dl[0][1][0]), horizontalalignment='left')
			drawLine2P((evl0,evl1), (dl0,dl1), xlims, "g")
			drawLine2P((evu0,evu1), (du0,du1), xlims, "r")
			plt.scatter(bandgap[0], bandgap[1], c="k", zorder=10)
			plt.text(bandgap[0], bandgap[1], str(int(bandgap[0]*100)/100), horizontalalignment='left', verticalalignment='top')
			plt.ylim((0, np.max(d_or_i)*1.1))
			plt.title(title)
			plt.show()
		return bandgap[0], (evl0,dl0,evl1,dl1,evu0,du0,evu1,du1)
	
	# Use the plotter function to calculate all the band gaps and scores (pearson product * (upper) or / (lower) by the slope) starting from the chosen points
	# For the upper slope, the nearest highest slope is determined first. Data should not have a high amount of noise at the slope
	direct = None
	if udb != None:
		udb_i = find_nearest(normdifd, np.max(normdifd))[1]+1
		du = []
		print("\nDirect upper slope fitting all points starting from "+str(int(evd[udb_i]*1000)/1000)+" eV:")
		for i in range(udb_i+5, udb_i+1+100):
			du.append([(((stats.pearsonr(evd[j:i], dt[j:i])[0]))*
						((stats.linregress(evd[j:i], dt[j:i])[0]))
					   ,(j, i)) for j in range(udb_i-5, udb_i-200, -1) ])
			printProgressBar(i-(udb_i+1), 100)
		du = [item for sublist in du for item in sublist]
		du = sorted(du, key=itemgetter(0), reverse=True)
		
		dl = []
		print("\nDirect lower slope fitting all points starting from "+str(int(ldb*1000)/1000)+" eV:")
		for i in range(ldb_i+1, ldb_i+1+200):
			dl.append([(stats.pearsonr(evd[j:i], dt[j:i])[0]
						#/stats.linregress(evd[j:i], dt[j:i])[0]
					   ,(j, i)) for j in range(ldb_i-15, ldb_i-200, -1) ])
			printProgressBar(i-(ldb_i+1), 200)
		dl = [item for sublist in dl for item in sublist]
		dl = sorted(dl, key=itemgetter(0), reverse=True)
		
		# After odering the list of all the fits, by the pearson products, create a list of intersections from the best 50 pearson products of the upper and lower slopes, generating 2500 combinations and then calculate the average band gap and its standard deviation
		bgd = plotter((dl, du), 0, 0, "Direct", (evd, dt), True, (np.min(evd), np.max(evd)))
		bgds = []
		print("\nMatching the best 50 scores between upper and lower fits for the direct tauc:")
		for i in range(50):
			for j in range(50):
				bgds.append(plotter((dl, du), i, j, "Doesn't matter", (evd, dt), False)[0])
				printProgressBar((i+1)*(j+1)-1, 50*50)
		bgds = [np.average(bgds), np.std(bgds)]
		direct = ([bgd, [du[0], dl[0]]], bgds)
		bandgaps = [direct, None]
	
	# Same as before but for the indirect band gap
	indirect = None
	if uib != None:
		uib_i = find_nearest(normdifi, np.max(normdifi))[1]+1
		iu = []
		print("\nIndirect upper slope fitting all points starting from "+str(int(uib*1000)/1000)+" eV:")
		for i in range(uib_i+15, uib_i+1+50):
			iu.append([(((1+stats.pearsonr(evi[j:i], it[j:i])[0])**4)
						*((1+stats.linregress(evi[j:i], it[j:i])[0])**10)
					   ,(j, i)) for j in range(uib_i-15, uib_i-50, -1) ])
			printProgressBar(i-(uib_i+1), 50)
		iu = [item for sublist in iu for item in sublist]
		iu = sorted(iu, key=itemgetter(0), reverse=True)
		
		il = []
		print("\nIndirect lower slope fitting all points starting from "+str(int(lib*1000)/1000)+" eV:")
		for i in range(lib_i+15, lib_i+1+200):
			il.append([(stats.pearsonr(evi[j:i], it[j:i])[0]
						#/stats.linregress(evi[j:i],it[j:i])[0]
					   ,(j, i)) for j in range(lib_i-15, lib_i-200, -1) ])
			printProgressBar(i-(lib_i+1), 200)
		il = [item for sublist in il for item in sublist]
		il = sorted(il, key=itemgetter(0), reverse=True)
		
		bgi = plotter((il, iu), 0, 0, "Indirect", (evi, it), True, (np.min(evd), np.max(evd)))
		bgis = []
		print("\nMatching the best 50 scores between upper and lower fits for the indirect tauc:")
		for i in range(50):
			for j in range(50):
				bgis.append(plotter((il, iu), i, j, "Doesn't matter", (evi, it), False)[0])
				printProgressBar((i+1)*(j+1)-1, 50*50)
		bgis = [np.average(bgis), np.std(bgis)]
		indirect = ([bgi, [iu[0], il[0]]], bgis)
		bandgaps = [direct, indirect]
	
	print("Bandgaps = [(Best fit, (Best fir intersection coordinates)), Best fit paramters[Upper(Score, (Indexes)), Lower(Score, (Indexes)), [Average, standard deviation]]]")
	
	return direct, indirect



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 10, fill = 'X', printEnd = "\r"):
	percent = ("{0:." + str(decimals) + "f}").format(100 * ((iteration+1) / float(total)))
	filledLength = int(length * (iteration+1) // total) + 1
	bar = fill * filledLength + '-' * ((length) - filledLength)
	print('\r%s |%s| %s%% %s | %s/%s' % (prefix, bar, percent, suffix, str(iteration+1), str(total)), end = printEnd)
	return None
