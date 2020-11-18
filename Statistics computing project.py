import scipy as sp
import STOM_higgs_tools as higgs
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit# Import the function curve_fit from the optimize package in Scipy
import scipy.stats as stats
from scipy import special
import random as random
from tqdm import tqdm
from tqdm import tqdm_gui

#Constants
N_b = 10e5 # Number of background events, used in generation and in fit.
b_tau = 30. # Spoiler.


vals = higgs.generate_data()

# Make a histogram.
N_bins = 30
bin_heights, bin_edges, patches  = plt.hist(vals, range = [104, 155], bins = N_bins, histtype = "stepfilled") # bin_heights and bin_edges are numpy arrays.
bincentres = 0.5*(bin_edges[1:]+bin_edges[:-1]) #calculate the bin centres
plt.errorbar(bincentres[0:N_bins], bin_heights, yerr = np.sqrt(bin_heights), fmt = ',' , capsize  = 4) #plot errorbar
bin_width = 1.7
#print(len(bin_edges))

#Background estimation histogram
N_bins_b =  71 #120 / 1.7
bin_heights_b, bin_edges_b  = np.histogram(vals, range = [0, 120], bins = N_bins_b) # bin_heights and bin_edges are numpy arrays.
bincentres_b = 0.5*(bin_edges_b[1:]+bin_edges_b[:-1]) #calculate the bin centres
#plt.errorbar(bincentres[0:N_bins], bin_heights, np.sqrt(bin_heights), fmt = ',' , capsize  = 4) #plot errorbar


#Fitting exponential function
def background(x, A, lambd):
	return A * np.exp(-x / lambd)

#guesses
guess_A = N_b
guess_lambd = b_tau
p0 = [guess_A, guess_lambd] #initial parameters array

#fit

#Find  where E < 120 GeV
E_max = max(max(np.where(bincentres_b < 120)))
#print("You can fit until the bin number", E_max, "to fit on background only.")
fit  = curve_fit(background, bincentres_b[0:E_max], bin_heights_b[0:E_max],  p0 = p0)  #fitted on values for E < 120 GeV (excluding possible signal)

#Params
A = fit[0][0]
A_sig = np.sqrt(fit[1][0][0])
lambd = fit[0][1]
lambd_sig = np.sqrt(fit[1][1][1])

print(f"The exponential fit parameters (E < 120 GeV): A ={A: .4f} +- {A_sig: .0f}, lambd =  {lambd: .2f} +- {lambd_sig: .2f}")
#print(len(bincentres_b[0:E_max]))

exponential_fit = background(bincentres, *fit[0])
plt.plot(bincentres, exponential_fit, color = "red")

plt.show()

################### 3. Goodness of Fit #####################################

#Chi-squared value - background

chi_background = higgs.get_B_chi(vals, (0,120), N_bins_b, fit[0][0], fit[0][1])
N_dof_b = N_bins_b -2
print(chi_background, "N_dof =", N_dof_b)


#################### 4. Hypothesis testing #################################
########## Part a)  ######################        

#Chi-squared value - background but also including signal

chi_background_bs = higgs.get_B_chi(vals, (104,155), N_bins, fit[0][0], fit[0][1])
N_dof_bs = int(N_bins - 2)
#print(chi_background_bs[1])
print("Overlaying the background fit over signal:", chi_background_bs, "N_dof =", N_dof_bs)

# p-value
p_value_b  = stats.chi2.sf(chi_background_bs[1], N_dof_bs)
print("Background overlayed over signal p-value:", p_value_b)

plt.ylabel("Number of events")
plt.xlabel("$m_{\gamma \gamma }$ [GeV]")
plt.savefig("Background_only_fit.png")
plt.show()


#################### Part b) #####################
N_loop = 1000

chi_b = []
p_value_b = []

for i in tqdm(range(N_loop)):

	#Create new background signal that will change through loop
	vals_4b = higgs.generate_data(0)
	N_dof_4b = int(N_bins - 2)
	bin_heights_4b, bin_edges_4b = np.histogram(vals_4b, range = [104, 155], bins = N_bins) # bin_heights and bin_edges are numpy arrays.

	#Find Chi-squared value each loop
	chi_background_4b = higgs.get_B_chi(vals_4b, (104,155), N_bins, fit[0][0], fit[0][1])
	p_value_4b  = stats.chi2.sf(chi_background_4b[1], N_dof_4b)
	p_value_b.append(p_value_4b)
	chi_b.append(chi_background_4b[3])


#Numpy histogram from chi_b distribution
bin_heights_chi, bin_edges_chi  = np.histogram(chi_b, bins = 30)           # bin_heights and bin_edges are numpy arrays.
bincentres_chi = 0.5*(bin_edges_chi[1:]+bin_edges_chi[:-1])     #calculate the bin centres

#Fitting Gaussian distribution of reduced Chi^2
def Gauss_chi_distrib(x, A, mu, sig):
	return A  / (np.sqrt(2 * sp.pi) * sig) * np.exp(-(x-mu)**2 / (2 * sig**2))

#guesses
guess_A = 100
guess_mu = 1.05
guess_sig = 1
p0_chi = [guess_A, guess_mu, guess_sig] #initial parameters array

#fit
fit_chi  = curve_fit(Gauss_chi_distrib, bincentres_chi, bin_heights_chi,  p0 = p0_chi)  #fitted on values for E < 120 GeV (excluding possible signal)

#Params
mu = fit_chi[0][1]
mu_sig = np.sqrt(fit_chi[1][1][1])
sig = fit_chi[0][2]
sig_sig = np.sqrt(fit_chi[1][2][2])

#print("Params for chi2 distrib:", fit_chi[0], fit_chi[1])
print(f"Gauss_chi_distrib: mu ={mu : .4f} +- {mu_sig: .4f} sigma =  {sig: .4f} +- {sig_sig: .4f}")

chi_distrib_fit = Gauss_chi_distrib(bincentres_chi, *fit_chi[0])

#b)
plt.hist(chi_b, bins = 30)
plt.plot(bincentres_chi, chi_distrib_fit, color = "red", label = f'Gaussian($\mu = {round(mu,2)} \pm {mu_sig: .2f}, \sigma = {round(sig,2)} \pm {round(sig_sig,2)}$)')
plt.errorbar(bincentres_chi, bin_heights_chi, yerr = np.sqrt(bin_heights_chi), fmt = ',' , capsize  = 4) #plot errorbar
plt.ylabel("Number of values")
plt.xlabel("Reduced Chi-squared")
plt.legend()
plt.savefig("4b_Chi2_distrib.png")
plt.show()

	#################### Part c) #######################

p_value_c = []
amplitudes = []

for i in tqdm(range(49)):
	#Lists to append in
	N_signal_l = []
	p_value_4c_l = []

	for j in range(500):
		N_signal = int(random.uniform(0 + 10*i,10 + 10*i))
		N_dof_4c = int(N_bins - 2)

		vals_4c = higgs.generate_data(N_signal)

		#Find Chi-squared value each loop and p-value
		chi_background_4c = higgs.get_B_chi(vals_4c, (104,155), N_bins, fit[0][0], fit[0][1])
		p_value_4c  = stats.chi2.sf(chi_background_4c[1], N_dof_4c)

		#Append to lists that we can then average the values
		N_signal_l.append(N_signal)
		p_value_4c_l.append(p_value_4c)

	#Arrays of lists
	N_signal_arr = np.array(N_signal_l)
	p_value_4c_arr = np.array(p_value_4c_l)

	#Average the intermediate results
	avg_N_signal = np.mean(N_signal_arr)
	avg_p_value_4c = np.mean(p_value_4c_arr)

	#Append averaged values
	amplitudes.append(avg_N_signal)
	p_value_c.append(avg_p_value_4c)

	#print(i / N_loop * 100, "%")
sig_amp_005 = min(amplitudes, key=lambda x:abs(x-0.05))                ### It gives: 4.428 idk why, but you can see on graph

print("Signal amplitude that gives p-value = 0.05 is",sig_amp_005)
#c)
plt.scatter(amplitudes, p_value_c)
plt.axhline(y=0.05, color='r', linestyle='-')
plt.xlabel("Signal amplitude")
plt.ylabel("p-value")
plt.savefig("4c_p-value_distrib.png")
plt.show()



################### 5. Signal estimation ###########################################

#Chi-squared value - background + signal

#a)

#print("Number of bincentres:",len(bincentres))
chi_signal = higgs.get_SB_chi(vals, (104,155), len(bincentres), fit[0][0], fit[0][1], 125, 1.5, 700)
N_dof_s = len(bincentres) - 5
print(chi_signal,". N_dof = ", N_dof_s)
# b)

p_value_s = stats.chi2.sf(chi_signal[1], N_dof_s)
print("Background + Gaussian fit over signal p-value = ", p_value_s)

#Residuals for Gaussian to be fitted
#Fitting exp + Gauss function
def signal_fit(x, mu, sig, signal_amp):
	return  background(x, *fit[0]) + signal_amp/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

#guesses

guess_mu = 125
guess_sig = 1.5
guess_signal_amp = 700
p0_all = [guess_mu, guess_sig, guess_signal_amp] #initial parameters array


fit_all = curve_fit(signal_fit, bincentres, bin_heights, p0 = p0_all)
#Params
mu = fit_all[0][0]
mu_sig = np.sqrt(fit_all[1][0][0])
sig = fit_all[0][1]
sig_sig = np.sqrt(fit_all[1][1][1])
amp = fit_all[0][2]
amp_sig = np.sqrt(fit_all[1][2][2])

print(f"Exponent + Gauss function parameters: mu ={mu: .4f} +- {mu_sig: .4f}, sigma_gauss =  {mu: .2f} +- {mu_sig: .2f}, sig_amp = {amp: .2f} +- {amp_sig: .2f}")

exp_gauss_fit = signal_fit(bincentres, *fit_all[0])
bin_heights, bin_edges  = np.histogram(vals, range = [104, 155], bins = N_bins) # bin_heights and bin_edges are numpy arrays.
bincentres = 0.5*(bin_edges[1:]+bin_edges[:-1]) #calculate the bin centres

plt.errorbar(bincentres[0:N_bins], bin_heights, yerr = np.sqrt(bin_heights), fmt = '.' , capsize  = 3, color = 'black') #plot errorbar
plt.ylabel("Number of events")
plt.xlabel("$m_{\gamma \gamma }$ [GeV]")
plt.plot(bincentres, exp_gauss_fit, color = "cyan", label = "Signal distribution ($A_{signal}$ " f'= $ {700} \pm {100}, \mu = {125.2} \pm {0.3}, \sigma = {125.2} \pm {0.2}$)')
plt.plot(bincentres, exponential_fit, color = "red", label = f'Background-only distribution ($A = {56340}\pm{50}, \lambda = {30.02}\pm{0.04}$)')
plt.legend(loc=1, prop={'size': 7.5})
plt.savefig("Signal_fit.png")

plt.show()

#c)

chi_unknown_signal = []
sig_mass_pos = []
N_loop = 10000
for i in range (N_loop):
	#Each loop different signal mass position
	mass_pos = random.uniform(104,155)
	#Finding Chi-squared
	chi_signal_unknown = higgs.get_SB_chi(vals, (104,155), len(bincentres), fit[0][0], fit[0][1], mass_pos, 1.5, 700)

	chi_unknown_signal.append(chi_signal_unknown[3])
	sig_mass_pos.append(mass_pos)
	
	print(i / N_loop * 100, "%")

plt.scatter(sig_mass_pos, chi_unknown_signal)
plt.xlabel("Mass of signal [GeV]")
plt.ylabel("Reduced chi-squared value")
plt.savefig("chi_unknown_signal_5c.png")
plt.show()








