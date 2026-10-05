import warnings
import numpy as np
import scipy.optimize as spopt
from scipy.constants import hbar
from scipy.interpolate import splrep, splev

from resonator_tools.utilities import plotting, save_load, Watt2dBm, dBm2Watt
from resonator_tools.circlefit import circlefit
from resonator_tools.calibration import calibration

##
## z_data_raw denotes the raw data
## z_data denotes the normalized data
##		  
	
class reflection_port(circlefit, save_load, plotting, calibration):
	'''
	normal direct port probed in reflection
	'''
	def __init__(self, f_data=None, z_data_raw=None):
		self.porttype = 'direct'
		self.fitresults = {}
		self.z_data = None
		if f_data is not None:
			self.f_data = np.array(f_data)
		else:
			self.f_data=None
		if z_data_raw is not None:
			self.z_data_raw = np.array(z_data_raw)
		else:
			self.z_data=None
		self.phasefitsmooth = 3
	
	def _S11(self,f,fr,k_c,k_i):
		return ((k_c-k_i)+2j*(f-fr))/((k_c+k_i)-2j*(f-fr))
	
	def get_delay(self,f_data,z_data,delay=None,ignoreslope=True,guess=True):
		maxval = np.max(np.absolute(z_data))
		z_data = z_data/maxval
		A1, A2, A3, A4, fr, Ql = self._fit_skewed_lorentzian(f_data,z_data)
		if self.df_error/fr > 0.0001 or self.dQl_error/Ql>0.1:
			A1 = np.mean(np.absolute(z_data))
			A2 = 0.
			A3 = 0.
			A4 = 0.
			f = splrep(f_data,np.unwrap(np.angle(z_data)),k=5,s=self.phasefitsmooth)
			fr = f_data[np.argmax(np.absolute(splev(f_data,f,der=1)))]
			Ql = 1e4
		if ignoreslope==True:
			A2 = 0.
		else:
			print("WARNING: The ignoreslope option is ignored!")
		if delay is None:
			if guess==True:
				delay = self._guess_delay(f_data,z_data)
			else:
				delay=0.
			delay = self._fit_delay(f_data,z_data,delay,maxiter=500)
		params = [A1, A2, A3, A4, fr, Ql]
		return delay, params 
	
	def do_calibration(self,f_data,z_data,ignoreslope=True,guessdelay=True,fixed_delay=None):
		delay, params = self.get_delay(f_data,z_data,ignoreslope=ignoreslope,guess=guessdelay,delay=fixed_delay)
		z_data = (z_data-params[1]*(f_data-params[4]))*np.exp(2.*1j*np.pi*delay*f_data)
		xc, yc, r0 = self._fit_circle(z_data)
		zc = complex(xc,yc)
		fitparams = self._phase_fit(f_data,self._center(z_data,zc),0.,np.absolute(params[5]),params[4])
		theta, Ql, fr = fitparams
		beta = self._periodic_boundary(theta+np.pi,np.pi)
		offrespoint = complex((xc+r0*np.cos(beta)),(yc+r0*np.sin(beta)))
		alpha = self._periodic_boundary(np.angle(offrespoint)+np.pi,np.pi)
		a = r0 + np.absolute(zc)
		return delay, a, alpha, fr, Ql, params[1], params[4]
	
	def do_normalization(self,f_data,z_data,delay,amp_norm,alpha,A2,frcal):
		return (z_data-A2*(f_data-frcal))/amp_norm*np.exp(1j*(-alpha+2.*np.pi*delay*f_data))
	
	def circlefit(self,f_data,z_data,fr=None,Ql=None,refine_results=False,calc_errors=True):
		if fr is None: fr=f_data[np.argmin(np.absolute(z_data))]
		if Ql is None: Ql=1e6
		xc, yc, r0 = self._fit_circle(z_data,refine_results=refine_results)
		phi0 = -np.arcsin(yc/r0)
		theta0 = self._periodic_boundary(phi0+np.pi,np.pi)
		z_data_corr = self._center(z_data,complex(xc,yc))
		theta0, Ql, fr = self._phase_fit(f_data,z_data_corr,theta0,Ql,fr)
		Qi = Ql/(1.-r0)
		Qc = 1./(1./Ql-1./Qi)
	
		results = {"Qi":Qi,"Qc":Qc,"Ql":Ql,"fr":fr,"theta0":theta0}
	
		p = [fr,Qc,Ql]
		if calc_errors==True:
			chi_square, cov = self._get_cov_fast_directrefl(f_data,z_data,p)
			if cov is not None:
				errors = np.sqrt(np.diagonal(cov))
				fr_err,Qc_err,Ql_err = errors
				dQl = 1./((1./Ql-1./Qc)**2*Ql**2)
				dQc = - 1./((1./Ql-1./Qc)**2*Qc**2)
				Qi_err = np.sqrt((dQl**2*cov[2][2]) + (dQc**2*cov[1][1])+(2*dQl*dQc*cov[2][1]))
				errors = {"Ql_err":Ql_err, "Qc_err":Qc_err, "fr_err":fr_err,"chi_square":chi_square,"Qi_err":Qi_err}
				results.update( errors )
			else:
				print("WARNING: Error calculation failed!")
		else:
			fun2 = lambda x: self._residuals_notch_ideal(x,f_data,z_data)**2
			chi_square = 1./float(len(f_data)-len(p)) * (fun2(p)).sum()
			errors = {"chi_square":chi_square}
			results.update(errors)
	
		return results,xc,yc,r0
		
	def autofit(self,electric_delay=None,fcrop=None):
		if fcrop is None:
			self._fid = np.ones(self.f_data.size,dtype=bool)
		else:
			f1, f2 = fcrop
			self._fid = np.logical_and(self.f_data>=f1,self.f_data<=f2)
		delay, amp_norm, alpha, fr, Ql, A2, frcal =\
				self.do_calibration(self.f_data[self._fid],self.z_data_raw[self._fid],ignoreslope=True,guessdelay=False,fixed_delay=electric_delay)
		self.z_data = self.do_normalization(self.f_data,self.z_data_raw,delay,amp_norm,alpha,A2,frcal)
		self.fitresults = self.circlefit(self.f_data[self._fid],self.z_data[self._fid],fr,Ql,refine_results=False,calc_errors=True)
		self.z_data_sim = A2*(self.f_data-frcal)+self._S11_directrefl(self.f_data,fr=self.fitresults["fr"],Ql=self.fitresults["Ql"],Qc=self.fitresults["Qc"],a=amp_norm,alpha=alpha,delay=delay)
		self.z_data_sim_norm = self._S11_directrefl(self.f_data,fr=self.fitresults["fr"],Ql=self.fitresults["Ql"],Qc=self.fitresults["Qc"],a=1.,alpha=0.,delay=0.)		
		self._delay = delay

	def _S11_directrefl(self,f,fr=10e9,Ql=900,Qc=1000.,a=1.,alpha=0.,delay=.0):
		return a*np.exp(complex(0,alpha))*np.exp(-2j*np.pi*f*delay) * ( 2.*Ql/Qc - 1. + 2j*Ql*(fr-f)/fr ) / ( 1. - 2j*Ql*(fr-f)/fr )	   
		
	def get_single_photon_limit(self,unit='dBm'):
		if self.fitresults!={}:
			fr = self.fitresults['fr']
			k_c = 2*np.pi*fr/self.fitresults['Qc']
			k_i = 2*np.pi*fr/self.fitresults['Qi']
			if unit=='dBm':
				return Watt2dBm(1./(4.*k_c/(2.*np.pi*hbar*fr*(k_c+k_i)**2)))
			elif unit=='watt':
				return 1./(4.*k_c/(2.*np.pi*hbar*fr*(k_c+k_i)**2))
		else:
			warnings.warn('Please perform the fit first',UserWarning)
			return None
		
	def get_photons_in_resonator(self,power,unit='dBm'):
		if self.fitresults!={}:
			if unit=='dBm':
				power = dBm2Watt(power)
			fr = self.fitresults['fr']
			k_c = 2*np.pi*fr/self.fitresults['Qc']
			k_i = 2*np.pi*fr/self.fitresults['Qi']
			return 4.*k_c/(2.*np.pi*hbar*fr*(k_c+k_i)**2) * power
		else:
			warnings.warn('Please perform the fit first',UserWarning)
			return None
	
class notch_port(circlefit, save_load, plotting, calibration):
	'''
	notch type port probed in transmission
	'''
	def __init__(self, f_data=None, z_data_raw=None):
		self.porttype = 'notch'
		self.fitresults = {}
		self.z_data = None
		if f_data is not None:
			self.f_data = np.array(f_data)
		else:
			self.f_data=None
		if z_data_raw is not None:
			self.z_data_raw = np.array(z_data_raw)
		else:
			self.z_data_raw=None
	
	def get_delay(self,f_data,z_data,delay=None,ignoreslope=True,guess=True):
		maxval = np.max(np.absolute(z_data))
		z_data = z_data/maxval
		A1, A2, A3, A4, fr, Ql = self._fit_skewed_lorentzian(f_data,z_data)
		if ignoreslope==True:
			A2 = 0.
		else:
			A2 = 0.
			print("WARNING: The ignoreslope option is ignored!")
		if delay is None:
			if guess==True:
				delay = self._guess_delay(f_data,z_data)
			else:
				delay=0.
		delay = self._fit_delay(f_data,z_data,delay,maxiter=500)
		params = [A1, A2, A3, A4, fr, Ql]
		return delay, params	
	
	def do_calibration(self,f_data,z_data,ignoreslope=True,guessdelay=True,fixed_delay=None, Ql_guess=None, fr_guess=None):
		delay, params = self.get_delay(f_data,z_data,ignoreslope=ignoreslope,guess=guessdelay,delay=fixed_delay)
		z_data = (z_data-params[1]*(f_data-params[4]))*np.exp(2.*1j*np.pi*delay*f_data)
		xc, yc, r0 = self._fit_circle(z_data,refine_results=True)
		zc = complex(xc,yc)
		if Ql_guess is None: Ql_guess=np.absolute(params[5]) 
		if fr_guess is None: fr_guess=params[4] 
		fitparams = self._phase_fit(f_data,self._center(z_data,zc),0.,Ql_guess,fr_guess) 
		theta, Ql, fr = fitparams
		beta = self._periodic_boundary(theta+np.pi,np.pi)
		offrespoint = complex((xc+r0*np.cos(beta)),(yc+r0*np.sin(beta)))
		alpha = np.angle(offrespoint)
		a = np.absolute(offrespoint)
		return delay, a, alpha, fr, Ql, params[1], params[4]
	
	def do_normalization(self,f_data,z_data,delay,amp_norm,alpha,A2,frcal):
		return (z_data-A2*(f_data-frcal))/amp_norm*np.exp(1j*(-alpha+2.*np.pi*delay*f_data))

	def circlefit(self,f_data,z_data,fr=None,Ql=None,refine_results=False,calc_errors=True):
		if fr is None: fr=f_data[np.argmin(np.absolute(z_data))]
		if Ql is None: Ql=1e6
		xc, yc, r0 = self._fit_circle(z_data,refine_results=refine_results)
		phi0 = -np.arcsin(yc/r0)
		theta0 = self._periodic_boundary(phi0+np.pi,np.pi)
		z_data_corr = self._center(z_data,complex(xc,yc))
		theta0, Ql, fr = self._phase_fit(f_data,z_data_corr,theta0,Ql,fr)
		absQc = Ql/(2.*r0)
		complQc = absQc*np.exp(1j*((-1.)*phi0))
		Qc = 1./(1./complQc).real
		Qi_dia_corr = 1./(1./Ql-1./Qc)
		Qi_no_corr = 1./(1./Ql-1./absQc)
	
		results = {"Qi_dia_corr":Qi_dia_corr,"Qi_no_corr":Qi_no_corr,"absQc":absQc,"Qc_dia_corr":Qc,"Ql":Ql,"fr":fr,"theta0":theta0,"phi0":phi0}
	
		p = [fr,absQc,Ql,phi0]
		if calc_errors==True:
			chi_square, cov = self._get_cov_fast_notch(f_data,z_data,p)
			if cov is not None:
				errors = np.sqrt(np.diagonal(cov))
				fr_err,absQc_err,Ql_err,phi0_err = errors
				dQl = 1./((1./Ql-1./absQc)**2*Ql**2)
				dabsQc = - 1./((1./Ql-1./absQc)**2*absQc**2)
				Qi_no_corr_err = np.sqrt((dQl**2*cov[2][2]) + (dabsQc**2*cov[1][1])+(2*dQl*dabsQc*cov[2][1]))
				dQl = 1/((1/Ql-np.cos(phi0)/absQc)**2 *Ql**2)
				dabsQc = -np.cos(phi0)/((1/Ql-np.cos(phi0)/absQc)**2 *absQc**2)
				dphi0 = -np.sin(phi0)/((1/Ql-np.cos(phi0)/absQc)**2 *absQc)
				err1 = ( (dQl**2*cov[2][2]) + (dabsQc**2*cov[1][1]) + (dphi0**2*cov[3][3]) )
				err2 = ( dQl*dabsQc*cov[2][1] + dQl*dphi0*cov[2][3] + dabsQc*dphi0*cov[1][3] )
				Qi_dia_corr_err =  np.sqrt(err1+2*err2)
				errors = {"phi0_err":phi0_err, "Ql_err":Ql_err, "absQc_err":absQc_err, "fr_err":fr_err,"chi_square":chi_square,"Qi_no_corr_err":Qi_no_corr_err,"Qi_dia_corr_err": Qi_dia_corr_err}
				results.update( errors )
			else:
				print("WARNING: Error calculation failed!")
		else:
			fun2 = lambda x: self._residuals_notch_ideal(x,f_data,z_data)**2
			chi_square = 1./float(len(f_data)-len(p)) * (fun2(p)).sum()
			errors = {"chi_square":chi_square}
			results.update(errors)
	
		return results
		
	def autofit(self,electric_delay=None,fcrop=None,Ql_guess=None, fr_guess=None,guessdelay=True):
		if fcrop is None:
			self._fid = np.ones(self.f_data.size,dtype=bool)
		else:
			f1, f2 = fcrop
			self._fid = np.logical_and(self.f_data>=f1,self.f_data<=f2)
		delay, amp_norm, alpha, fr, Ql, A2, frcal =\
				self.do_calibration(self.f_data[self._fid],self.z_data_raw[self._fid],ignoreslope=True,guessdelay=guessdelay,fixed_delay=electric_delay,Ql_guess=Ql_guess, fr_guess=fr_guess)
		self.z_data = self.do_normalization(self.f_data,self.z_data_raw,delay,amp_norm,alpha,A2,frcal)
		self.fitresults = self.circlefit(self.f_data[self._fid],self.z_data[self._fid],fr,Ql,refine_results=True,calc_errors=True)
		self.fitresults["a"] = amp_norm
		self.fitresults["alpha"] = alpha
		self.fitresults["delay"] = delay
		self.z_data_sim = A2*(self.f_data-frcal)+self._S21_notch(self.f_data,fr=self.fitresults["fr"],Ql=self.fitresults["Ql"],Qc=self.fitresults["absQc"],phi=self.fitresults["phi0"],a=amp_norm,alpha=alpha,delay=delay)
		self.z_data_sim_norm = self._S21_notch(self.f_data,fr=self.fitresults["fr"],Ql=self.fitresults["Ql"],Qc=self.fitresults["absQc"],phi=self.fitresults["phi0"],a=1.0,alpha=0.,delay=0.)
		self._delay = delay

	def _S21_notch(self,f,fr=10e9,Ql=900,Qc=1000.,phi=0.,a=1.,alpha=0.,delay=.0):
		return a*np.exp(complex(0,alpha))*np.exp(-2j*np.pi*f*delay)*(1.-Ql/Qc*np.exp(1j*phi)/(1.+2j*Ql*(f-fr)/fr))	 
	
	def get_single_photon_limit(self,unit='dBm',diacorr=True):
		if self.fitresults!={}:
			fr = self.fitresults['fr']
			if diacorr:
				k_c = 2*np.pi*fr/self.fitresults['Qc_dia_corr']
				k_i = 2*np.pi*fr/self.fitresults['Qi_dia_corr']
			else:
				k_c = 2*np.pi*fr/self.fitresults['absQc']
				k_i = 2*np.pi*fr/self.fitresults['Qi_no_corr']
			if unit=='dBm':
				return Watt2dBm(1./(4.*k_c/(2.*np.pi*hbar*fr*(k_c+k_i)**2)))
			elif unit=='watt':
				return 1./(4.*k_c/(2.*np.pi*hbar*fr*(k_c+k_i)**2))				  
		else:
			warnings.warn('Please perform the fit first',UserWarning)
			return None
		
	def get_photons_in_resonator(self,power,unit='dBm',diacorr=True):
		if self.fitresults!={}:
			if unit=='dBm':
				power = dBm2Watt(power)
			fr = self.fitresults['fr']
			if diacorr:
				k_c = 2*np.pi*fr/self.fitresults['Qc_dia_corr']
				k_i = 2*np.pi*fr/self.fitresults['Qi_dia_corr']
			else:
				k_c = 2*np.pi*fr/self.fitresults['absQc']
				k_i = 2*np.pi*fr/self.fitresults['Qi_no_corr']
			return 4.*k_c/(2.*np.pi*hbar*fr*(k_c+k_i)**2) * power
		else:
			warnings.warn('Please perform the fit first',UserWarning)
			return None	  

class transmission_port(circlefit,save_load,plotting):
	def __init__(self,f_data=None,z_data_raw=None):
		self.porttype = 'transm'
		self.fitresults = {}
		if f_data is not None:
			self.f_data = np.array(f_data)
		else:
			self.f_data=None
		if z_data_raw is not None:
			self.z_data_raw = np.array(z_data_raw)
		else:
			self.z_data=None
		
	def _S21(self,f,fr,Ql,A):
		return A**2/(1.+4.*Ql**2*((f-fr)/fr)**2) 
		
	def fit(self):
		self.ampsqr = (np.absolute(self.z_data_raw))**2
		p = [self.f_data[np.argmax(self.ampsqr)],300.,np.amax(self.ampsqr)]
		popt, pcov = spopt.curve_fit(self._S21, self.f_data, self.ampsqr,p)
		errors = np.sqrt(np.diag(pcov))
		self.fitresults = {'fr':popt[0],'fr_err':errors[0],'Ql':popt[1],'Ql_err':errors[1],'Ampsqr':popt[2],'Ampsqr_err':errors[2]}
		self.z_data_sim = self._S21(self.f_data,self.fitresults['fr'],-self.fitresults['Ql'],self.fitresults['Ampsqr'])
	
class resonator(object):
	def __init__(self, ports = {}, comment = None):
		self.comment = comment
		self.port = {}
		self.transm = {}
		if len(ports) > 0:
			for key, pname in iter(ports.items()):
				if pname=='direct':
					self.port.update({key:reflection_port()})
				elif pname=='notch':
					self.port.update({key:notch_port()})
				else:
					warnings.warn("Undefined input type! Use 'direct' or 'notch'.", SyntaxWarning)
		if len(self.port) == 0: warnings.warn("Resonator has no coupling ports!", UserWarning)
			
	def add_port(self,key,pname):
		if pname=='direct':
			self.port.update({key:reflection_port()})
		elif pname=='notch':
			self.port.update({key:notch_port()})
		else:
			warnings.warn("Undefined input type! Use 'direct' or 'notch'.", SyntaxWarning)
		if len(self.port) == 0: warnings.warn("Resonator has no coupling ports!", UserWarning)
			
	def delete_port(self,key):
		del self.port[key]
		if len(self.port) == 0: warnings.warn("Resonator has no coupling ports!", UserWarning)

class batch_processing(object):
	def __init__(self,porttype):
		self.porttype = porttype
		self.results = []
	
	def autofit(self,cal_dataslice = 0):
		pass
	
class coupled_resonators(batch_processing):
	def __init__(self,porttype):
		self.porttype = porttype
		self.results = []