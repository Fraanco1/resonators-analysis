import warnings
import numpy as np
import matplotlib.pyplot as plt

def Watt2dBm(x):
	'''
	converts from units of watts to dBm
	'''
	return 10.*np.log10(x*1000.)
	
def dBm2Watt(x):
	'''
	converts from units of watts to dBm
	'''
	return 10**(x/10.) /1000.	

class plotting(object):
	'''
	some helper functions for plotting
	'''
	def plotall(self,fq,latex=True,suptitle=None):
		real = self.z_data_raw.real
		imag = self.z_data_raw.imag
		real2 = self.z_data_sim.real
		imag2 = self.z_data_sim.imag
		fig,ax=plt.subplots(1,3,figsize=(17,5))
		ax[0].plot(real,imag,label='rawdata')
		ax[0].plot(real2,imag2,label='fit')
		ax[0].set_xlabel('Re(S21)')
		ax[0].set_ylabel('Im(S21)')
		ax[0].legend()
		ax[1].plot(self.f_data/1e9,np.absolute(self.z_data_raw),label='rawdata')
		ax[1].plot(self.f_data/1e9,np.absolute(self.z_data_sim),label='fit')
		results1=self.fitresults
		fr1=results1['fr']/1e9
		fr1_err=results1['fr_err']/1e9
		Qi1=results1['Qi_dia_corr']
		Qi1_err=results1['Qi_dia_corr_err']
		Qc1=results1['absQc']
		Qc1_err=results1['absQc_err']
		Ql1=results1['Ql']
		Ql1_err=results1['Ql_err']
		phi0=results1['phi0']
		phi0_err=results1['phi0_err']
		a=results1['a']
		alpha=results1['alpha']
		delay=results1['delay']

	
		ax[1].axvline(x=fr1,color='g',linestyle='--',label='fr_fit')
	

		if latex:
			ax[1].set_title(r"$S_{21}=a e^{i \alpha} e^{-2\pi i f \tau} (1-\frac{(Q_l/|Q_c|)e^{i\phi}}{1+2iQ_l(f/f_r-1)})$")
		if suptitle!=None:
			fig.suptitle(suptitle)
		ax[1].text(0.6,0.4,'fr = %.3f GHz +/- %.3f MHz' %(fr1,fr1_err),transform=ax[1].transAxes)
		ax[1].text(0.6,0.35,'Qi = %.3f +/- %.3f' %(Qi1,Qi1_err),transform=ax[1].transAxes)
		ax[1].text(0.6,0.3,'Qc = %.3f +/- %.3f' %(Qc1,Qc1_err),transform=ax[1].transAxes)
		ax[1].text(0.6,0.25,'Ql = %.3f +/- %.3f' %(Ql1,Ql1_err),transform=ax[1].transAxes)
		ax[1].text(0.6,0.20,'phi0 = %.3f +/- %.3f' %(phi0,phi0_err),transform=ax[1].transAxes)
		ax[1].text(0.6,0.10,'a = %.3f' %(a),transform=ax[1].transAxes)
		ax[1].text(0.6,0.05,'alpha = %.3f' %(alpha),transform=ax[1].transAxes)
		ax[1].text(0.6,0.00,'delay = %.7f ns' %(delay),transform=ax[1].transAxes)

		ax[1].text(0.6,0.15,'fr_pred = %.3f GHz' %(fq),transform=ax[1].transAxes)

		ax[1].set_xlabel('f (GHz)')
		ax[1].set_ylabel('S21')
		ax[1].legend()
		ax[2].plot(self.f_data/1e9,np.angle(self.z_data_raw),label='rawdata')
		ax[2].plot(self.f_data/1e9,np.angle(self.z_data_sim),label='fit')
		ax[2].set_xlabel('f (GHz)')
		ax[2].set_ylabel('arg(S21)')
		ax[2].legend()
		plt.show()

		
		

	def plotall2(self):
		real = self.z_data_raw.real
		imag = self.z_data_raw.imag
		real2 = self.z_data_sim.real
		imag2 = self.z_data_sim.imag
		plt.figure(figsize=(10,10))
		plt.plot(real,imag,label='rawdata')
		plt.plot(real2,imag2,label='fit')
		plt.xlabel('Re(S21)')
		plt.ylabel('Im(S21)')
		plt.legend()
		plt.show()
		plt.figure(figsize=(10,10))
		plt.plot(self.f_data*1e-9,np.absolute(self.z_data_raw),label='rawdata')
		plt.plot(self.f_data*1e-9,np.absolute(self.z_data_sim),label='fit')
		plt.xlabel('f (GHz)')
		plt.ylabel('|S21|')
		plt.legend()
		plt.show()
		plt.figure(figsize=(10,10))
		plt.plot(self.f_data*1e-9,np.angle(self.z_data_raw),label='rawdata')
		plt.plot(self.f_data*1e-9,np.angle(self.z_data_sim),label='fit')
		plt.xlabel('f (GHz)')
		plt.ylabel('arg(|S21|)')
		plt.legend()
		plt.show()
		
	def plotcalibrateddata(self):
		real = self.z_data.real
		imag = self.z_data.imag
		plt.subplot(221)
		plt.plot(real,imag,label='rawdata')
		plt.xlabel('Re(S21)')
		plt.ylabel('Im(S21)')
		plt.legend()
		plt.subplot(222)
		plt.plot(self.f_data*1e-9,np.absolute(self.z_data),label='rawdata')
		plt.xlabel('f (GHz)')
		plt.ylabel('|S21|')
		plt.legend()
		plt.subplot(223)
		plt.plot(self.f_data*1e-9,np.angle(self.z_data),label='rawdata')
		plt.xlabel('f (GHz)')
		plt.ylabel('arg(|S21|)')
		plt.legend()
		plt.show()
		
	def plotrawdata(self):
		real = self.z_data_raw.real
		imag = self.z_data_raw.imag
		plt.subplot(221)
		plt.plot(real,imag,label='rawdata')
		plt.xlabel('Re(S21)')
		plt.ylabel('Im(S21)')
		plt.legend()
		plt.subplot(222)
		plt.plot(self.f_data*1e-9,np.absolute(self.z_data_raw),label='rawdata')
		plt.xlabel('f (GHz)')
		plt.ylabel('|S21|')
		plt.legend()
		plt.subplot(223)
		plt.plot(self.f_data*1e-9,np.angle(self.z_data_raw),label='rawdata')
		plt.xlabel('f (GHz)')
		plt.ylabel('arg(|S21|)')
		plt.legend()
		plt.show()

class save_load(object):
	'''
	procedures for loading and saving data used by other classes
	'''
	def _ConvToCompl(self,x,y,dtype):
		'''
		dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
		'''
		if dtype=='realimag':
			return x+1j*y
		elif dtype=='linmagphaserad':
			return x*np.exp(1j*y)
		elif dtype=='dBmagphaserad':
			return 10**(x/20.)*np.exp(1j*y)
		elif dtype=='linmagphasedeg':
			return x*np.exp(1j*y/180.*np.pi)
		elif dtype=='dBmagphasedeg':
			return 10**(x/20.)*np.exp(1j*y/180.*np.pi)	 
		else: warnings.warn("Undefined input type! Use 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg' or 'linmagphasedeg'.", SyntaxWarning)
	
	def add_data(self,f_data,z_data):
		self.f_data = np.array(f_data)
		self.z_data_raw = np.array(z_data)
		
	def cut_data(self,f1,f2):
		def findpos(f_data,val):
			pos = 0
			for i in range(len(f_data)):
				if f_data[i]<val: pos=i
			return pos
		pos1 = findpos(self.f_data,f1)
		pos2 = findpos(self.f_data,f2)
		self.f_data = self.f_data[pos1:pos2]
		self.z_data_raw = self.z_data_raw[pos1:pos2]
		
	def add_fromtxt(self,fname,dtype,header_rows,usecols=(0,1,2),fdata_unit=1.,delimiter=None):
		'''
		dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
		'''
		data = np.loadtxt(fname,usecols=usecols,skiprows=header_rows,delimiter=delimiter)
		self.f_data = data[:,0]*fdata_unit
		self.z_data_raw = self._ConvToCompl(data[:,1],data[:,2],dtype=dtype)
		
	def add_fromhdf():
		pass
	
	def add_froms2p(self,fname,y1_col,y2_col,dtype,fdata_unit=1.,delimiter=None):
		'''
		dtype = 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg', 'linmagphasedeg'
		'''
		if dtype == 'dBmagphasedeg' or dtype == 'linmagphasedeg':
			phase_conversion = 1./180.*np.pi
		else: 
			phase_conversion = 1.
		f = open(fname)
		lines = f.readlines()
		f.close()
		z_data_raw = []
		f_data = []
		if dtype=='realimag':
			for line in lines:
				if ((line!="\n") and (line[0]!="#") and (line[0]!="!")) :
					lineinfo = line.split(delimiter)
					f_data.append(float(lineinfo[0])*fdata_unit)
					z_data_raw.append(np.complex(float(lineinfo[y1_col]),float(lineinfo[y2_col])))
		elif dtype=='linmagphaserad' or dtype=='linmagphasedeg':
			for line in lines:
				if ((line!="\n") and (line[0]!="#") and (line[0]!="!") and (line[0]!="M") and (line[0]!="P")):
					lineinfo = line.split(delimiter)
					f_data.append(float(lineinfo[0])*fdata_unit)
					z_data_raw.append(float(lineinfo[y1_col])*np.exp( np.complex(0.,phase_conversion*float(lineinfo[y2_col]))))
		elif dtype=='dBmagphaserad' or dtype=='dBmagphasedeg':
			for line in lines:
				if ((line!="\n") and (line[0]!="#") and (line[0]!="!") and (line[0]!="M") and (line[0]!="P")):
					lineinfo = line.split(delimiter)
					f_data.append(float(lineinfo[0])*fdata_unit)
					linamp = 10**(float(lineinfo[y1_col])/20.)
					z_data_raw.append(linamp*np.exp( np.complex(0.,phase_conversion*float(lineinfo[y2_col]))))
		else:
			warnings.warn("Undefined input type! Use 'realimag', 'dBmagphaserad', 'linmagphaserad', 'dBmagphasedeg' or 'linmagphasedeg'.", SyntaxWarning)
		self.f_data = np.array(f_data)
		self.z_data_raw = np.array(z_data_raw)
		
	def save_fitresults(self,fname):
		pass
	


