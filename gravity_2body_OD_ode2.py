import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def a_alpha(rt, t, p):
	r1,v1,theta1,omega1,r2,v2,theta2,omega2=rt
	m1,m2,G=p
	c12=np.cos(theta1-theta2)
	s12=np.sin(theta1-theta2)
	bot=((r1**2)+(r2**2)-2*r1*r2*c12)**(-3/2)
	A1=-2*G*m2*(r1-r2*c12)*bot
	A2=-2*G*m1*(r2-r1*c12)*bot
	B1=-2*G*m2*(r2/r1)*s12*bot
	B2=2*G*m1*(r1/r2)*s12*bot
	return [v1,r1*(omega1**2)+A1,omega1,-2*(v1/r1)*omega1+B1,v2,r2*(omega2**2)+A2,omega2,-2*(v2/r2)*omega2+B2]

rng=np.random.default_rng(9248282)

mmin=1
mmax=2
sfac=1/100
rmin=2
rmax=2
vmin=0
vmax=0
omin=-0.5
omax=0.5

m1=(mmax-mmin)*rng.random()+mmin
m2=(mmax-mmin)*rng.random()+mmin
ro1=(rmax-rmin)*rng.random()+rmin
ro2=(rmax-rmin)*rng.random()+rmin
vo1=(vmax-vmin)*rng.random()+vmin
vo2=(vmax-vmin)*rng.random()+vmin
theta1=360*rng.random()
theta2=360*rng.random()
omega1=(omax-omin)*rng.random()+omin
omega2=(omax-omin)*rng.random()+omin
#G=6.674e-11
G=1

cnvrt=np.pi/180
theta1*=cnvrt
theta2*=cnvrt

p=[m1,m2,G]
rt=[ro1,vo1,theta1,omega1,ro2,vo2,theta2,omega2]

tf = 60 
nfps = 60
nframes = tf * nfps
t = np.linspace(0, tf, nframes)

rth = odeint(a_alpha, rt, t, args = (p,))

rd1=rth[:,0]
th1=rth[:,2]
rd2=rth[:,4]
th2=rth[:,6]

x1=rd1*np.cos(th1)
y1=rd1*np.sin(th1)
x2=rd2*np.cos(th2)
y2=rd2*np.sin(th2)

if min(x1)<min(x2):
	xmin=min(x1)-2
else:
	xmin=min(x2)-2
if max(x1)>max(x2):
	xmax=max(x1)+2
else:
	xmax=max(x2)+2
if min(y1)<min(y2):
	ymin=min(y1)-2
else:
	ymin=min(y2)-2
if max(y1)>max(y2):
	ymax=max(y1)+2
else:
	ymax=max(y2)+2

dx=xmax-xmin
dy=ymax-ymin
dr=np.sqrt(dx**2+dy**2)
maxmr=sfac*dr
mass_list=[m1,m2]
massmax=max(mass_list)
massmax=1/mmax
f=maxmr*massmax
mr=[f*m1,f*m2]
shift=max(mr)

xmax=xmax+2*shift
xmin=xmin-2*shift
ymax=ymax+2*shift
ymin=ymin-2*shift

v1=rth[:,1]
w1=rth[:,3]
v2=rth[:,5]
w2=rth[:,7]

ke1=0.5*m1*((v1**2)+((rd1*w1)**2))
ke2=0.5*m2*((v2**2)+((rd2*w2)**2))
ke=ke1+ke2
r12=np.sqrt((rd1**2)+(rd2**2)-2*rd1*rd2*np.cos(th1-th2))
pe=-2*G*m1*m2/r12
E=ke+pe
Emax=abs(max(E))
ke/=Emax
pe/=Emax
E/=Emax
Emax=max(E)
ke-=Emax
pe-=Emax
E-=Emax

fig, a=plt.subplots()
fig.tight_layout()

def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((x1[frame],y1[frame]),radius=mr[0],fc='r')
	plt.gca().add_patch(circle)
	circle=plt.Circle((x2[frame],y2[frame]),radius=mr[1],fc='r')
	plt.gca().add_patch(circle)
	plt.title("2 Body Orbital Dynamics")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(t[0:frame],ke[0:frame],'r',lw=1)
	plt.plot(t[0:frame],pe[0:frame],'b',lw=1)
	plt.plot(t[0:frame],E[0:frame],'g',lw=1)
	plt.xlim([0,tf])
	plt.title("Energy (Rescaled and Shifted)")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')


ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('gravity_2body_ode_wgphs.mp4', writer=writervideo)

plt.show()


