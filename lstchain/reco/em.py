"""This module contain functions for the calculation of Hillas Parameters using
s expectation-maximization algorithm without cleaning

Usage:

"import em"

"""
import numpy as np
import math
from ctapipe.io import event_source
import astropy.units as u
from astropy.coordinates import Angle
from astropy.units import Quantity
from ctapipe.io.containers import HillasParametersContainer
from scipy import ndimage

__all__ = [
    'em_hillas_parameters',
    'HillasParameterizationError',
]

class HillasParameterizationError(RuntimeError):
    pass

def bigaus(x,y,pars):
    xm = pars[0].value
    ym = pars[1].value
    xx = pars[2].value
    yy = pars[3].value
    xy = pars[4].value
    '''
    det = np.sqrt((sigma_xx*sigma_yy - sigma_xy*sigma_xy)**2)
    prod = 0.5*(sigma_yy*(x-mean_x)**2 - 2*sigma_xy*(x-mean_x)*(y-mean_y) + sigma_xx*(y-mean_y)**2)/det

    A = 1/(2*np.pi*np.sqrt(det))
    p = A*np.exp(-prod.value)
    '''
    det = xx*yy - xy*xy;

    det = np.sqrt(pow(det,2));
    prod = 0.5*(yy*pow((x-xm),2) - 2*xy*(x-xm)*(y-ym) + xx*pow((y-ym),2))/det;
    A = 1/(2*np.pi*np.sqrt(det));
    p = A*np.exp(-prod);

    return p

def initialize_image_pars(image,camera):

    idx_max = image.argmax()
    #Find baricenter of 3 higher pixels in image
    img_sorted = np.argsort(-image)
    pix_high = img_sorted[0]
    pix_mid = img_sorted[1]
    pix_low = img_sorted[2]

    mean_x = (camera.pix_x[pix_high]*image[pix_high]+camera.pix_x[pix_mid]*image[pix_mid]+camera.pix_x[pix_low]*image[pix_low])/(image[pix_high]+image[pix_mid]+image[pix_low])
    mean_y = (camera.pix_y[pix_high]*image[pix_high]+camera.pix_y[pix_mid]*image[pix_mid]+camera.pix_y[pix_low]*image[pix_low])/(image[pix_high]+image[pix_mid]+image[pix_low])

    sigma_xx = 20000 * u.mm * u.mm
    sigma_yy = 20000 *u.mm * u.mm
    sigma_xy = 0 * u.mm * u.mm
    size_shower = image.sum()/2
    size_back = image.sum()/2
    pars = [mean_x,mean_y,sigma_xx,sigma_yy,sigma_xy]
    return pars,size_shower,size_back

def update_pars(shower,camera,total_size):

    size = shower.sum()

    mean_x = (shower*camera.pix_x).sum()/size
    mean_y = (shower*camera.pix_y).sum()/size

    sigma_xx = (shower*camera.pix_x**2).sum()/size - mean_x**2
    sigma_yy = (shower*camera.pix_y**2).sum()/size - mean_y**2
    sigma_xy = (shower*camera.pix_x*camera.pix_y).sum()/size - mean_x*mean_y

    size_shower = size
    size_back = total_size - size

    pars = [mean_x,mean_y,sigma_xx,sigma_yy,sigma_xy]

    return pars,size_shower,size_back

def gaus_estimate(shower,camera,pars,size_shower,size_back):

    pix_area = (camera.pix_area[0]).value
    Pgauss_bin = bigaus(camera.pix_x.value,camera.pix_y.value,pars)*pix_area

    Pgauss = size_shower
    Pback_bin = pix_area/((camera.pix_area).sum()).value
    Pback = size_back
    Pbin = Pgauss*Pgauss_bin + Pback*Pback_bin

    Pbin_gauss = Pgauss_bin*Pgauss/Pbin
    Pbin_back = Pback_bin*Pback/Pbin
    '''
    for i in range(len(Pgauss_bin)):
        print(Pgauss_bin[i],Pgauss,Pback_bin,Pback,Pbin[i],Pbin_gauss[i],Pbin_back[i])
    '''
    n_of_gaus = Pbin_gauss*shower
    n_of_back = Pbin_back*shower

    #Calculate the likelihood function
    loglike = np.sum(shower*np.log(Pgauss*Pgauss_bin + Pback*Pback_bin))
    return n_of_gaus, n_of_back, loglike

def em_loop(image, camera):

    '''
    =============================================================
    Convert units from m to mm because EM prefer big numbers (>1)
    =============================================================
    '''
    camera.pix_x = camera.pix_x.to(u.mm);
    camera.pix_y = camera.pix_y.to(u.mm);
    camera.pix_area = camera.pix_area.to(u.mm*u.mm);
    '''
    ==============================
    Initialize parameters for EM
    ==============================
    '''
    init_pars,size_shower,size_back = initialize_image_pars(image,camera)
    pars = list.copy(init_pars)
    '''
    ===========================================================================
    Loop for EM: It will separate the image in shower and background (img, back)
    ===========================================================================
    '''
    niter = 0
    maxiter = 100
    tol=1e-4
    diff = 10000
    old_loglike=0
    while (diff>tol):
        #for i in range(0,niter):
        img,back,loglike = gaus_estimate(image,camera,pars,size_shower,size_back)
        diff = np.sqrt((loglike-old_loglike)**2)
        old_loglike = loglike
        new_pars,size_shower,size_back = update_pars(img,camera,image.sum())
        pars = new_pars
        if pars[2].value <= 50 or pars[3].value <= 50 or niter > maxiter:
            break;
        niter = niter+1

    '''
    =============================================================
    Convert units back to m
    =============================================================
    '''
    camera.pix_x = camera.pix_x.to(u.m);
    camera.pix_y = camera.pix_y.to(u.m);
    camera.pix_area = camera.pix_area.to(u.m*u.m);

    img = image-back
    img[img<1] = 0
    return pars, img, loglike

def em_hillas_parameters(image, camera):

    try:
        pars, image, loglike = em_loop(image, camera)
        xm = pars[0].value
        ym = pars[1].value
        xx = pars[2].value
        yy = pars[3].value
        xy = pars[4].value

        cog_x = xm
        cog_y = ym
        cog_r = np.linalg.norm([cog_x, cog_y])

        tr = 0.5*(xx+yy)
        det = xx*yy-xy*xy

        disc = tr*tr-det
        if disc < 0:
            print("Negative value found, skip event")
            return None, None, None
        else:
            disc=math.sqrt(disc)

        width = tr-disc

        if width < 0:
            print("Negative value found, skip event")
            return None, None, None

        if xy==0:
            print("Zero value found, skip event")
            return None, None, None

        else:
            width=math.sqrt(width)
            length = math.sqrt(tr+disc)

            psi = math.atan((length*length-xx)/xy)

            size = image.sum()

            phi=math.atan(cog_y/cog_x)

            delta_x = camera.pix_x.to(u.mm).value - cog_x
            delta_y = camera.pix_y.to(u.mm).value - cog_y

            longitudinal = delta_x * np.cos(psi) + delta_y * np.sin(psi)
            m3_long = np.average(longitudinal**3, weights=image)
            skewness_long = m3_long / length**3
            m4_long = np.average(longitudinal**4, weights=image)
            kurtosis_long = m4_long / length**4

        return HillasParametersContainer(
            x=u.Quantity((cog_x*u.mm).to(u.m), u.m),
            y=u.Quantity((cog_y*u.mm).to(u.m), u.m),
            r=u.Quantity((cog_r*u.mm).to(u.m), u.m),
            phi=Angle(phi, unit=u.rad),
            intensity=size,
            length=u.Quantity((length*u.mm).to(u.m), u.m),
            width=u.Quantity((width*u.mm).to(u.m), u.m),
            psi=Angle(psi, unit=u.rad),
            skewness=skewness_long,
            kurtosis=kurtosis_long
        ), image, loglike
    except:
        return None, None, None
