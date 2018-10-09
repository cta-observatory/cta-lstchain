#!/usr/bin/env python3    
from astropy.io import fits
from scipy.stats import norm
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import math as m
from ctapipe.instrument import OpticsDescription
import astropy.units as u
import ctapipe.coordinates as c
import matplotlib as mpl
import Disp
import sys
import pandas as pd
from astropy.table import Table
import seaborn as sns

def plot_features(data):


    #Energy distribution
    plt.subplot(331)
    plt.hist(data[data['hadroness']<1]['mcEnergy'],histtype=u'step',bins=100,label="Gammas")
    plt.hist(data[data['hadroness']>0]['mcEnergy'],histtype=u'step',bins=100,label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"$log_{10}E$(GeV)")
    plt.legend()

    #Disp distribution
    plt.subplot(332)
    plt.hist(data[data['hadroness']<1]['disp'],histtype=u'step',bins=100,label="Gammas")
    plt.hist(data[data['hadroness']>0]['disp'],histtype=u'step',bins=100,label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"Disp (m)")

    #Intensity distribution
    plt.subplot(333)
    plt.hist(data[data['hadroness']<1]['intensity'],histtype=u'step',bins=100,label="Gammas")
    plt.hist(data[data['hadroness']>0]['intensity'],histtype=u'step',bins=100,label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"$log_{10}Intensity$")

    dataforwl = data[data['intensity']>np.log10(200)]
    #Width distribution
    plt.subplot(334)
    plt.hist(dataforwl[dataforwl['hadroness']<1]['width'],histtype=u'step',bins=100,label="Gammas")
    plt.hist(dataforwl[dataforwl['hadroness']>0]['width'],histtype=u'step',bins=100,label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"Width (º)")

    #Length distribution
    plt.subplot(335)
    plt.hist(dataforwl[dataforwl['hadroness']<1]['length'],histtype=u'step',bins=100,label="Gammas")
    plt.hist(dataforwl[dataforwl['hadroness']>0]['length'],histtype=u'step',bins=100,label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"Length (º)")

    #r distribution
    plt.subplot(336)
    plt.hist(data[data['hadroness']<1]['r'],histtype=u'step',bins=100,label="Gammas")
    plt.hist(data[data['hadroness']>0]['r'],histtype=u'step',bins=100,label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"r (m)")

    #psi distribution

    plt.subplot(337)
    plt.hist(data[data['hadroness']<1]['psi'],histtype=u'step',bins=100,label="Gammas")
    plt.hist(data[data['hadroness']>0]['psi'],histtype=u'step',bins=100,label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"psi angle(rad)")
    
    #psi distribution

    plt.subplot(338)
    plt.hist(data[data['hadroness']<1]['phi'],histtype=u'step',bins=100,label="Gammas")
    plt.hist(data[data['hadroness']>0]['phi'],histtype=u'step',bins=100,label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"phi angle(m)")
    
    #Time gradient

    plt.subplot(339)
    plt.hist(data[data['hadroness']<1]['time_gradient'],histtype=u'step',bins=100,label="Gammas")
    plt.hist(data[data['hadroness']>0]['time_gradient'],histtype=u'step',bins=100,label="Protons")
    plt.ylabel(r'# of events',fontsize=15)
    plt.xlabel(r"Time gradient")

def plot_E(test):
    
    plt.subplot(221)
    difE = ((test['mcEnergy']-test['Erec'])*np.log(10))
    section = difE[abs(difE) < 1.]
    mu,sigma = norm.fit(section)
    print(mu,sigma)
    n, bins, patches = plt.hist(difE,100,density=1,alpha=0.75)
    y = norm.pdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('$(log_{10}(E_{test})-log_{10}(E_{rec}))*log_{N}(10)$',fontsize=10)
    plt.figtext(0.15,0.7,'Mean: '+str(round(mu,4)),fontsize=10)
    plt.figtext(0.15,0.65,'Std: '+str(round(sigma,4)),fontsize=10)
    
    plt.subplot(222)
    hE = plt.hist2d(test['mcEnergy'],test['Erec'],bins=100)
    plt.colorbar(hE[3])
    plt.xlabel('$log_{10}E_{test}$',fontsize=15)
    plt.ylabel('$log_{10}E_{rec}$',fontsize=15)
    plt.plot(test['mcEnergy'],test['mcEnergy'],"-",color='red')

    #Plot a profile
    subplot = plt.subplot(223)
    means_result = scipy.stats.binned_statistic(test['mcEnergy'], [difE, difE**2], bins=50, 
                                                range=(1,6), statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    
    #fig = plt.figure()
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[2, 1],subplot_spec=subplot)
    ax0 = plt.subplot(gs[0])
    plot0 = ax0.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='.')
    plt.ylabel('$(log_{10}(E_{test})-log_{10}(E_{rec}))*log_{N}(10)$',fontsize=10)
    
    ax1 = plt.subplot(gs[1], sharex = ax0)
    plot1 = ax1.plot(bin_centers, standard_deviations,marker='+',linestyle='None')
    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    plt.ylabel("Std",fontsize=10)
    plt.xlabel('$log_{10}E_{test}$',fontsize=10)

    plt.subplots_adjust(hspace=.0)
    
def plot_Disp(test):

    plt.subplot(221)
    difD = ((test['disp']-test['Disprec'])/test['disp'])
    section = difD[abs(difD) < 0.25]
    mu,sigma = norm.fit(section)
    print(mu,sigma)
    n,bins,patches = plt.hist(difD,100,density=1,alpha=0.75,range=[-2,1.5])
    y = norm.pdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('$\\frac{Disp_{test}-Disp_{rec}}{Disp_{test}}$',fontsize=15)
    plt.figtext(0.15,0.7,'Mean: '+str(round(mu,4)),fontsize=12)
    plt.figtext(0.15,0.65,'Std: '+str(round(sigma,4)),fontsize=12)
                
    plt.subplot(222)
    hD = plt.hist2d(test['disp'],test['Disprec'],bins=100,range=([0,1.1],[0,1.1]))
    plt.colorbar(hD[3])
    plt.xlabel('$Disp_{test}$',fontsize=15)
    plt.ylabel('$Disp_{rec}$',fontsize=15)
    plt.plot(test['disp'],test['disp'],"-",color='red')   
 
    plt.subplot(223)
    #plt.hist(test['theta2_true'],bins=100,range=[0,0.05],histtype=u'step',label=r'With Hillas Disp')
    plt.hist(test['theta2_reco'],bins=100,range=[0,0.01],histtype=u'step',label=r'With Reconstructed Disp')
    plt.legend()
    plt.xlabel(r'$\theta^{2}(º)$',fontsize=15)
    plt.ylabel(r'# of events',fontsize=15)
    
    '''
    plt.subplot(224)
    plt.hist(test[test['hadroness']<1]['theta2_true'],bins=50,range=[0,20],histtype=u'step',label=r'With Hillas Disp')
    plt.hist(test[test['hadroness']<1]['theta2_reco'],bins=50,range=[0,20],histtype=u'step',label=r'With Reconstructed Disp')
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r'$\theta^{2}$',fontsize=15)
    plt.ylabel(r'# of Only Gamma events',fontsize=15)
    '''

def plot_pos(data,Energy_cut):
    
    data = data[data['Erec']>Energy_cut]
    #True position
    trueX = data[data['hadroness']==0]['SrcX']
    trueY = data[data['hadroness']==0]['SrcY']
    trueXprot = data[data['hadroness']==1]['SrcX']
    trueYprot = data[data['hadroness']==1]['SrcY']
    #Reconstructed position
    recX = data[data['hadroness']==0]['SrcXrec']
    recY = data[data['hadroness']==0]['SrcYrec']
    recXprot = data[data['hadroness']==1]['SrcXrec']
    recYprot = data[data['hadroness']==1]['SrcYrec']

    plt.subplot(221)
    plt.hist2d(trueXprot,trueYprot,bins=100,label="Protons")
    plt.colorbar()
    plt.title("True position Protons")
    plt.xlabel("x(m)")
    plt.ylabel("y (m)")
    plt.subplot(222)
    plt.hist2d(trueX,trueY,bins=100,label="Gammas",range=np.array([(-1, 1), (-1, 1)]))
    plt.colorbar()
    plt.title("True position Gammas")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.subplot(223)
    plt.colorbar()
    plt.hist2d(recXprot,recYprot,bins=100,label="Protons",range=np.array([(-1, 1), (-1, 1)]))
    plt.title("Reconstructed position Protons")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.subplot(224)
    plt.colorbar()
    plt.hist2d(recX,recY,bins=100,label="Gammas",range=np.array([(-1, 1), (-1, 1)]))
    plt.title("Reconstructed position Gammas ")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")



def GHsep(data,features,Energy_cut):
    
    data = data[data['Erec']>Energy_cut]

    #Select the training/test events
    data['is_train'] = np.random.uniform(0,1,len(data))<= 0.50
    
    #Build new train/test sets:
    train,test = data[data['is_train']==True],data[data['is_train']==False]
    
    #features = ['intensity','r','width','length','w/l','phi','psi','impact','mcXmax','mcHfirst']
    

    #Classify Gamma/Hadron
    clf = RandomForestClassifier(max_depth = 50,
                             n_jobs=10,
                             random_state=4,
                             n_estimators=500)

    clf.fit(train[features],train['hadroness'])
    result = clf.predict(test[features])
    test['hadroreco'] = result
    #Print some quantity results:
    print("TOTAL NUMBER OF EVENTS FOR TRAIN: ",train.shape[0])
    print("Number of gamma events for train: ",train[train['hadroness']==0].shape[0])
    print("Number of proton events for train: ",train[train['hadroness']==1].shape[0])
    print("\n")
    print("TOTAL NUMBER OF EVENTS TESTED: ",test.shape[0])
    print("Number of gamma events tested: ",test[test['hadroness']==0].shape[0])
    print("Number of proton events tested: ",test[test['hadroness']==1].shape[0])
    print("Number of events classified as gammas: ",test[test['hadroreco']==0].shape[0])
    print("Number of events classified as protons: ",test[test['hadroreco']==1].shape[0])
    
    return test, clf


    
def plot_importances(clf,features):

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature importances (gini index)")
    for f in range(len(features)):
        print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

    ordered_features=[]
    for index in indices:
        ordered_features=ordered_features+[features[index]]
    
    plt.title("Feature importances for G/H separation",fontsize=15)
    plt.bar(range(len(features)), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(features)), ordered_features)
    plt.xlim([-1, len(features)])

def plot_ROC(data, features, Energy_cut):
    # Plot ROC curve:
    check = clf.predict_proba(data[features])[0:,1]
    accuracy = accuracy_score(data['hadroness'], data['hadroreco'])
    print(accuracy)
    
    fpr_rf, tpr_rf, _ = roc_curve(data['hadroness'], check)
    
    plt.plot(fpr_rf, tpr_rf, label='Energy Cut: '+'%.3f'%(pow(10,Energy_cut)/1000)+' TeV')
    plt.xlabel('False positive rate',fontsize=15)
    plt.ylabel('True positive rate',fontsize=15)
    plt.legend(loc='best')
 
# ======================== #                                                                  
# Main routine entry point #                                                                   
# ======================== #                                                                    
if __name__ == '__main__': 

    DATA_PATH = "/scratch/bernardos/LST1/Events/"
    data = pd.read_hdf(DATA_PATH+"recoevents_point.hdf5","test")

    Energy_cut=2.699
    plot_pos(data,Energy_cut)
    plt.show()

    
    plot_features(data)
    plt.show()
    
    Energies = [-1,2.,2.398,2.699,2.875,3.,4]
    features = ['Erec','Disprec','intensity','time_gradient','width','length','w/l']
    '''
    for Energy_cut in Energies:
        result,clf = GHsep(data,features,Energy_cut)
        plot_ROC(result,features,Energy_cut)


    plt.show()
    '''
    
    Energies = [2.699]
    theta2_cut = 0.01
    #Energies = [-1.]
    for Energy_cut in Energies:
    
        result,clf = GHsep(data,features,Energy_cut)
        plt.subplot(121)
        plot_importances(clf,features)
        plt.subplot(122)
        plot_ROC(result,features,Energy_cut)
        plt.show()
        
        gammas = result[result['hadroreco']<1]
        gammas = result[result['theta2_reco']<theta2_cut]
        plot_E(gammas)
        plt.show()
        plot_Disp(gammas)
        plt.show()
        plot_pos(result,Energy_cut)
        plt.show()
    
    
