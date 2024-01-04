import os, sys, re, json
import numpy as np
import nkUtilities.plot1D         as pl1
import nkUtilities.load__config   as lcf
import nkUtilities.configSettings as cfs

# ========================================================= #
# ===  validate__Melville_RIprod.py                     === #
# ========================================================= #

def validate__Melville_RIprod():

    pngFile      = "png/validate__Melville_RIprod.png"
    e_, pf_      = 0, 1
    N_Avogadro   = 6.02e23
    mb2cm2       = 1.0e-27

    params       = {
        "EMin":0.0,
        "EMax":20.0,
        "nE":201,
        "Er":13.45,
        "Et":6.4,
        "Gamma":3.97,
        "sigma_r":521.0,
        "EfitLim":5.5,
        "density":5.0,
        "massNumber":226,
        "thickness":1.0,
        "normalizeFactor":80.7,
        "photon.interpolate":"exp_fit"
    }
    
    # ------------------------------------------------- #
    # --- [1] prepare parameters & energy Axis      --- #
    # ------------------------------------------------- #        
    energy  = np.linspace( params["EMin"], params["EMax"], params["nE"] )
    Nt_prod = ( N_Avogadro * params["density"] / params["massNumber"] ) * params["thickness"]

    # ------------------------------------------------- #
    # --- [2] photon flux                           --- #
    # ------------------------------------------------- #
    import nkUtilities.load__pointFile as lpf
    inpFile   = "dat/photon_full_spectrum.dat"
    spectrum  = lpf.load__pointFile( inpFile=inpFile, returnType="point" )

    if   ( params["photon.interpolate"] == "exp_fit" ):
        # - use E > EfitLim
        spectrum  = spectrum[ np.where( spectrum[:,e_] > params["EfitLim"] ) ]
        # - fitting function - #
        def fit__photonSpectrum( eng, c1, c2, c3, c4 ):
            ret   = c1*np.exp( - c2*( eng - c3 ) ) + c4
            return( ret )
        # - fitting
        import scipy.optimize as opt
        p0        = [ 1.0e12, 1.0, 6.4, 1.0e11 ] # - initial coefficients.
        copt,cvar = opt.curve_fit( fit__photonSpectrum, spectrum[:,e_], spectrum[:,pf_], p0=p0 )
        pflux     = fit__photonSpectrum( energy, *copt )
        pflux     = np.where( energy > params["Et"], pflux, 0.0 )   # - 0.0 for E < Et
    elif ( params["photon.interpolate"] == "spline" ):
        import scipy.interpolate as itp
        splineFunc = itp.CubicSpline( spectrum[:,e_], spectrum[:,pf_] )
        pflux      = splineFunc( energy )
        pflux      = np.where( energy > params["Et"], pflux, 0.0 )   # - 0.0 for E < Et
    
    # ------------------------------------------------- #
    # --- [3] calculate x-section (Breit-Wigner)    --- #
    # ------------------------------------------------- #
    amp   = params["sigma_r"] * ( energy / params["Er"] )
    res   = (0.5*params["Gamma"])**2 / ( ( energy-params["Er"] )**2 + (0.5*params["Gamma"])**2 )
    thr   = ( energy - params["Et"] ) / ( params["Er"] - params["Et"] )
    thr   = np.sqrt( np.where( thr >= 0.0, thr, 0.0 ) )
    cs_mb = amp * res * thr
    cs    = cs_mb *mb2cm2
    
    # ------------------------------------------------- #
    # --- [4] yield rate                            --- #
    # ------------------------------------------------- #
    dYield = Nt_prod * cs * pflux
    print( "(min,max) = ( {0:12.5e}, {1:12.5e} )".format( np.min( dYield ), np.max( dYield ) ) )

    # ------------------------------------------------- #
    # --- [5] integrate                             --- #
    # ------------------------------------------------- #
    import scipy.integrate as itg
    Yield = itg.simpson( dYield, energy ) / ( params["EMax"] - params["EMin"] )
    Yield = params["normalizeFactor"] * Yield
    print()
    print( "RI prod :: {0:12.5e} ".format( Yield ) )
    print()

    # ------------------------------------------------- #
    # --- [6] display                               --- #
    # ------------------------------------------------- #
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["FigSize"]        = (4.5,4.5)
    config["plt_position"]   = [ 0.16, 0.16, 0.94, 0.94 ]
    config["plt_xAutoRange"] = False
    config["plt_yAutoRange"] = False
    config["plt_xRange"]     = [  0.0, +20.0 ]
    config["plt_yRange"]     = [  0.0, +2.0 ]
    config["xMajor_Nticks"]  = 11
    config["yMajor_Nticks"]  = 11
    config["plt_marker"]     = "o"
    config["plt_markersize"] = 3.0
    config["plt_linestyle"]  = "-"
    config["plt_linewidth"]  = 2.0
    config["xTitle"]         = "Energy (MeV)"
    config["yTitle"]         = "$dY \mathrm{(atoms/s)}, \phi \mathrm{(photons/s)}, \sigma \mathrm{(mb)}$"

    fig     = pl1.plot1D( config=config, pngFile=pngFile )
    fig.add__plot( xAxis=energy, yAxis=dYield/1e9, label="$dY/10^{9}$" )
    fig.add__plot( xAxis=energy, yAxis=pflux /1e12, label="$\phi(E)/10^{12}$" )
    fig.add__plot( xAxis=energy, yAxis=cs_mb /1e3 , label="$\sigma(E)/10^{3}$" )
    fig.add__legend()
    fig.set__axis()
    fig.save__figure()
    
    return()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    validate__Melville_RIprod()
