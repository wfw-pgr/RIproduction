import os, sys, re, json, math
import numpy                      as np
import scipy.interpolate          as itp
import scipy.integrate            as itg
import scipy.optimize             as opt
import nkUtilities.plot1D         as pl1
import nkUtilities.load__config   as lcf
import nkUtilities.configSettings as cfs


# ========================================================= #
# ===  fit__forRIproduction                             === #
# ========================================================= #
def fit__forRIproduction( xD=None, yD=None, xI=None, mode="linear", p0=None ):
    
    if   ( mode == "linear" ):
        fitFunc = itp.interp1d( xD, yD, kind="linear" )
        yI      = fitFunc( xI )
    elif ( mode == "gaussian" ):
        fitFunc   = lambda eng,c1,c2,c3,c4,c5 : \
            c1*np.exp( -1.0/c2*( eng-c3 )**2 ) +c4*eng +c5
        copt,cvar = opt.curve_fit( fitFunc, xD, yD, p0=p0 )
        yI        = fitFunc( xI, *copt )
    else:
        print( "[estimate__RIproduction.py] undefined mode :: {} ".format( mode ) )
        sys.exit()
    return( yI )


# ========================================================= #
# ===  draw__figures                                    === #
# ========================================================= #
def draw__figures( params=None, EAxis=None, pf_fit=None, xs_fit=None, dYield=None ):

    min_, max_, num_ = 0, 1, 2

    # ------------------------------------------------- #
    # --- [1] configure data                        --- #
    # ------------------------------------------------- #
    xs_plot     = xs_fit / params["plot.xsection.norm"]
    pf_plot     = pf_fit / params["plot.photon.norm"]
    dY_plot     = dYield / params["plot.dYield.norm"]
    xs_norm_str = "10^{" + str( round( math.log10( params["plot.xsection.norm"] ) ) ) + "}"
    pf_norm_str = "10^{" + str( round( math.log10( params["plot.photon.norm"]   ) ) ) + "}"
    dY_norm_str = "10^{" + str( round( math.log10( params["plot.dYield.norm"]   ) ) ) + "}"
    label_xs    = "$\sigma(E)/" + xs_norm_str + "\ \mathrm{(mb)}$"
    label_pf    = "$\phi(E)/"   + pf_norm_str + "\ \mathrm{(photons/MeV/s)}$"
    label_dY    = "$dY/ "       + dY_norm_str + "\ \mathrm{(atoms/s)}$"
    
    # ------------------------------------------------- #
    # --- [2] configure plot                        --- #
    # ------------------------------------------------- #
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["FigSize"]        = (4.5,4.5)
    config["plt_position"]   = [ 0.16, 0.16, 0.94, 0.94 ]
    config["plt_xAutoRange"] = False
    config["plt_yAutoRange"] = False
    config["plt_xRange"]     = [ params["plot.xRange"][min_], params["plot.xRange"][max_] ]
    config["plt_yRange"]     = [ params["plot.yRange"][min_], params["plot.yRange"][max_] ]
    config["xMajor_Nticks"]  = int( params["plot.xRange"][num_] )
    config["yMajor_Nticks"]  = int( params["plot.yRange"][num_] )
    config["plt_marker"]     = "o"
    config["plt_markersize"] = 2.0
    config["plt_linestyle"]  = "-"
    config["plt_linewidth"]  = 1.2
    config["xTitle"]         = "Energy (MeV)"
    config["yTitle"]         = "$dY, \ \phi, \ \sigma$"

    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    fig     = pl1.plot1D( config=config, pngFile=params["plot.filename"] )
    fig.add__plot( xAxis=EAxis, yAxis=xs_plot, label=label_xs )
    fig.add__plot( xAxis=EAxis, yAxis=pf_plot, label=label_pf )
    fig.add__plot( xAxis=EAxis, yAxis=dY_plot, label=label_dY )
    fig.add__legend()
    fig.set__axis()
    fig.save__figure()
    return()


# ========================================================= #
# ===  write__results                                   === #
# ========================================================= #
def write__results( Data=None, outFile="dat/results.dat" ):

    if ( Data is None ): sys.exit( "[estimate__RIproduction.py] Data == ???" )
    texts        = ""
    paramsFormat = "{0:>20} :: {1}\n"
        
    # ------------------------------------------------- #
    # --- [1] pack texts                            --- #
    # ------------------------------------------------- #
    texts += "\n[paramters]\n"
    for key,val in Data["params"].items():
        texts += paramsFormat.format( key, val )
    texts += "\n[Yield]\n"
    texts += paramsFormat.format( "Yield", "{:15.8e}".format(Data["Yield"]) )
    texts += "\n"
    
    # ------------------------------------------------- #
    # --- [2] save and print texts                  --- #
    # ------------------------------------------------- #
    print( texts )
    with open( outFile, "w" ) as f:
        f.write( texts )

    
# ========================================================= #
# ===  estimate__RIproduction.py                        === #
# ========================================================= #
def estimate__RIproduction():

    e_, pf_, xs_ = 0, 1, 1
    N_Avogadro   = 6.02e23
    mb2cm2       = 1.0e-27
    mm2cm        = 0.1
    paramsFile   = "dat/parameters.jsonc"
    
    # ------------------------------------------------- #
    # --- [1] load parameters from file             --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        text     = re.sub(r'/\*[\s\S]*?\*/|//.*', '', f.read() )
        params   = json.loads( text )

    
    # ------------------------------------------------- #
    # --- [2] EAxis / tN_product                    --- #
    # ------------------------------------------------- #
    EAxis        = np.linspace( params["EAxis.min"], params["EAxis.max"], params["EAxis.num"] )
    atomDensity  = N_Avogadro * ( params["target.g/cm3"] / params["target.g/mol"] )
    tN_product   = atomDensity * ( params["target.thick.mm"] * mm2cm )

    # ------------------------------------------------- #
    # --- [3] load photon flux                      --- #
    # ------------------------------------------------- #
    import nkUtilities.load__pointFile as lpf
    photonFlux   = lpf.load__pointFile( inpFile=params["photon.filename"], returnType="point" )
    pf_fit       = fit__forRIproduction( xD=photonFlux[:,e_], yD=photonFlux[:,pf_], \
                                         xI=EAxis, mode=params["photon.fit.method"], \
                                         p0=params["photon.fit.p0"] )
    pf_fit       = params["photon.current"] * pf_fit
    
    # ------------------------------------------------- #
    # --- [4] load cross-section                    --- #
    # ------------------------------------------------- #
    import nkUtilities.load__pointFile as lpf
    xsection     = lpf.load__pointFile( inpFile=params["xsection.filename"], returnType="point")
    xs_fit       = fit__forRIproduction( xD=xsection[:,e_], yD=xsection[:,xs_], \
                                         xI=EAxis, mode=params["xsection.fit.method"], \
                                         p0=params["xsection.fit.p0"] )
    if ( params["xsection.unit"] == "mb" ):
        xs_fit_mb = np.copy( xs_fit )
        xs_fit    = mb2cm2 * xs_fit
    else:
        print( "[estimate__RIproduction.py] xsection.unit == {} is not supported... [ERROR] ".format( params["xsection.unit"] ) )
        sys.exit()
    
    # ------------------------------------------------- #
    # --- [5] calculate dY(E)                       --- #
    # ------------------------------------------------- #
    dYield       = tN_product * pf_fit * xs_fit
    
    # ------------------------------------------------- #
    # --- [6] integrate dY(E) with respect to E     --- #
    # ------------------------------------------------- #
    if ( params["integral.method"] == "simpson" ):
        Yield = itg.simpson( dYield, x=EAxis )

    # ------------------------------------------------- #
    # --- [7] draw sigma(E), phi(E), dY(E)          --- #
    # ------------------------------------------------- #
    draw__figures( params=params, EAxis=EAxis, pf_fit=pf_fit, xs_fit=xs_fit_mb, dYield=dYield )
    
    # ------------------------------------------------- #
    # --- [8] save & return                         --- #
    # ------------------------------------------------- #
    Data = { "params":params, "Yield":Yield  , "EAxis":EAxis, \
             "pf_fit":pf_fit, "xs_fit":xs_fit, "dYield":dYield }
    write__results( Data=Data, outFile=params["results.filename"] )
    return( Yield )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #
if ( __name__=="__main__" ):
    estimate__RIproduction()
