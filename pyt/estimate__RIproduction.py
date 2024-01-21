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
def fit__forRIproduction( xD=None, yD=None, xI=None, mode="linear", p0=None, threshold=None ):
    
    if   ( mode == "linear" ):
        fitFunc = itp.interp1d( xD, yD, kind="linear", fill_value="extrapolate" )
        yI      = fitFunc( xI )
    elif ( mode == "gaussian" ):
        fitFunc   = lambda eng,c1,c2,c3,c4,c5 : \
            c1*np.exp( -1.0/c2*( eng-c3 )**2 ) +c4*eng +c5
        copt,cvar = opt.curve_fit( fitFunc, xD, yD, p0=p0 )
        yI        = fitFunc( xI, *copt )
    else:
        print( "[estimate__RIproduction.py] undefined mode :: {} ".format( mode ) )
        sys.exit()
    if ( threshold is not None ):
        yI = np.where( xI > threshold, yI, 0.0 )
    return( yI )


# ========================================================= #
# ===  draw__figures                                    === #
# ========================================================= #
def draw__figures( params=None, EAxis=None, pf_fit=None, xs_fit=None, \
                   pf_raw=None, xs_raw=None, dYield=None ):

    min_, max_, num_ = 0, 1, 2
    e_, xs_, pf_     = 0, 1, 1

    # ------------------------------------------------- #
    # --- [1] configure data                        --- #
    # ------------------------------------------------- #
    if ( params["plot.dYield.norm.auto"] ):
        params["plot.dYield.norm"] = 10.0**( math.floor( np.log10( abs( np.max( dYield ) ) ) ) )
    xs_fit_plot = xs_fit        / params["plot.xsection.norm"]
    pf_fit_plot = pf_fit        / params["plot.photon.norm"]
    xs_raw_plot = xs_raw[:,xs_] / params["plot.xsection.norm"]
    pf_raw_plot = pf_raw[:,pf_] / params["plot.photon.norm"]
    dY_plot     = dYield / params["plot.dYield.norm"]
    xs_norm_str = "10^{" + str( round( math.log10( params["plot.xsection.norm"] ) ) ) + "}"
    pf_norm_str = "10^{" + str( round( math.log10( params["plot.photon.norm"]   ) ) ) + "}"
    dY_norm_str = "10^{" + str( round( math.log10( params["plot.dYield.norm"]   ) ) ) + "}"
    label_xs    = "$\sigma_{fit}(E)/" + xs_norm_str + "\ \mathrm{(mb)}$"
    label_pf    = "$\phi_{fit}(E)/"   + pf_norm_str + "\ \mathrm{(photons/MeV/uA/s)}$"
    label_dY    = "$dY/ "       + dY_norm_str + "\ \mathrm{(atoms/s)}$"
    label_xs    = "$\sigma_{raw}(E)/" + xs_norm_str + "\ \mathrm{(mb)}$"
    label_pf    = "$\phi_{raw}(E)/"   + pf_norm_str + "\ \mathrm{(photons/MeV/uA/s)}$"
    
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
    config["plt_markersize"] = 1.0
    config["plt_linestyle"]  = "-"
    config["plt_linewidth"]  = 1.2
    config["xTitle"]         = "Energy (MeV)"
    config["yTitle"]         = "$dY, \ \phi, \ \sigma$"

    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    fig     = pl1.plot1D( config=config, pngFile=params["plot.filename"] )
    fig.add__plot( xAxis=EAxis       , yAxis=dY_plot    , label=label_dY, color="C0" )
    fig.add__plot( xAxis=EAxis       , yAxis=xs_fit_plot, label=label_xs, color="C1",\
                   marker="none" )
    fig.add__plot( xAxis=xs_raw[:,e_], yAxis=xs_raw_plot, label=label_xs, color="C1", \
                   linestyle="none" )
    fig.add__plot( xAxis=EAxis       , yAxis=pf_fit_plot, label=label_pf, color="C2", \
                   marker="none" )
    fig.add__plot( xAxis=pf_raw[:,e_], yAxis=pf_raw_plot, label=label_pf, color="C2", \
                   linestyle="none" )
    fig.add__legend( FontSize=9.0 )
    fig.set__axis()
    fig.save__figure()
    return()


# ========================================================= #
# ===  write__results                                   === #
# ========================================================= #
def write__results( Data=None, outFile="dat/results.dat" ):

    if ( Data is None ): sys.exit( "[estimate__RIproduction.py] Data == ???" )
    texts        = ""
    paramsFormat = "{0:>30} :: {1}\n"
    resultFormat = "{0:>30} :: {1:15.8e}   {2}\n"
    resultUnits  = { "Yield":"(atoms/s)", "Nproduced":"(atoms)", "Aproduced":"(Bq)", \
                     "t_max":"(s)", "t_max_d":"(d)", "ratio":"(%)", "Adecayed":"(Bq)" }
        
    # ------------------------------------------------- #
    # --- [1] pack texts                            --- #
    # ------------------------------------------------- #
    texts += "\n[paramters]\n"
    for key,val in Data["params"].items():
        texts += paramsFormat.format( key, val )
    texts += "\n[results]\n"
    for key,val in Data["results"].items():
        texts += resultFormat.format( key, val, resultUnits[key] )
    texts += "\n"
    
    # ------------------------------------------------- #
    # --- [2] save and print texts                  --- #
    # ------------------------------------------------- #
    print( texts )
    with open( outFile, "w" ) as f:
        f.write( texts )


# ========================================================= #
# ===  unit__halfLifeTime_second                        === #
# ========================================================= #
def unit__halfLifeTime_second( unit=None, value=None ):
    
    if   ( unit.lower() == "y" ):
        ret = value * 365*24*60*60.0
    elif ( unit.lower() == "d" ):
        ret = value *     24*60*60.0
    elif ( unit.lower() == "h" ):
        ret = value *        60*60.0
    elif ( unit.lower() == "m" ):
        ret = value *           60.0
    elif ( unit.lower() == "s" ):
        ret = value
    else:
        print( "[estimate__RIproduction.py] unknown unit :: {} ".format( unit ) )
        sys.exit()
    return( ret )


# ========================================================= #
# ===  calculate__parameters                            === #
# ========================================================= #
def calculate__parameters( params=None ):

    N_Avogadro = 6.02e23
    mm2cm      = 0.1
    
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( params is None ): sys.exit( "[estimate__RIproduction.py] params == ???" )

    # ------------------------------------------------- #
    # --- [2] calculate half-life time & lambda     --- #
    # ------------------------------------------------- #
    #  -- [2-1] half-life time of nuclide           --  #
    params["target.halflife.value"]  = unit__halfLifeTime_second( \
        unit=params["target.halflife.unit"] , value=params["target.halflife.value"]  )
    params["product.halflife.value"] = unit__halfLifeTime_second( \
        unit=params["product.halflife.unit"], value=params["product.halflife.value"] )
    params["decayed.halflife.value"] = unit__halfLifeTime_second( \
        unit=params["decayed.halflife.unit"], value=params["decayed.halflife.value"] )
    #  -- [2-2] decay constant lambda               --  #
    params["target.lambda.1/s"]  = np.log(2.0) / params["target.halflife.value"]
    params["product.lambda.1/s"] = np.log(2.0) / params["product.halflife.value"]
    params["decayed.lambda.1/s"] = np.log(2.0) / params["decayed.halflife.value"]

    # ------------------------------------------------- #
    # --- [3] volume & area model                   --- #
    # ------------------------------------------------- #
    if   ( params["target.area.type"].lower() == "direct"   ):
        params["target.area.cm2"] = params["target.area.direct.cm2"]
    elif ( params["target.area.type"].lower() == "cylinder" ):
        params["target.area.cm2"] = 0.25*np.pi * ( mm2cm * params["target.area.diameter.mm"] )**2

    # ------------------------------------------------- #
    # --- [4] thickness x atom density              --- #
    # ------------------------------------------------- #
    params["target.atoms/cm3"]   = N_Avogadro*( params["target.g/cm3"] / params["target.g/mol"] )
    if   ( params["target.thick.type"].lower() == "bq" ):
        params["target.thick.cm"]   = params["target.activity.Bq"] / \
            ( params["target.lambda.1/s"]*params["target.atoms/cm3"]*params["target.area.cm2"] )
        params["target.tN_product"] = params["target.atoms/cm3"]*params["target.thick.cm"]
    elif ( params["target.thick.type"].lower() == "direct" ):
        params["target.thick.cm"]   = params["target.thick.direct.mm"] * mm2cm
        params["target.tN_product"] = params["target.atoms/cm3"]*params["target.thick.cm"]
    elif ( params["target.thick.type"].lower() == "fluence" ):
        params["target.thick.cm"]   = params["target.activity.Bq"] / \
            ( params["target.lambda.1/s"]*params["target.atoms/cm3"]*params["target.area.cm2"] )
        params["target.tN_product"] = params["target.atoms/cm3"]
        # thick t is included in :: photonFlux profile :: 
        # fluence = count/m2 = count*m/m3, => fluence x volume = count*m
        
    # ------------------------------------------------- #
    # --- [5] return                                --- #
    # ------------------------------------------------- #
    return( params )

    
# ========================================================= #
# ===  estimate__RIproduction.py                        === #
# ========================================================= #
def estimate__RIproduction():

    e_, pf_, xs_ = 0, 1, 1
    mb2cm2       = 1.0e-27
    paramsFile   = "dat/parameters.jsonc"
    
    # ------------------------------------------------- #
    # --- [1] load parameters from file             --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        text     = re.sub(r'/\*[\s\S]*?\*/|//.*', '', f.read() )
        params   = json.loads( text )
    
    # ------------------------------------------------- #
    # --- [2] calculate parameters & define EAxis   --- #
    # ------------------------------------------------- #
    #  -- [2-1] energy axis                         --  #
    EAxis  = np.linspace( params["integral.EAxis.min"], params["integral.EAxis.max"], \
                          params["integral.EAxis.num"] )
    #  -- [2-2] calculate other parameters          --  #
    params = calculate__parameters( params=params )

    # ------------------------------------------------- #
    # --- [3] load photon flux                      --- #
    # ------------------------------------------------- #
    import nkUtilities.load__pointFile as lpf
    pf_raw     = lpf.load__pointFile( inpFile=params["photon.filename"], returnType="point" )
    pf_fit_uA  = fit__forRIproduction( xD=pf_raw[:,e_], yD=pf_raw[:,pf_], \
                                       xI=EAxis, mode=params["photon.fit.method"], \
                                       p0=params["photon.fit.p0"], \
                                       threshold=params["photon.fit.Eth"] )
    pf_fit     = params["photon.beam.current"] * pf_fit_uA
    
    # ------------------------------------------------- #
    # --- [4] load cross-section                    --- #
    # ------------------------------------------------- #
    import nkUtilities.load__pointFile as lpf
    xs_raw     = lpf.load__pointFile( inpFile=params["xsection.filename"], returnType="point")
    xs_fit_mb  = fit__forRIproduction( xD=xs_raw[:,e_], yD=xs_raw[:,xs_], \
                                       xI=EAxis, mode=params["xsection.fit.method"], \
                                       p0=params["xsection.fit.p0"], \
                                       threshold=params["xsection.fit.Eth"] )
    xs_fit     = mb2cm2 * xs_fit_mb
    
    # ------------------------------------------------- #
    # --- [5] calculate dY(E)                       --- #
    # ------------------------------------------------- #
    dYield       = params["target.tN_product"] * pf_fit * xs_fit
    
    # ------------------------------------------------- #
    # --- [6] integrate dY(E) with respect to E     --- #
    # ------------------------------------------------- #
    if ( params["integral.method"] == "simpson" ):
        Yield = itg.simpson( dYield, x=EAxis )
    Nproduced = Yield * params["photon.beam.duration"] * 3600.0
    Aproduced = Nproduced * params["product.lambda.1/s"]
    lam1,lam2 = params["product.lambda.1/s"], params["decayed.lambda.1/s"]
    t_max     = np.log( lam1/lam2 ) / ( lam1 - lam2 )
    t_max_d   = t_max / (3600*24.0)
    ratio     = ( lam2/( lam2-lam1 ) )*( np.exp( -lam1*t_max ) - np.exp( -lam2*t_max ) ) * 100.0
    Adecayed  = ( ratio/100.0 ) * Aproduced
    results   = { "Yield":Yield, "Nproduced":Nproduced, "Aproduced":Aproduced, \
                  "t_max":t_max, "t_max_d":t_max_d, "ratio":ratio, "Adecayed":Adecayed }
    
    # ------------------------------------------------- #
    # --- [7] draw sigma(E), phi(E), dY(E)          --- #
    # ------------------------------------------------- #
    draw__figures( params=params, EAxis=EAxis, pf_fit=pf_fit_uA, pf_raw=pf_raw, \
                   xs_fit=xs_fit_mb, xs_raw=xs_raw, dYield=dYield )
    
    # ------------------------------------------------- #
    # --- [8] save & return                         --- #
    # ------------------------------------------------- #
    Data = { "params":params, "EAxis":EAxis, "results":results, \
             "pf_fit":pf_fit, "xs_fit":xs_fit, "dYield":dYield }
    write__results( Data=Data, outFile=params["results.filename"] )
    return( Yield )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #
if ( __name__=="__main__" ):
    estimate__RIproduction()
