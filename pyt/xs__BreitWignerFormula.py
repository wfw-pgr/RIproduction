import os, sys, json, re
import numpy                      as np
import nkUtilities.plot1D         as pl1
import nkUtilities.load__config   as lcf
import nkUtilities.configSettings as cfs

# ========================================================= #
# ===  xs__BreitWignerFormula.py                        === #
# ========================================================= #
def xs__BreitWignerFormula():

    mb2cm2    = 1e-27

    # ------------------------------------------------- #
    # --- [1] load parameters                       --- #
    # ------------------------------------------------- #
    inpFile = "dat/parameters.jsonc"
    with open( inpFile, "r" ) as f:
        text   = re.sub(r'/\*[\s\S]*?\*/|//.*', '', f.read() )
        params = json.loads( text )

    # ------------------------------------------------- #
    # --- [2] expand parameters                     --- #
    # ------------------------------------------------- #
    Emin    = params["BreitWigner.EAxis.min"]
    Emax    = params["BreitWigner.EAxis.max"]
    Enum    = params["BreitWigner.EAxis.num"]
    Er,Et   = params["BreitWigner.Er"], params["BreitWigner.Et"]
    Gamma   = params["BreitWigner.Gamma"]
    sigma_r = params["BreitWigner.sigma_r"]

    # ------------------------------------------------- #
    # --- [2] calculate Breit-Wigner Formula        --- #
    # ------------------------------------------------- #
    EAxis   = np.linspace( Emin, Emax, Enum )
    amp     = sigma_r * ( EAxis / Er )
    res     = ( 0.5*Gamma )**2 / ( ( EAxis-Er )**2 + ( 0.5*Gamma )**2 )
    thr     = ( EAxis-Et ) / ( Er-Et )
    thr     = np.sqrt( np.where( thr >= 0.0, thr, 0.0 ) )
    cs      = amp * res * thr

    # ------------------------------------------------- #
    # --- [3] save Data in a file                   --- #
    # ------------------------------------------------- #
    Data    = np.concatenate( [ EAxis[:,np.newaxis], cs[:,np.newaxis] ], axis=1 )
    import nkUtilities.save__pointFile as spf
    names   = [ "energy(MeV)", "xsection(mb)" ]
    spf.save__pointFile( outFile=params["BreitWigner.datFile"], Data=Data, names=names )
    
    # ------------------------------------------------- #
    # --- [4] configure plot                        --- #
    # ------------------------------------------------- #
    x_,y_                    = 0, 1
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["FigSize"]        = (4.5,4.5)
    config["plt_position"]   = [ 0.16, 0.16, 0.94, 0.94 ]
    config["plt_xAutoRange"] = False
    config["plt_yAutoRange"] = False
    config["plt_xRange"]     = params["BreitWigner.xRange"]
    config["plt_yRange"]     = params["BreitWigner.yRange"]
    config["xMajor_Nticks"]  = params["BreitWigner.xTicks"]
    config["yMajor_Nticks"]  = params["BreitWigner.yTicks"]
    config["plt_marker"]     = "o"
    config["plt_markersize"] = 1.0
    config["plt_linestyle"]  = "-"
    config["plt_linewidth"]  = 1.0
    config["xTitle"]         = "Energy (MeV)"
    config["yTitle"]         = "Cross-section (mb)"

    # ------------------------------------------------- #
    # --- [5] plot                                  --- #
    # ------------------------------------------------- #
    fig     = pl1.plot1D( config=config, pngFile=params["BreitWigner.pngFile"] )
    fig.add__plot( xAxis=EAxis, yAxis=cs, label=params["BreitWigner.label"] )
    fig.add__legend()
    fig.set__axis()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [6] return                                --- #
    # ------------------------------------------------- #
    return( Data )
    
# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    xs__BreitWignerFormula()
