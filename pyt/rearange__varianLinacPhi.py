import json, re, os, sys
import numpy                      as np
import scipy.interpolate          as itp
import scipy.integrate            as itg
import scipy.optimize             as opt
import nkUtilities.plot1D         as pl1
import nkUtilities.load__config   as lcf
import nkUtilities.configSettings as cfs

# ========================================================= #
# ===  rearange__varianLinacPhi.py                      === #
# ========================================================= #
def rearange__varianLinacPhi( paramsFile = "dat/parameters.jsonc" ):

    e_, pf_ = 0, 1
    
    # ------------------------------------------------- #
    # --- [1] load parameters                       --- #
    # ------------------------------------------------- #
    with open( paramsFile, "r" ) as f:
        text   = re.sub(r'/\*[\s\S]*?\*/|//.*', '', f.read() )
        params = json.loads( text )

    # ------------------------------------------------- #
    # --- [2] make EAxis & load spectrum Data       --- #
    # ------------------------------------------------- #
    EAxis = np.linspace( params["LinacPhi.EAxis.min"], params["LinacPhi.EAxis.max"],\
                         params["LinacPhi.EAxis.num"] )
    import nkUtilities.load__pointFile as lpf
    spectrum  = lpf.load__pointFile( inpFile=params["LinacPhi.refFile"], \
                                     returnType="point" )
    if ( params["LinacPhi.fit.eRange"] is not None ):
        spect = spectrum[ np.where( ( spectrum[:,e_] >= params["LinacPhi.fit.eRange"][0] )& \
                                    ( spectrum[:,e_] <= params["LinacPhi.fit.eRange"][1] ) ) ]
        
    # ------------------------------------------------- #
    # --- [3] fitting (1) :: linear                 --- #
    # ------------------------------------------------- #
    if ( params["LinacPhi.fit.method"] == "linear"  ):
        fit__photonSpectrum = itp.interp1d( spect[:,e_], spect[:,pf_], kind="linear" )
        pflux               = fit__photonSpectrum( EAxis )

    # ------------------------------------------------- #
    # --- [4] fitting (2) :: exponential            --- #
    # ------------------------------------------------- #
    if ( params["LinacPhi.fit.method"] == "exponential" ):
        # -- fitting function -- #
        def fit__photonSpectrum( eng, c1, c2, c3, c4 ):
            ret   = c1*np.exp( - c2*( eng - c3 ) ) + c4
            return( ret )
        # -- fitting -- #
        p0        = params["LinacPhi.fit.p0"] # - initial coefficients.
        copt,cvar = opt.curve_fit( fit__photonSpectrum, spect[:,e_], spect[:,pf_], p0=p0 )
        pflux     = fit__photonSpectrum( EAxis, *copt )

    # ------------------------------------------------- #
    # --- [5] fitting (3) :: spline interpolation   --- #
    # ------------------------------------------------- #
    if ( params["LinacPhi.fit.method"] == "CubicSpline" ):
        fit__photonSpectrum = itp.CubicSpline( spect[:,e_], spect[:,pf_] )
        pflux               = fit__photonSpectrum( EAxis )

    # ------------------------------------------------- #
    # --- [6] Energy threshold for phi(E)           --- #
    # ------------------------------------------------- #
    if ( params["LinacPhi.fit.Eth"] >= 0.0 ):
        pflux  = np.where( EAxis > params["LinacPhi.fit.Eth"], pflux, 0.0 ) 

    # ------------------------------------------------- #
    # --- [7] normalize phi(E) to match beam curr.  --- #
    # ------------------------------------------------- #
    Fnorm           = normalize__photonSpectrum( EAxis=EAxis, pflux=pflux, spectrum=spectrum, params=params  )
    pflux           = Fnorm * pflux
    spectrum[:,pf_] = Fnorm * spectrum[:,pf_]
    
    print( "\n"+" normalization Factor :: {}".format( Fnorm ) + "\n" )
    
    # ------------------------------------------------- #
    # --- [8] draw figures                          --- #
    # ------------------------------------------------- #
    draw__figures( EAxis=EAxis, pflux=pflux, spectrum=spectrum, params=params )
        
    # ------------------------------------------------- #
    # --- [9] save in a file                        --- #
    # ------------------------------------------------- #
    import nkUtilities.save__pointFile as spf
    Data = np.concatenate( [EAxis[:,np.newaxis], pflux[:,np.newaxis]], axis=1 )
    spf.save__pointFile( outFile=params["LinacPhi.outFile"], Data=Data )

    # ------------------------------------------------- #
    # --- [10] return                               --- #
    # ------------------------------------------------- #
    return( Data )


# ========================================================= #
# ===  draw__figures                                    === #
# ========================================================= #
def draw__figures( EAxis=None, pflux=None, spectrum=None, params=None ):

    e_, pf_ = 0, 1
    
    # ------------------------------------------------- #
    # --- [1] configure plot                        --- #
    # ------------------------------------------------- #
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["FigSize"]        = (4.5,4.5)
    config["plt_position"]   = [ 0.16, 0.16, 0.94, 0.94 ]
    config["plt_xAutoRange"] = False
    config["plt_yAutoRange"] = False
    config["plt_xRange"]     = [ -0.0, +20.0 ]
    config["plt_yRange"]     = [ -0.0, +6.0e11 ]
    config["xMajor_Nticks"]  = 11
    config["yMajor_Nticks"]  = 7
    config["xTitle"]         = "Energy (MeV)"
    config["yTitle"]         = "Photon Flux (photons/MeV/uA/s)"

    # ------------------------------------------------- #
    # --- [2] plot                                  --- #
    # ------------------------------------------------- #
    fig     = pl1.plot1D( config=config, pngFile=params["LinacPhi.pngFile"] )
    fig.add__plot( xAxis=EAxis         , yAxis=pflux         , label="fitted", \
                   linestyle="--"  , linewidth=1.0, marker="o", markersize=3.0 )
    fig.add__plot( xAxis=spectrum[:,e_], yAxis=spectrum[:,pf_], label="original", \
                   linestyle="none", linewidth=0.0, marker="D", markersize=1.5 )
    fig.add__cursor( xAxis=params["LinacPhi.fit.Eth"], linestyle="--", \
                     linewidth=0.6, color="gray" )
    fig.add__legend()
    fig.set__axis()
    fig.save__figure()
    return()


# ========================================================= #
# ===  normalize__photonSpectrum                        === #
# ========================================================= #
def normalize__photonSpectrum( EAxis=None, pflux=None, spectrum=None, params=None ):

    e_, pf_ = 0, 1
    qe      = 1.602e-19
    uA      = 1.0e-6
    
    # ------------------------------------------------- #
    # --- [1] integration                           --- #
    # ------------------------------------------------- #
    deltaE     = np.max(spectrum[:,e_]) - np.min(spectrum[:,e_])

    if ( params["LinacPhi.norm.method"] == "simpson" ):
        S_spectrum = itg.simpson( y=spectrum[:,pf_], x=spectrum[:,e_] )
    else:
        print( "[rearange__varianLinacPhi.py] LinacPhi.norm.method == {} is not supported... "\
               .format( params["LinacPhi.norm.method"] ) )
        
    # ------------------------------------------------- #
    # --- [2] define factor                         --- #
    # ------------------------------------------------- #
    AvgCount       = S_spectrum / deltaE
    totalElectrons = ( params["LinacPhi.norm.current"]*uA ) / qe
    totalPhotons   = totalElectrons * ( params["LinacPhi.norm.e-g.efficiency"] / 100.0 )
    Fnorm_         = totalPhotons / AvgCount
    Fnorm          = Fnorm_ / deltaE
    # -- totalPhotons :: [photons/s]      -- #
    # -- AvgCount     :: [photons/s]      -- #
    # -- Fnorm_       :: [dimensionless]  -- #
    # -- Fnorm        :: [1/MeV]          -- #
    # --  [IMPORTANT] to normalize phi(E) as [photons/s/MeV]  -- #
    return( Fnorm )



# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):
    rearange__varianLinacPhi()

