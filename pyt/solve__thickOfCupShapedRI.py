import json, re, os, sys
import numpy as np

# ========================================================= #
# ===  solve__thickOfCupShapedRI.py                     === #
# ========================================================= #

def solve__thickOfCupShapedRI( diameter=None, length=None, volume=None, \
                               activity=None, massDensity=None, molarMass=None, \
                               halflife=None, N_Avogadro=6.02e23, silent=False ):

    cm3 = ( 1.0e-2 )**3
    mm3 = ( 1.0e-3 )**3
    
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( diameter is None ): sys.exit( "[solve__thickOfCupShapedRI.py] diameter == ???" )
    if ( length   is None ): sys.exit( "[solve__thickOfCupShapedRI.py] length   == ???" )
    if ( volume   is None ):
        if ( activity    is None ): sys.exit("[solve__thickOfCupShapedRI.py] activity   == ??")
        if ( halflife    is None ): sys.exit("[solve__thickOfCupShapedRI.py] halflife   == ??")
        if ( massDensity is None ): sys.exit("[solve__thickOfCupShapedRI.py] massDensity== ??")
        if ( molarMass   is None ): sys.exit("[solve__thickOfCupShapedRI.py] molarMass  == ??")
        N_atoms    = activity * halflife / np.log(2.0)
        n_atoms    = massDensity * N_Avogadro / molarMass
        volume_cm3 = N_atoms / n_atoms
        volume_mm3 = volume_cm3 * cm3 / mm3
        volume     = volume_mm3
    
    # ------------------------------------------------- #
    # --- [2] calculate coeff.                      --- #
    # ------------------------------------------------- #
    a3     = np.pi
    a2     = np.pi * ( diameter + length )
    a1     = np.pi * diameter * ( length + 0.25*diameter )
    a0     = -1.0  * volume
    coeffs = [ a3, a2, a1, a0 ]
    t_disk = ( 4.0*volume ) / ( np.pi * diameter**2 )

    # ------------------------------------------------- #
    # --- [3] search root                           --- #
    # ------------------------------------------------- #
    roots  = np.roots( coeffs )
    reals  = roots[ np.where( roots == roots.real ) ].real
    reals  = reals[ np.where( ( reals > 0.0 ) & ( reals < length ) & ( reals < 0.5*diameter ) ) ]
    nReals = len( reals )
    if   ( nReals == 0 ):
        if not( silent ):
            print( "[solve__thickOfCupShapedRI.py] no root was found... [ERROR] " )
        t_cup = None
    elif ( nReals == 1 ):
        if not( silent ):
            print( "[solve__thickOfCupShapedRI.py]  1 root was found " )
        t_cup = reals[0]
    elif ( nReals >= 2 ):
        if not( silent ):
            print( "[solve__thickOfCupShapedRI.py] more than 2 roots were found...." )
        t_cup = np.min( reals )

    # ------------------------------------------------- #
    # --- [4] return                                --- #
    # ------------------------------------------------- #
    if not( silent ):
        print( "[solve__thickOfCupShapedRI.py]  coeffs  == {}".format( coeffs ) )
        print( "[solve__thickOfCupShapedRI.py]  nReals  == {}".format( nReals ) )
        print( "[solve__thickOfCupShapedRI.py]  root    == {}".format( t_cup  ) )
    return( volume, t_cup, t_disk )



# ========================================================= #
# ===  display__length_vs_thick.py                      === #
# ========================================================= #

def display__length_vs_thick( datFile ="dat/length_dependency.dat", \
                              pngFile ="png/length_vs_thick.png", \
                              diameter=None, length_array=None, \
                              activity=None, massDensity=None, molarMass=None, \
                              halflife=None, N_Avogadro=6.02e23, silent=False ):

    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( diameter     is None ): sys.exit( "[solve__thickOfCupShapedRI.py] diameter     == ??" )
    if ( length_array is None ): sys.exit( "[solve__thickOfCupShapedRI.py] length_array == ??" )
    if ( activity     is None ): sys.exit( "[solve__thickOfCupShapedRI.py] activity     == ??" )
    if ( halflife     is None ): sys.exit( "[solve__thickOfCupShapedRI.py] halflife     == ??" )
    if ( massDensity  is None ): sys.exit( "[solve__thickOfCupShapedRI.py] massDensity  == ??" )
    if ( molarMass    is None ): sys.exit( "[solve__thickOfCupShapedRI.py] molarMass    == ??" )

    # ------------------------------------------------- #
    # --- [2] calculate thickness                   --- #
    # ------------------------------------------------- #
    thicks, volumes = [], []
    for ik,length in enumerate( length_array ):
        vol, t_cup, t_disk = solve__thickOfCupShapedRI( diameter=diameter, length=length, \
                                                        activity=activity, halflife=halflife, \
                                                        massDensity=massDensity, \
                                                        molarMass=molarMass, silent=True )
        thicks  += [ t_cup ]
        volumes += [ vol   ]
    thicks  = np.array( thicks  )
    volumes = np.array( volumes )

    # ------------------------------------------------- #
    # --- [3] save in a file                        --- #
    # ------------------------------------------------- #
    Data = np.concatenate( [ length_array[:,np.newaxis],
                             thicks      [:,np.newaxis],
                             volumes     [:,np.newaxis] ], axis=1 )
    import nkUtilities.save__pointFile as spf
    names = ["length(mm)", "thickness(mm)", "volume(mm3)"]
    spf.save__pointFile( outFile=datFile, Data=Data, names=names )

    # ------------------------------------------------- #
    # --- [3] display                               --- #
    # ------------------------------------------------- #
    import nkUtilities.plot1D         as pl1
    import nkUtilities.load__config   as lcf
    import nkUtilities.configSettings as cfs
    x_,y_                    = 0, 1
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["FigSize"]        = (4.5,4.5)
    config["plt_position"]   = [ 0.20, 0.20, 0.97, 0.97 ]
    config["plt_xAutoRange"] = False
    config["plt_yAutoRange"] = False
    config["plt_xRange"]     = [  0.0, +12.0   ]
    config["xMajor_Nticks"]  = 7
    config["xTitle"]         = "length (mm)"
    config["plt_marker"]     = "o"
    config["plt_markersize"] = 3.0
    config["plt_linestyle"]  = "-"
    config["plt_linewidth"]  = 2.0

    # ------------------------------------------------- #
    # --- [4] display thick                         --- #
    # ------------------------------------------------- #
    config["plt_yAutoRange"] = False
    config["plt_yRange"]     = [  0.0, +1.0e-4 ]
    config["yMajor_Nticks"]  = 11
    config["yTitle"]         = "thickness (mm)"
    fig     = pl1.plot1D( config=config, pngFile=pngFile )
    fig.add__plot ( xAxis=length_array, yAxis=thicks )
    fig.set__axis()
    fig.save__figure()



    
    

# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1] usage type 1                          --- #
    # ------------------------------------------------- #
    print()
    print( "[Direct Volume Designation]" )
    print()
    diameter   = 3.0
    length     = 6.0
    volume     = 30.0
    silent     = False
    vol, t_cup, t_disk = solve__thickOfCupShapedRI( diameter=diameter, length=length, \
                                                    volume=volume, silent=silent )
    print()
    print( "[solve__thickOfCupShapedRI.py]  volume  == {}".format( vol    ) )
    print( "[solve__thickOfCupShapedRI.py]  t_cup   == {}".format( t_cup  ) )
    print( "[solve__thickOfCupShapedRI.py]  t_disk  == {}".format( t_disk ) )
    print()

    # ------------------------------------------------- #
    # --- [2] usage type 2                          --- #
    # ------------------------------------------------- #
    print()
    print( "[From Activity of RI.]" )
    print()
    diameter    = 3.0
    length      = 1.3e-4
    activity    = 120.0e3
    halflife_y  = 1600.0
    halflife_s  = halflife_y * 365 * 24 * 60 * 60
    massDensity = 4.90
    molarMass   = 297.0
    silent      = False
    vol,t_cup,t_disk = solve__thickOfCupShapedRI( diameter=diameter, length=length, \
                                                  activity=activity, halflife=halflife_s, \
                                                  massDensity=massDensity, molarMass=molarMass,\
                                                  silent=silent )
    print()
    print( "[solve__thickOfCupShapedRI.py]  volume  == {}".format( vol    ) )
    print( "[solve__thickOfCupShapedRI.py]  t_cup   == {}".format( t_cup  ) )
    print( "[solve__thickOfCupShapedRI.py]  t_disk  == {}".format( t_disk ) )
    print()

    # ------------------------------------------------- #
    # --- [3] display graph                         --- #
    # ------------------------------------------------- #
    pngFile      = "png/length_vs_thick.png"
    length_array = np.linspace( 1.0, 10.0, 10 )
    display__length_vs_thick( diameter=diameter, length_array=length_array, \
                              activity=activity, halflife=halflife_s, \
                              massDensity=massDensity, molarMass=molarMass )
