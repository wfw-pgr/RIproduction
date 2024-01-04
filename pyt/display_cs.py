import numpy as np
import nkUtilities.plot1D         as pl1
import nkUtilities.load__config   as lcf
import nkUtilities.configSettings as cfs

# ------------------------------------------------- #
# --- [1] parameters                            --- #
# ------------------------------------------------- #
e_, cs_  = 0, 1
datFile = "dat/Ra226_gn_Ra225_TENDL.dat"
pngFile = "png/Ra226_gn_Ra225_TENDL.png"

# ------------------------------------------------- #
# --- [2] fitting function                      --- #
# ------------------------------------------------- #
def fitfunc( eng, c1, c2, c3, c4, c5 ):
    ret = c1*np.exp( - 1.0/c2*( eng - c3 )**2 ) + c4*eng + c5
    return( ret )

# ------------------------------------------------- #
# --- [3] load data                             --- #
# ------------------------------------------------- #
import nkUtilities.load__pointFile as lpf
Data          = lpf.load__pointFile( inpFile=datFile, returnType="point" )
energy        = Data[:,e_ ]
cross_section = Data[:,cs_]

# ------------------------------------------------- #
# --- [4] curve fit                             --- #
# ------------------------------------------------- #
import scipy.optimize as opt
# init_parameters = None
init_parameters = [ 288, 6.0, 11.0, 1.0, 5.0 ]
copt, cvar      = opt.curve_fit( fitfunc, energy, cross_section, p0=init_parameters )
xEst            = np.linspace( 0.0, 20.0, 101 )
yEst            = fitfunc( xEst, *copt )

# ------------------------------------------------- #
# --- [5] configuration                         --- #
# ------------------------------------------------- #
config                   = lcf.load__config()
config                   = cfs.configSettings( configType="plot.def", config=config )
config["FigSize"]        = (4.5,4.5)
config["plt_position"]   = [ 0.16, 0.16, 0.94, 0.94 ]
config["plt_xAutoRange"] = False
config["plt_yAutoRange"] = False
config["plt_xRange"]     = [ -0.0, +20.0 ]
config["plt_yRange"]     = [ -0.0, +6.0e2 ]
config["xMajor_Nticks"]  = 11
config["yMajor_Nticks"]  =  7
config["xTitle"]         = "Energy (MeV)"
config["yTitle"]         = "Cross-section (mb)"

fig     = pl1.plot1D( config=config, pngFile=pngFile )
fig.add__plot( xAxis=energy, yAxis=cross_section, \
               color="C0", \
               linestyle="none", linewidth=1.0, \
               marker="o", markersize=4.0 )
fig.add__plot( xAxis=xEst, yAxis=yEst, \
               color="C0", \
               linestyle="-", linewidth=1.0, \
               marker="none", markersize=4.0 )
fig.set__axis()
fig.save__figure()

# ------------------------------------------------- #
# --- [6] print out fitted function             --- #
# ------------------------------------------------- #
print( "\n" + "fitting result :: " )
print( "fitting function :: y = c1 exp( - ( x-c3 )**2 / c2 ) + c4 * x + c5" )
print()
print( "              c1 :: {:10.4e}".format( copt[0] ) )
print( "              c2 :: {:10.4e}".format( copt[1] ) )
print( "              c3 :: {:10.4e}".format( copt[2] ) )
print( "              c4 :: {:10.4e}".format( copt[3] ) )
print( "              c5 :: {:10.4e}".format( copt[4] ) )
print()
