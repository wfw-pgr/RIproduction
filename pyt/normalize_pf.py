import numpy as np
import nkUtilities.plot1D         as pl1
import nkUtilities.load__config   as lcf
import nkUtilities.configSettings as cfs

# ------------------------------------------------- #
# --- [1] parameters                            --- #
# ------------------------------------------------- #
e_, p_  = 0, 1
datFile = "dat/photon_flux.dat"
spcFile = "dat/photon_full_spectrum.dat"
pngFile = "png/photon_full_spectrum.png"

# ------------------------------------------------- #
# --- [2] load data                             --- #
# ------------------------------------------------- #
import nkUtilities.load__pointFile as lpf
Data     = lpf.load__pointFile( inpFile=datFile, returnType="point" )
spectrum = lpf.load__pointFile( inpFile=spcFile, returnType="point" )
energy   = Data[:,e_]
pflux    = Data[:,p_]

# ------------------------------------------------- #
# --- [3] spline interpolation                  --- #
# ------------------------------------------------- #
import scipy.interpolate as itp
splineFunc   = itp.CubicSpline( spectrum[:,e_], spectrum[:,p_] )
Emin,Emax,nE = 0.0, 20.0, 101
e_int        = np.linspace( Emin, Emax, nE )
p_int        = splineFunc( e_int )
print( e_int.shape, p_int.shape )

# ------------------------------------------------- #
# --- [4] integration (1) :: lower-energy side  --- #
# ------------------------------------------------- #
import scipy.integrate as itg
Eth        = 6.4
eta        = 40.0                  # e - photon conversion efficiency (%)
Phi_e_tot  = 1.625e14              # (electrons/s) @ 26 uA
Phi_p_tot  = Phi_e_tot * eta/100.0 # (photons/s)
eindex     = np.argmin( np.abs( e_int - Eth ) )
eSpline_lw = e_int[:(eindex+1)]
pSpline_lw = p_int[:(eindex+1)]
eSpline_up = e_int[(eindex):]
pSpline_up = p_int[(eindex):]
S_lw       = itg.simpson( y=pSpline_lw, x=eSpline_lw ) / ( Emax - Emin )
S_up       = itg.simpson( y=pSpline_up, x=eSpline_up ) / ( Emax - Emin )
S_tot      = S_lw + S_up
R_lw       = S_lw / S_tot * 100.0
R_up       = S_up / S_tot * 100.0
F_pflux    = Phi_p_tot / S_tot
print()
print( "e_int :: ", e_int  )
print( "Eth   :: ", Eth    )
print( "index :: ", eindex )
print( "value :: ", e_int[eindex] )
print()

print( eSpline_lw.shape, pSpline_lw.shape )
print( eSpline_up.shape, pSpline_up.shape )
print()
print( "Area ( E < 6.4 MeV ) == {:12.5e}".format( S_lw  ) )
print( "Area ( E > 6.4 MeV ) == {:12.5e}".format( S_up  ) )
print( "Area ( total )       == {:12.5e}".format( S_tot ) )
print()
print( "Ratio ( E < 6.4 MeV ) == {:12.5f} (%)".format( R_lw  ) )
print( "Ratio ( E > 6.4 MeV ) == {:12.5f} (%)".format( R_up  ) )
print()
print( "Correction Factor     == {:12.5e}".format( F_pflux ) )
print()

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
config["plt_yRange"]     = [ -0.0, +4.0e12 ]
config["xMajor_Nticks"]  = 11
config["yMajor_Nticks"]  = 11
config["xTitle"]         = "Energy (MeV)"
config["yTitle"]         = "Photon Flux (photons/s)"

fig     = pl1.plot1D( config=config, pngFile=pngFile )
fig.add__plot( xAxis=spectrum[:,e_], yAxis=spectrum[:,p_], label="full dataset", \
               color="C1", \
               linestyle="-", linewidth=1.0, \
               marker="D", markersize=2.0 )
fig.add__plot( xAxis=energy, yAxis=pflux, label="reduced dataset (E>6.4 MeV)", \
               color="C0", \
               linestyle="-", linewidth=1.0, \
               marker="o", markersize=4.0 )
fig.add__plot( xAxis=e_int, yAxis=p_int, label="spline intepolated", \
               color="C2", \
               linestyle="--", linewidth=0.6, marker="none" )

fig.add__cursor( xAxis=6.4, linestyle="--", linewidth=0.6, color="gray" )
fig.add__legend()
fig.set__axis()
fig.save__figure()
