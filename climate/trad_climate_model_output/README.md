The data in this directory are results of SimpleClimateModel().runModel()
from the pySCM package.  The option to save output can be turned on by setting
save_output=True as an argument of SimpleClimateModel.runModel()
The output files are:
    SeaLevelChange.dat: Columns are 'years' and 'sea level due to changes in
                        global mean surface temperatures'
    TempChange.dat: Columns are 'years' and 'changes in mean surface temperature
                    due to changes in CO2, CH4, N2O concentrations and SOx
		    emissions
    TempChangeCommented.dat: This is the same as TempChange.dat but includes a
    			     header
    TempChangePlot.png: This is a plot of global surface temperature vs. time in
                        years, from the TempChange.dat file