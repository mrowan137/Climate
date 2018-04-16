import matplotlib.pyplot as plt
import numpy as np
import math
import codecs
import json
from climate.data_io import *

'''
Set the constants that are used for running the simple climate model at the beginning of the class.
(1) Carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O) concentrations at their pre-industrial level (e.g. 1750 values).
(2) CO2 emissions are reported in Peta grams of carbon (PgC) where 1 PgC = 10^15 g carbon and therefore we need the 
PgCperppm constant which is the conversion factor for PgC to ppm of CO2.
(3) We need estimates of direct (aerDirectFac) and indirect (aerIndirectFac) aerosol radiative forcing factors in units of (W/m^2)/TgS. 
'''
#-------------------------------------------------------------------------------
# These are our global variables
#-------------------------------------------------------------------------------
baseCO2 = 278.305 # [ppm] 
baseCH4 = 700.0   # [ppb]
baseN2O = 270.0   # [ppb]

# PgC to ppm
PgCperppm = 2.123

# Direct and indirect RF factors 
aerDirectFac = -0.002265226
aerIndirectFac = -0.013558119

#-------------------------------------------------------------------------------
# Error handling.
#-------------------------------------------------------------------------------
class SCMError(Exception):
    '''
    Error handling.
    '''
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

#-------------------------------------------------------------------------------
# Emissions record
#-------------------------------------------------------------------------------
class EmissionRec:
    '''
    We construct a class called EmissionRec which holds the atmospheric emissions of |CO2|, |CH4|, |N2O| and |SOx|.
    The emissions need to be provided as an input to the Simple Climate Model class, where the filename and path will be set in the
    *SimpleClimateModelParameterFile.txt*. The emissions will be read from file when the Simple Climate Model class is constructed by typing:

    >>> SCM = pySCM.SimpleClimateModel('PathAndFileNameOfParameterFile')
    '''
    def __init__(self):
        self.CO2 = None
        self.CH4 = None
        self.SOx = None
        self.N2O = None
     
#-------------------------------------------------------------------------------
# Simple Climate Model Class
#------------------------------------------------------------------------------- 
     
class SimpleClimateModel:
    '''
    This is the Simple Climate Model class which can be created by typing:
    
    >>> SCM = pySCM.SimpleClimateModel('PathAndFileNameOfParameterFile')
    
    During the call of the constructor, all parameters detailed in the *SimpleClimateModelParameterFile.txt* will be read from file
    as well as the atmospheric emissions of GHGs (detailed above).
    
    .. note::
        The file containing the emissions should be in a certain format.
        Please refer to the example file (*EmissionsForSCM.dat*) for details.
    '''
    def __init__(self, Filename, theta):
        '''
        This is the constructor of the class. By calling the constructor, the emissions will be read from file 
        (filling the EmissionRec) and the parameters will be read from the parameter file.
        :param Filename: path and filename of the parameter file.
        '''
        # get emissions normalizations
        self.CO2_norm = theta[1]
        self.CH4_norm = theta[2]
        self.N2O_norm = theta[3]
        self.SOx_norm = theta[4]
        
        self._ReadParameters(Filename)
        # get start and end year of simulation
        self.startYr = int(self._GetParameter('Start year'))
        self.endYr = int(self._GetParameter('End year'))
        fileload = get_example_data_file_path('EmissionsForSCM.dat', data_dir='pySCM')
        self.emissions = self._ReadEmissions(self._GetParameter(fileload))


    
    def runModel(self, RadForcingFlag = False):
        ''' 
        This function runs the simple climate model. A number of private functions will be called but also a number of
        'independent' functions (detailed below). The model takes the atmosheric GHG emissions as input, converts them
        into concentrations and calculates the radiative forcing from the change in GHG concentrations over the years. Finally,
        the temperature change is derived from the change in radiative forcing which is required to calculate the change in sea
        level. For more information on the theory behind those calculations, please refer to 'Theory' page.
        
        To run the simple climate model, type:
        
        >>> SCM.runModel() 
        
        By default, the calculated temperature change and sea level change will be written to a textfile where the location and name
        of the textfile need to be specified in the parameter file. If the user wants to, a figure showing the temperature change and 
        sea level change, respectively, will be saved to file and again the path and filename have to be specified in the parameter file.   
        
        :param: RadForcingFlag (bool) which is set to 'False' by default. If it is set to 'True' the function returns the calculated radiative forcing.
        :returns: This function returns the radiative forcing (numpy.array) if the flag was set to true. Otherwise, nothing will be returned.
        '''
        simYears =  int(self._GetParameter('Years to evaluate response functions'))
        oceanMLDepth = float(self._GetParameter('Ocean mixed layer depth [in meters]'))
        self.CO2Concs = CO2EmissionsToConcs(self.emissions, simYears, oceanMLDepth)
        self.CH4Concs = CH4EmssionstoConcs(self.emissions)
        self.N2OConcs = N2OEmssionstoConcs(self.emissions)
        self.RadForcing = CalcRadForcing(self.emissions, self.CO2Concs, self.CH4Concs, self.N2OConcs)
        
        self.temperatureChange = CalculateTemperatureChange(simYears, self.RadForcing)
        self.seaLevelChange = CalculateSeaLevelChange(simYears, self.temperatureChange)
        
        self._SaveTempAndSeaLevelChange()
        
        if (RadForcingFlag):
            return self.RadForcing
    
    '''
    Optional output can be produced, e.g. concentration output file & figure
    '''  
    def saveOutput(self): 
        '''
        This function is optional and allows the user to save the calculated GHG concentrations and also a figure showing the
        evolution of GHG concentrations. You can call this function by typing:
        
        >>> SCM.saveOutput()
         
        If this function gets called, the user has to make sure that the path and filenames are given in the parameter file.
        '''       
        # write to file if required
        Filename = self._GetParameter('Write CH4 concentrations to file')
        if Filename:
            self._writeConcsToFile('CH4', Filename)
        
        # plot CH4 concentrations
        switch = self._GetParameter('Plot CH4 concentrations to file')
        if switch:
            self.plotConcs('CH4', switch)
            
        # write to file if required
        Filename = self._GetParameter('Write N20 concentrations to file')
        if Filename:
            self._writeConcsToFile('N2O', Filename)
        
        # plot N2O concentrations
        switch = self._GetParameter('Plot N2O concentrations to file')
        if switch:
            self.plotConcs('N2O', switch)
        
        # write to file if required
        Filename = self._GetParameter('Write CO2 concentrations to file')
        if Filename:
            self._writeConcsToFile('CO2', Filename)
        
        # plot CO2 concentrations
        switch = self._GetParameter('Plot CO2 concentrations to file')
        if switch:
            self.plotConcs('CO2', switch)

    
    def _ReadParameters(self, Filename):
        '''
        This private function will read all the parameters from the given parameter file. 
        The list of parameters will be stored in self._parameters which is a private dictionary. 
        '''
        reader = open(Filename,'r')
        
        self._parameters = dict()
        for line in reader:
            pos = line.find('=')
            if (pos > 1):
                tag = line[0:pos]
                value = line[pos+1:].strip()
                self._parameters[tag] = value
        reader.close()
        
    
    def _GetParameter(self, key):
        '''
        This private function will return the value (parameter) corresponding to the key provided.
        '''
        try:
            result = self._parameters[key]
        except KeyError:
            result = None
            
        return result
    
    def _ReadEmissions(self, Filename):
        '''
        |CO2|, |CH4|, |N2O| and |SOx| emissions will be read from file. The input file (Filename) has to be in a certain format.
        Please refer to the example file: EmissionsForSCM.dat. If there are missing values, this function will interpolate the
        values so that the emissions are available for the whole time period from startYr to endYr of the simulation.
        
	    :param Filename: path and filename of the emissions file.
        :type: string
        :returns: nothing
        '''
        
        # create empty list
        returnval = []
        # Set the length of the list. The length depends on the number of years.
        for i in range(self.endYr-self.startYr+1):
            returnval.append(EmissionRec())
    
        # read data
        fileload = get_example_data_file_path('EmissionsForSCM.dat', data_dir='pySCM')
        table =np.loadtxt(fileload, skiprows=3)
    
        for col in range(1,table.shape[1]):
            data = np.zeros((len(returnval)))
            data.fill(float('NaN'))
            for row in range(len(table)):
                index = int(table[row][0] - self.startYr)
                data[index] = table[row][col]
    
            # now you should have all data for one species
            # interpolate missing values
            x = np.arange(0,len(returnval))
    
            xp_hold = np.where(~np.isnan(data))[0]
            xp = np.zeros(len(xp_hold)+1)
            xp[1:len(xp)] = xp_hold[0:len(xp_hold)]
    
            fp_hold = data[np.where(~np.isnan(data))[0]]
            fp = np.zeros(len(fp_hold)+1)
            fp[1:len(fp)] = fp_hold[0:len(fp_hold)]
    
            interpolVal = np.interp(x, xp, fp)

            # Adjust emissions by normalization
            for index in range(len(interpolVal)):
                if col == 1:
                    returnval[index].CO2 = interpolVal[index]*self.CO2_norm
                elif col == 2:
                    returnval[index].CH4 = interpolVal[index]*self.CH4_norm
                elif col == 3:
                    returnval[index].N2O = interpolVal[index]*self.N2O_norm
                elif col == 4:
                    returnval[index].SOx = interpolVal[index]*self.SOx_norm

            
        return returnval
    
    #---------------------------------------------------------
    # write and plot GHG concentrations to file (if required)
    #---------------------------------------------------------
    
    def _writeConcsToFile(self, species, outputfilename):
        '''
        This private function writes the calculated GHG concentrations for all years the simple climate model was running for to text 
        file. This function is called within the model and takes the calculated concentrations of the given species and the 
        output-filename as input. The user can set the output path and file name in the parameter file that gets given to the 
        model at initialisation. As this is a private function, this function should not be called by the user!   
        '''
        if species == 'CO2':
            concs2write = self.CO2Concs+baseCO2
            unit = "ppm"
        elif species == 'CH4':
            concs2write = self.CH4Concs+baseCH4
            unit = "ppb"
        elif species == 'N2O':
            concs2write = self.N2OConcs+baseN2O
            unit = "ppb"
            
        writer = open(outputfilename, 'w')
        writer.write("This files contains the CH4 concentrations ["+unit+"] for the years the model has been running for."+'\n')
        for i in range(len(concs2write)):
            writer.write(str(self.startYr+i)+"    "+str(concs2write[i])+"\n")
        writer.close()    
    
    #-----------------------------------------------
    # plot concentrations and save figures to file.
    #-----------------------------------------------
    
    def plotConcs(self, species, outputFile):
        x = np.arange(self.startYr,self.endYr+1)
         
        if species == 'CO2':
            concs2plot = self.CO2Concs+baseCO2
            unit = "ppm"
        elif species == 'CH4':
            concs2plot = self.CH4Concs+baseCH4
            unit = "ppb"
        elif species == 'N2O':
            concs2plot = self.N2OConcs+baseN2O
            unit = "ppb"    
         
        fig = plt.figure(1)
        plt.plot(x,concs2plot)
        # title and axes labels
        fig.suptitle(species + ' concentrations', fontsize=20)
        plt.xlabel('Year', fontsize=18)
        plt.ylabel(species + ' concentration ['+unit+']', fontsize=18)
        # axes limits
        plt.xlim([self.startYr,self.endYr])
        plt.ylim([np.min(concs2plot), np.max(concs2plot)])
        # save figure to file
        plt.savefig(outputFile)
        plt.clf()
        
    def _SaveTempAndSeaLevelChange(self):
        '''
        This private function saves the calculated temperature change and resulting sea level change to file. The path and filenames
        need to be provided in the parameter file. If the user also wants to save a figure showing the evolution of the temperature
        change and sea level change, the user also needs to provide the filename for the figure files in the parameter file.
        '''
        try:
            """
            Filename = get_example_data_file_path('TempChangeCommented.dat', data_dir='trad_climate_model_output')
            if not Filename:
                raise SCMError('You need to provide a filename in the Parameter set up file!') 
            # write values to file (including comment)
            writer = open(Filename, 'w')
            writer.write("This files contains change in temperature [degC] for the years the model has been run for."+'\n')
            for i in range(len(self.temperatureChange)):
                writer.write(str(self.startYr+i)+"    "+str(self.temperatureChange[i])+"\n")
            writer.close()
            """
            # Write TempChange data to file
            #Filename = get_example_data_file_path('TempChange.dat', data_dir='trad_climate_model_output')
            Filename = get_example_data_file_path('TempChange.json', data_dir='trad_climate_model_output')
            if not Filename:
                raise SCMError('You need to provide a filename in the Parameter set up file!') 
            # Write values to file (no comment)
            #writer = open(Filename, 'w')
            #for i in range(len(self.temperatureChange)):
            #    writer.write(str(self.startYr+i)+"    "+str(self.temperatureChange[i])+"\n")
            #writer.close()
            years = self.startYr+np.arange(len(self.temperatureChange))
            temps = self.temperatureChange[:]
            z = np.array(list(zip(years,temps))).tolist()
            
            # open the file for writing
            fileObj = codecs.open(Filename, 'w', encoding='utf-8')             
            json.dump(z, fileObj, separators=(',', ':'), sort_keys=True, indent=4)
            fileObj.close()
            
            """
            # Plot temperature change and save figure to file if required
            fileload = get_example_data_file_path('TempChangePlot.png', data_dir='trad_climate_model_output')
            PlotFile = None#fileload
            if PlotFile:
                x = np.arange(self.startYr,self.endYr+1)
                fig = plt.figure(1)
                plt.plot(x,self.temperatureChange)
                # title and axes labels
                fig.suptitle(' Temperature change ', fontsize=20)
                plt.xlabel('Year', fontsize=18)
                plt.ylabel(' Temperature change [degC]', fontsize=18)
                # axes limits
                plt.xlim([self.startYr,self.endYr])
                plt.ylim([np.min(self.temperatureChange), np.max(self.temperatureChange)])
                # save figure to file
                plt.savefig(PlotFile)
                plt.clf()
            """
            """
            # Output SeaLevelChange if needed
            filesave = get_example_data_file_path('SeaLevelChange.dat', data_dir='trad_climate_model_output')
            SeaLevelFilename = filesave
            if not SeaLevelFilename:
                raise SCMError('You need to provide a filename in the Parameter set up file!') 
            # write values to file
            writer = open(SeaLevelFilename, 'w')
            writer.write("This files contains change in sea level [cm] for the years the model has been run for."+'\n')
            for i in range(len(self.seaLevelChange)):
                writer.write(str(self.startYr+i)+"    "+str(self.seaLevelChange[i])+"\n")
            writer.close() 
            """
            """
            # Plot temperature change and save figure to file if required
            PlotFile = self._GetParameter('Plot sea level change')
            if PlotFile:
                x = np.arange(self.startYr,self.endYr+1)
                fig = plt.figure(1)
                plt.plot(x,self.seaLevelChange)
                # title and axes labels
                fig.suptitle(' Sea level change ', fontsize=20)
                plt.xlabel('Year', fontsize=18)
                plt.ylabel(' Sea level change [m]', fontsize=18)
                # axes limits
                plt.xlim([self.startYr,self.endYr])
                plt.ylim([np.min(self.seaLevelChange), np.max(self.seaLevelChange)])
                # save figure to file
                plt.savefig(PlotFile)
                plt.clf()
            """
        except:
            raise
            
#-----------------------------------------------------------------------------
# Public functions which can be called outside the simple climate model class
#-----------------------------------------------------------------------------
        
def GenerateOceanResponseFunction(numYears, OceanMLDepth):
    '''
    This function calculates the ocean mixed layer response function (HILDA model) as described in Joos et al., 1996. 
    This function returns the amount of carbon remaining in the surface layer of the ocean after an input (pulse) from the atmosphere
    scaled to units of micromol/kg.
    
    :param numYears: The number of years to calculate the response function for.
    :param OceanMLDepth: Ocean mixed layer depth in meters.
    :returns:  numpy.array -- contains the remaining carbon per year.
    '''
    
    #---------------------------------------------------------------------
    # The following constants were taken from Joos et al., 1996, pg 400.
    #---------------------------------------------------------------------
    
    OceanArea = 3.62E14                                         # ocean area in square meters
    gCperMole = 12.0113                                         # molar mass of carbon.
    SeaWaterDens = 1.0265E3                                     # sea water density in kg/m^3.

    returnVal = np.zeros(numYears)

    for yr in range(numYears):
        if yr < 2.0:
            value = 0.12935 + 0.21898*np.exp(-yr/0.034569) + 0.17003*np.exp(-yr/0.26936) + 0.24071*np.exp(-yr/0.96083) + 0.24093*np.exp(-yr/4.9792)
        else:
            value = 0.022936 + 0.24278*np.exp(-yr/1.2679) + 0.13963*np.exp(-yr/5.2528) + 0.089318*np.exp(-yr/18.601) + 0.037820*np.exp(-yr/68.736) + 0.035549*np.exp(-yr/232.30);
        
        # scale values to micromole per kg
        returnVal[yr] = value * (1E21*PgCperppm/gCperMole)/(SeaWaterDens*OceanMLDepth*OceanArea)

    return returnVal

def CO2EmissionsToConcs(emissions, numYears, OceanMLDepth):
    """
    This function converts atmospheric |CO2| emissions to concentrations as described in Joos et al. 1996.
    
    :param emissions: atmospheric |CO2| emissions [PgC/year]
    :param numYears: number of years the response function is going to be calculated for
    :param OceanMLDepth: ocean mixed layer depth [m]
    :returns: numpy array -- containing the atmospheric |CO2| concentrations for each year [ppm]
    """
    # XAtmosBio is the amount of CO2 returned to the atmosphere as a result
    # of decay of the enhanced plant growth resulting from higher CO2.
    XAtmosBio = 0.0
    AirSeaGasExchangeCoeff = 0.1042                         # kg m^-2 year^-1
    BiosphereNPP_0 = 60.0                                   # GtC/year.
    # 0.287 balances LUC emission of 1.1 PgC/yr in 1980s (Joos et al, 1996)
    # 0.380  balances LUC emission of 1.6 PgC/yr in 1980s (IPCC 1994)
    CO2FertFactor = 0.287
    CO2ppm_0 = 278.305
    atmosCO2 = np.zeros(len(emissions))
    atmosBioFlux = np.zeros(len(emissions))
    surfaceOceanDIC = np.zeros(len(emissions))
    seaWaterPCO2 = np.zeros(len(emissions))
    atmosSeaFlux = np.zeros(len(emissions))

    oceanResponse = GenerateOceanResponseFunction(numYears, OceanMLDepth)
    bioResponse = GenerateBiosphereResponseFunction(numYears)

    for yrInd in range(len(emissions)-1):
        if (yrInd > 0):
            seaWaterPCO2[yrInd] = DeltaSeaWaterCO2FromOceanDIC(surfaceOceanDIC[yrInd])

        atmosSeaFlux[yrInd] = AirSeaGasExchangeCoeff*(atmosCO2[yrInd]-seaWaterPCO2[yrInd])
        # delta is the amount of CO2 taken out of the atmosphere due to stimulated plant growth minus the amount of CO2 returned
        # to the atmosphere due to the decay of organic material.
        delta = BiosphereNPP_0*CO2FertFactor*np.log(1.0+(atmosCO2[yrInd]/CO2ppm_0))/PgCperppm-XAtmosBio
        XAtmosBio += delta
        atmosBioFlux[yrInd] += XAtmosBio
        # Accumulate committments of these fluxes to all future times for SurfaceOceanDIC and AtmosBioFlux.
        for j in range(yrInd+1,len(surfaceOceanDIC)):
            Hold = surfaceOceanDIC[j]
            Hold = Hold + atmosSeaFlux[yrInd] * oceanResponse[j-yrInd];
            surfaceOceanDIC[j] = Hold

        for j in range(yrInd+1,len(atmosBioFlux)):
            atmosBioFlux[j] = atmosBioFlux[j] - XAtmosBio * bioResponse[j-yrInd]

        atmosCO2[yrInd+1] = atmosCO2[yrInd]+(emissions[yrInd].CO2/PgCperppm)-atmosSeaFlux[yrInd]-atmosBioFlux[yrInd]
    
    return atmosCO2

def GenerateBiosphereResponseFunction(numYears):
    '''
    This function calculates the decay response function for the biosphere.
    
    :param numYears: number of years to calculate the response function for.
    :returns: numpy.array -- contains the biosphere-atmospheric flux after initial carbon input per year
    '''
    returnVal = np.zeros(numYears)
    for yr in range(numYears):
        # Biosphere decay response function from Joos et al. 1996, pg. 416
        returnVal[yr] = 0.7021*np.exp(-0.35*yr) + 0.01341*np.exp(-yr/20.0) - 0.7185*np.exp(-0.4583*yr)+0.002932*np.exp(-0.01*yr);

    return returnVal

def DeltaSeaWaterCO2FromOceanDIC(SurfaceOceanDIC):
    '''
    This function calculates the change in sea water |CO2| from equilibrium corresponding to change in ocean mixed layer carbon from 
    equilibrium.
    
    :param SurfaceOceanDIC: Surface ocean dissolved inorganic carbon (DIC) [micromol/kg]
    :returns: the change in sea water |CO2| [ppm]
    '''
    TC = 18.1716                        # Effective Ocean temperature for carbonate chemistry in deg C.
    A1 = (1.5568-1.3993E-2*TC);
    A2 = (7.4706-0.20207*TC)*1E-3;
    A3 = -(1.2748-0.12015*TC)*1E-5;
    A4 = (2.4491-0.12639*TC)*1E-7;
    A5 = -(1.5468-0.15326*TC)*1E-10;
    # from Joos et al. 1996, pg. 402
    returnVal = SurfaceOceanDIC*(A1+SurfaceOceanDIC*(A2+SurfaceOceanDIC*(A3+SurfaceOceanDIC*(A4+SurfaceOceanDIC*A5))));
    
    return returnVal

def CH4EmssionstoConcs(emissions):
    '''
    This function converts methane (|CH4|) emissions into concentrations.
    
    :param emissions: |CH4| emissions [TgCH4/year]
    :returns: numpy.array -- containing the |CH4| concentrations for each year [ppb]
    '''
    TauCH4 = 10.0                   # Lifetime of CH4 
    LamCH4 = 1.0/TauCH4             # inverse lifetime in years-1
    scaleCH4 = 2.78                 # TgCH4 per ppb (IPCC TAR report value, chapter 4)

    Result = np.zeros(len(emissions))
    decay = np.exp(-LamCH4)
    accum = (1.0-decay)/(LamCH4*scaleCH4)
    for i in range(1,len(emissions)):
        Result[i] = Result[i-1] * decay + emissions[i-1].CH4 * accum

    return Result

def N2OEmssionstoConcs(emissions):
    '''
    This function converts nitrous oxide (|N2O|) emissions into concentrations.
    
    :param emissions: |N2O| emissions [TgN2O/year]
    :returns: numpy.array -- containing the |N2O| concentrations for each year [ppb]
    '''
    tauN2O = 114.0                  # Lifetime of N2O
    lamN2O = 1.0/tauN2O             # inverse lifetime in years-1
    scaleN2O = 4.8                  # TgN2O per ppb (IPCC TAR report value, chapter 4)

    Result = np.zeros(len(emissions))
    decay = np.exp(-lamN2O)
    accum = (1.0-decay)/(lamN2O*scaleN2O)
    for i in range(1,len(emissions)):
        Result[i] = Result[i-1] * decay + emissions[i-1].N2O * accum

    return Result

def CalcRadForcing(emissions, CO2Concs, CH4Concs, N2OConcs):
    '''
    This function calculates the total radiative forcing (formula given in IPCC TAR Chapter 6). The total change in radiative forcing 
    is the sum of the changes in radiative forcing resulting from changes in |CO2|, |CH4|, and |N2O| concentrations and sulfate 
    emissions.
    
    :param emissions: |SOx| emissions [TgS/year]
    :param CO2Concs: |CO2| concentrations [ppm]
    :param CH4Concs: |CH4| concentrations [ppb]
    :param N2OConcs: |N2O| concentrations [ppb]
    :returns: numpy.array -- containing the change in radiative forcing per year.
    '''
    radForcingCH4 = np.zeros(len(emissions))
    radForcingN2O = np.zeros(len(emissions))

    # changes in radiative forcing due to changes in CO2 concentrations
    radForcingCO2 = np.array([5.35 * np.log(1 + (CO2Concs[i]/baseCO2)) for i in range(len(emissions))])

    # changes in radiative forcing due to changes in CH4 concentrations
    for i in range(len(emissions)):
        # the two functions below (fnow/fthen) account for the fact that methane and nitrous oxide have overlapping absoption bands so that higher concentrations of one gas will reduce the
        # effective absoption by the other and vice versa.
        fnow = 0.47 * np.log(1 + 2.01e-5 * (((baseCH4 + CH4Concs[i]) * baseN2O)**0.75) + 5.31e-15 * (baseCH4 + CH4Concs[i]) * (((baseCH4 + CH4Concs[i])*baseN2O)**1.52))
        fthen = 0.47 * np.log(1 + 2.01e-5 * ((baseCH4 * baseN2O)**0.75) + 5.31e-15 * baseCH4 * ((baseCH4 * baseN2O)**1.52))
        radForcingCH4[i] = 0.036 * (math.sqrt(baseCH4 + CH4Concs[i]) - math.sqrt(baseCH4)) - (fnow - fthen)

    # changes in radiative forcing due to changes in N2O concentrations
    for i in range(len(emissions)):
        fnow = 0.47 * np.log(1 + 2.01e-5 * ((baseCH4 * (baseN2O + N2OConcs[i]))**0.75) + 5.31e-15 * baseCH4 * ((baseCH4 * (baseN2O + N2OConcs[i]))**1.52))
        fthen = 0.47 * np.log(1 + 2.01e-5 * ((baseCH4 * baseN2O)**0.75) + 5.31e-15 * baseCH4 * ((baseCH4 * baseN2O)**1.52))
        radForcingN2O[i] = 0.12 * (math.sqrt(baseN2O + N2OConcs[i]) - math.sqrt(baseN2O)) - (fnow - fthen)

    # changes in radiative forcing due to changes in SOx emissions
    radForcingSOx = np.array([(aerDirectFac + aerIndirectFac) * emissions[i].SOx for i in range(len(emissions))])

    # sum 
    totalRadForcing = radForcingCO2 + radForcingCH4 + radForcingN2O + radForcingSOx
    
    return totalRadForcing 

def GenerateTempResponseFunction(numYrs):
    '''
    This function calculates the temperature response function that is used to calculate the change in global mean surface 
    temperature as a result of changes in radiative forcing.
    
    :param numYrs: number of years the response function will be evaluates for.
    :returns: numpy.array -- containing climate response function
    '''
    # The values below were determined by fitting a double exponentional impulse response function model (see documentation) to values from a HadCM3 4xCO2 simulation. 
    result = np.array([(0.59557/8.4007)*np.exp(-i/8.4007)+(0.40443/409.54)*np.exp(-i/409.54) for i in range(numYrs)])

    return result

def GenerateSeaLevelResponseFunction(numYrs):
    '''
    This function calculates the sea level response function that is used to calculate the change in sea level as a result of changes in in global mean surface temperature.
    This equation only accounts for changes in sea level resulting from thermal expansion of the ocean, it does not include the effects of melting glaciers and melting grounded 
    ice sheets.
    
    :param numYrs: number of years the response function will be evaluates for.
    :returns: numpy.array -- containing climate response function
    '''
    # The values below were determined by fitting a double exponentional impulse response function model (see documentation) to values from a HadCM3 4xCO2 simulation. 
    result = ([(0.96677/1700.2)*np.exp(-i/1700.2)+(0.03323/33.788)*np.exp(-i/33.788) for i in range(numYrs)])

    return np.array(result)


def CalculateTemperatureChange(numYears, radForcing):
    '''
    This function calculates the temperature change due to changes in radiative forcing.
    
    :param numYears: number of years the temperature response function will be evaluated for.
    :param radForcing: changes in radiative forcing due to changes in |CO2|, |CH4|, |N2O| concentrations and |SOx| emissions.
    :return: numpy.array --containing the temperature change for every year.
    '''
    
    # climate sensitivity := the equilibrium change in global mean surface temperature following a doubling of the atmospheric equivalent CO2 concentration
    ClimateSensitivity = 1.1;                               # (4.114/3.74)
    result = np.zeros(len(radForcing))

    tempResFunc = GenerateTempResponseFunction(numYears)

    for i in range(len(radForcing)):
        for j in range(i, len(radForcing)):
            result[j] = result[j] + radForcing[i] * tempResFunc[j-i]

    result = result * ClimateSensitivity

    return result

def CalculateSeaLevelChange(numYears, tempChange):
    '''
    This function calculated the changes in sea level due to changes in global mean surface temperatures.
    
    :param numYears: number of years the sea level response function will be evaluated for.
    :param tempChange: changes in global mean surface temperature due to changes in |CO2|, |CH4|, |N2O| concentrations and |SOx| emissions.
    :return: numpy.array -- containing the sea level change for every year.
    '''
    result = np.zeros(len(tempChange))
    seaLevelResFunc = GenerateSeaLevelResponseFunction(numYears)

    for i in range(len(tempChange)):
        for j in range(i, len(tempChange)):
            result[j] = result[j] + tempChange[i] * seaLevelResFunc[j-i]

    return result
