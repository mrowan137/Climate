import sys
import pySCM

try:
    Filename = '/home/stefanie/Projects/SimpleClimateModel/src/pySCM/SimpleClimateModelParameterFile.txt'
    scm = pySCM.SimpleClimateModel(Filename)
    scm.runModel()
    scm.saveOutput() 
except pySCM.SCMError as e:
    print 'Error occurred: ', e.value
except:
    print 'Unknown error occurred', sys.exc_info()[0]
    raise    

