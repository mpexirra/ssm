## importing packages
import numpy as np
import matplotlib.pyplot as plt
import ssm

## normalization function
def normV(X, method):   
    if method == 'max':
        return X/max(X)
    if method == 'zscore':
        return (X - np.mean(X))/np.std(X)

## restructuring input to be compatible with the ssm package
def ssmRestructure(df, inputdim):
    animals = df['animal'].unique()
    nanimals = len(animals)

    inputs = []
    Y = []
    for aa in range(nanimals):
        animaldata = df[df['animal'] == animals[aa]]
        sessions = animaldata['sessionid'].unique()
        nsessions = len(sessions)
    
        inputsaux = []
        yaux = []
        for ss in range(nsessions):
            sessiondata = animaldata[animaldata['sessionid'] == sessions[ss]]
            ntrials = len(sessiondata)
            inputsaux.append(np.ones([ntrials, inputdim]))
            yaux.append(np.zeros([ntrials,1], dtype = int))
        
            sbmag = sessiondata.surebetmag[sessiondata.lotterychoice == 0].unique()
            sbrwd = sessiondata.rewardreceived[sessiondata.lotterychoice == 0].unique()
            rwdmult = sbrwd/sbmag
            lotteryprob = sessiondata.lotteryprob.unique()
        
            # getting and saving deltaEV information
            deltaEV = rwdmult * (sessiondata.lotterymag * lotteryprob) - sbrwd
            normdeltaEV = normV(deltaEV, 'max')
            inputsaux[ss][:,0] = normdeltaEV
            yaux[ss][:,0] = sessiondata.lotterychoice

            if inputdim == 3:
                ## getting and saving previous choice information
                inputsaux[ss][:,2] = np.hstack((0, sessiondata.lotterychoice[1:]))
            if inputdim == 4:
                ## getting and sving previous reward information
                inputsaux[ss][:,2] = np.hstack((0, sessiondata.choiceout[1:]))
    
        inputs.append(inputsaux)
        Y.append(yaux)
        
    return(inputs, Y)  

## restructuring input to be compatible with the ssm package
def ssmBigAnimalRestructure(df, inputdim):
    inputs = []
    Y = []
    sessions = df['sessionid'].unique()
    nsessions = len(sessions)
    
    for ss in range(nsessions):
        sessiondata = df[df['sessionid'] == sessions[ss]]
        ntrials = len(sessiondata)
        inputs.append(np.ones([ntrials, inputdim]))
        Y.append(np.zeros([ntrials,1], dtype = int))
        
        sbmag = sessiondata.surebetmag[sessiondata.lotterychoice == 0].unique()
        sbrwd = sessiondata.rewardreceived[sessiondata.lotterychoice == 0].unique()
        rwdmult = sbrwd/sbmag
        lotteryprob = sessiondata.lotteryprob.unique()
        
        # getting and saving deltaEV information
        deltaEV = rwdmult * (sessiondata.lotterymag * lotteryprob) - sbrwd
        normdeltaEV = normV(deltaEV, 'max')
        inputs[ss][:,0] = normdeltaEV
        Y[ss][:,0] = sessiondata.lotterychoice

        if inputdim == 3:
            ## getting and saving previous choice information
            inputs[ss][:,2] = np.hstack((0, sessiondata.lotterychoice[1:]))
        if inputdim == 4:
             ## getting and sving previous reward information
            inputs[ss][:,2] = np.hstack((0, sessiondata.choiceout[1:]))

    return(inputs, Y)  


## likelihood for a set of nstates
def nstatesFit(df, nmaxstates, obsdim, inputdim, ncatergories, niterations):
    animals = df['animal'].unique()
    nanimals = len(animals)

    inputs = ssmRestructure(df, inputdim)[0]
    Y = ssmRestructure(df, inputdim)[1]

    LLstates = np.zeros((nanimals, nmaxstates))
    for aa in range(nanimals):       
        for kk in range(nmaxstates):
                nstates = kk + 1
                glmhmm = ssm.HMM(nstates, obsdim, inputdim,  observations = 'input_driven_obs',
                observation_kwargs = dict(C = ncatergories), transitions = "standard")
                
                glmhmm2fit = glmhmm.fit(Y[aa], inputs = inputs[aa], method = 'em', num_iters = niterations, tolerance = 10**-5)
        
                LLstates[aa, kk] = glmhmm2fit[-1]
    return LLstates


## plot LL per state
def plotLLstate(df, LLstates):
    animals = df['animal'].unique()
    nanimals = np.shape(LLstates)[0]
    nmaxstates = np.shape(LLstates)[1]

    xx = np.arange(nmaxstates) + 1
    for aa in range(nanimals):
        fig, ax = plt.subplots(1,1)

        plt.plot(xx, LLstates[aa,:], marker = 'o', markersize = 10, color = 'gray', linestyle = '-')

        # axis options
        plt.title(animals[aa])
        plt.xlabel('number of states', fontsize = 15)
        plt.xlim(.5, nmaxstates + .5)
        plt.ylabel('log probability', fontsize = 15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False) 

    return fig

## best overall nstate fit for all animals
def nstatesOptimal(LLstates):
    optAnimalFit = np.argmax(LLstates, axis = 1) + 1
    optOverallFit = np.where(np.bincount(optAnimalFit) == np.max(np.bincount(optAnimalFit)))

    return optOverallFit

# creating new variables for each state in dataframe
def addState2df(df, nstates):   
    nstatestr = np.arange(nstates) + 1
    stateprefix = 'pstate'
    statestr = [stateprefix + str(ii) for ii in nstatestr]

    for st in range(len(statestr)):
        str2add = statestr[st]
        df[str2add] = np.zeros(len(df))
    
    return statestr, df


## run glmhmm and save main parameters of the model
def runGLMHMM(df, nstates, obsdim, inputdim, ncatergories, niterations):
    # creating new df variables for estimated states from glmhmm
    df = addState2df(df, nstates)[1]
    statestr = addState2df(df, nstates)[0]

    # getting glmhmm inputs in correct shape
    inputs = ssmRestructure(df, inputdim)[0]
    Y = ssmRestructure(df, inputdim)[1]

    # getting number of animals from dataframe
    animals = df['animal'].unique()
    nanimals = len(animals)

    # creating variables to keep from HMM
    w = []
    transmat = []

    for aa in range(nanimals):
        w.append(np.zeros([nstates, inputdim]))
        transmat.append(np.zeros([nstates, nstates]))
    
        glmhmm = ssm.HMM(nstates, obsdim, inputdim,  observations = 'input_driven_obs',
                observation_kwargs = dict(C = ncatergories), transitions = "standard")
        glmhmm2fit = glmhmm.fit(Y[aa], inputs = inputs[aa], method = 'em', num_iters = niterations, tolerance = 10**-5)

        # saving the glm weights and transition matrix values
        waux = - glmhmm.observations.params.reshape(nstates,inputdim)
        w[aa] = waux

        transmat[aa] = glmhmm.transitions.transition_matrix
    
        # getting expected states
        posterior_pstate = [glmhmm.expected_states(data = data, input = inputs)[0]
                            for data, inputs
                            in zip(Y[aa], inputs[aa])]
   
        # saving expected states  for each trial in the data frame
        ppstate_aux = np.array([item for sublist in posterior_pstate for item in sublist])
        df.loc[df.animal == animals[aa], statestr] = ppstate_aux

    df['state'] = df[statestr].idxmax(axis = 1)    
    
    return w, transmat, df 



