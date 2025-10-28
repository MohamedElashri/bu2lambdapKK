import uproot
import ROOT as rt
import numpy as np
import awkward as ak
from array import array

'''
    rootv6.24.06 for python3.6.8
    rootv6.32.02 for pyhton3.9.18
'''

def getfolders(filename):

    """
        Return the list of TDictionaries in a root file.
    """
    
    folders = []
    with uproot.open(filename) as __file:
        folders = [treekey[:-2] for treekey in __file.keys() if "DecayTree" in treekey]
    __file.close()

    return folders




def invmass(arrs, particles) -> np.array:

    """
        Get the invariant mass of multiple particles.        
        Parameters:
        
        arrs: np.arrays from uproot.concatente
        particles: list of particle names like: ['L0_P', 'Pi_P', 'K_P']
    """

    Ps = []
    for component in ["E", "X", "Y", "Z"]:
        sumP = np.sum([arrs[f"{particle}{component}"] for particle in particles], axis=0)
        Ps.append(sumP)
    return np.sqrt(np.abs(Ps[0]**2 - Ps[1]**2 - Ps[2]**2 - Ps[3]**2))


    
    

def p2(arrs, particles) -> np.array:

    
    Ps = []
    for component in ["X", "Y", "Z"]:
        sumP = np.sum([arrs[f"{particle}{component}"] for particle in particles], axis=0)
        Ps.append(sumP)
    return Ps[0]**2 + Ps[1]**2 + Ps[2]**2



# def replace_mass(arrs, particles, mass):
    
#     newPE = np.sqrt(mass**2 + p2(arrs, particles[0]))
    
    
#     return arrs




def trigger(arrs):

    selection = ( (arrs["Bu_L0Global_TIS"] | arrs["Bu_L0HadronDecision_TOS"]) &
        (arrs["Bu_Hlt1TrackMVADecision_TOS"] | arrs["Bu_Hlt1TwoTrackMVADecision_TOS"]) &
        (arrs["Bu_Hlt2Topo2BodyDecision_TOS"] | arrs["Bu_Hlt2Topo3BodyDecision_TOS"] | arrs["Bu_Hlt2Topo4BodyDecision_TOS"])) 
    return arrs[selection]

 
def write_tuple(arrs, tfolders, outfile):

    ''' write the processed arrs to outfile/tfolder '''

    print(f"Writing to file ==> {outfile.file_path}")

    for ii in range(len(arrs)):
        allbranches = dict()
        for branch in arrs[ii].fields:
            # print(branch)
            # allbranches[branch] = arrs[ii][branch]
            # allbranches[branch] = arrs[ii][branch].to_list()
            if "_PP_" in branch:
                continue
            else:
                # allbranches[branch] = ak.to_numpy(arrs[ii][branch])
                allbranches[branch] = arrs[ii][branch].to_list()


        outfile[tfolders[ii]] = allbranches
    return outfile



def load_data(filelist, cutstr, variable):

    chain = rt.TChain("DecayTree")
    for file in filelist:
        chain.Add(file)
    tree = chain.CopyTree(cutstr)
    dataset = rt.RooDataSet("dataset", "", tree, rt.RooArgSet(variable))
    return dataset


def load_tree(filelist, cutstr):

    chain = rt.TChain("DecayTree")
    for file in filelist:
        chain.Add(file)
    tree = chain.CopyTree(cutstr)
    return tree


def sweight(dataset: rt.RooDataSet, fitmodel, rooargs: rt.RooArgSet, outname, oldtree=None):

    """
        Function to implement the SPlot.
        Args:
            rooargs: A RooArgSet contains the yields parameters,
                     better starts with RooArgSet(nsig, ncombkg, nphybkg1, nphybkg2, ...)
    """
    sData = rt.RooStats.SPlot("sData", "An SPlot", dataset, fitmodel, rooargs)
    sData.Print()
    
    outfile = rt.TFile(outname, "RECREATE")
    newtree = rt.TTree("sweight", "sweight")
    if oldtree is not None:
        newtree = oldtree.CloneTree(0)
    swlist = []
    for i in range(rooargs.getSize()):

        swlist.append(array("d", [0]))
        newtree.Branch(f"sw_{i}", swlist[i], f"sw_{i}/D")
    
    index = 0
    # get the args of the RooArgSet
    allargs = [arg for arg in rooargs]
    sumentries = int(dataset.sumEntries())
    if oldtree is not None:
        if sumentries != oldtree.GetEntries():
            raise("Tree and dataset should have same entries")
    for i in range(0, sumentries):
        if oldtree is not None:
            oldtree.GetEntry(i)
                 
        row = dataset.get(i)
        for j in range(rooargs.getSize()):

            swlist[j][0] = row.find(allargs[j].GetName()+"_sw").getVal()
        newtree.Fill()
        index += 1
    outfile.cd()
    newtree.Write()
    outfile.Close()
    return None



