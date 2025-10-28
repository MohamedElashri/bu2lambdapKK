'''
    Store the branches for uproot to speed up reading root files
'''
from utils.triggers import trigger_branches




def get_branches(particles=None):
    '''
    Define the branches which will be retained in the reduced files:
    Default (if no particles specified): particles = ['h1', 'h2', 'p']
    '''
    if particles is None:
        particles = ['h1', 'h2', 'p']
    
    h1, h2, h3 = particles
    
    particle_branches = [
        # branches of B daughter particles
        f'{h1}_P',f'{h1}_PT',f'{h1}_PE',f'{h1}_PX',f'{h1}_PY',f'{h1}_PZ', f'{h1}_ID', f'{h1}_TRACK_Type', f'{h1}_IPCHI2_OWNPV',
        f'{h2}_P',f'{h2}_PT',f'{h2}_PE',f'{h2}_PX',f'{h2}_PY',f'{h2}_PZ', f'{h2}_ID', f'{h2}_TRACK_Type', f'{h2}_IPCHI2_OWNPV',
        f'{h3}_P',f'{h3}_PT',f'{h3}_PE',f'{h3}_PX',f'{h3}_PY',f'{h3}_PZ', f'{h3}_ID', f'{h3}_TRACK_Type', f'{h3}_IPCHI2_OWNPV',
        
        f'{h1}_MC15TuneV1_ProbNNk', f'{h1}_MC15TuneV1_ProbNNpi', f'{h1}_MC15TuneV1_ProbNNp', f'{h1}_MC15TuneV1_ProbNNmu',
        f'{h2}_MC15TuneV1_ProbNNk', f'{h2}_MC15TuneV1_ProbNNpi', f'{h2}_MC15TuneV1_ProbNNp', f'{h2}_MC15TuneV1_ProbNNmu',
        f'{h3}_MC15TuneV1_ProbNNk', f'{h3}_MC15TuneV1_ProbNNpi', f'{h3}_MC15TuneV1_ProbNNp', f'{h3}_MC15TuneV1_ProbNNmu',
        
        f'{h1}_ProbNNk', f'{h1}_ProbNNpi', f'{h1}_ProbNNp', f'{h1}_ProbNNmu',
        f'{h2}_ProbNNk', f'{h2}_ProbNNpi', f'{h2}_ProbNNp', f'{h2}_ProbNNmu',
        f'{h3}_ProbNNk', f'{h3}_ProbNNpi', f'{h3}_ProbNNp', f'{h3}_ProbNNmu',
        
        # branches of Lambda and its daughter particles
        'Lp_P','Lp_PT','Lp_PE','Lp_PX','Lp_PY','Lp_PZ', 'Lp_ID', 'Lp_TRACK_Type',
        'Lpi_P','Lpi_PT','Lpi_PE','Lpi_PX','Lpi_PY','Lpi_PZ', 'Lpi_ID', 'Lpi_TRACK_Type',
        'L0_P','L0_PT','L0_PE','L0_PX','L0_PY','L0_PZ', 'L0_ID', 'L0_MM', 'L0_M',
        'Lp_ProbNNp', 'Lpi_ProbNNpi', 'L0_FDCHI2_ORIVX', 'L0_DIRA_OWNPV',
        'Lp_MC15TuneV1_ProbNNp', 'Lpi_MC15TuneV1_ProbNNpi', 'Lp_MC15TuneV1_ProbNNpi',
        'L0_ENDVERTEX_X', 'L0_ENDVERTEX_Y', 'L0_ENDVERTEX_Z',
        'L0_ENDVERTEX_XERR', 'L0_ENDVERTEX_YERR','L0_ENDVERTEX_ZERR',
        'L0_OWNPV_Z', 'L0_OWNPV_ZERR', 'L0_FD_OWNPV', 'L0_FDCHI2_OWNPV', 'L0_IPCHI2_OWNPV', 'Lp_IPCHI2_OWNPV', 'Lpi_IPCHI2_OWNPV',
        
        'Bu_DTFL0_Lambda0_pplus_PX', 'Bu_DTFL0_Lambda0_pplus_PY', 'Bu_DTFL0_Lambda0_pplus_PZ', 'Bu_DTFL0_Lambda0_pplus_PE',
        'Bu_DTFL0_Lambda0_piplus_PX', 'Bu_DTFL0_Lambda0_piplus_PY', 'Bu_DTFL0_Lambda0_piplus_PZ', 'Bu_DTFL0_Lambda0_piplus_PE',
        'Bu_DTF_Lambda0_pplus_PX', 'Bu_DTF_Lambda0_pplus_PY', 'Bu_DTF_Lambda0_pplus_PZ', 'Bu_DTF_Lambda0_pplus_PE',
        'Bu_DTF_Lambda0_piplus_PX', 'Bu_DTF_Lambda0_piplus_PY', 'Bu_DTF_Lambda0_piplus_PZ', 'Bu_DTF_Lambda0_piplus_PE',
        
        # branches of B particle
        'Bu_FDCHI2_OWNPV', 'nTracks',
        'Bu_DTF_Lambda0_decayLength', 'Bu_DTF_Lambda0_decayLengthErr',
        'Bu_ENDVERTEX_X', 'Bu_ENDVERTEX_Y','Bu_ENDVERTEX_Z',
        'Bu_ENDVERTEX_XERR', 'Bu_ENDVERTEX_YERR', 'Bu_ENDVERTEX_ZERR',
        'Bu_IPCHI2_OWNPV', 'Bu_MM','Bu_MMERR', 'Bu_ID',
        'Bu_P', 'Bu_PT', 'Bu_PE','Bu_PX', 'Bu_PY', 'Bu_PZ',
        'Bu_DTF_nPV', 'Bu_DTF_chi2', 'Bu_DTF_nDOF',
        'Bu_DTFL0_chi2', 'Bu_DTFL0_nDOF',
        'Bu_DTF_status','Bu_DTF_decayLength', 'Bu_DTF_decayLengthErr',
        'Bu_DTFL0_ctau', 'Bu_DTFL0_ctauErr',
        'Bu_DTF_ctau', 'Bu_DTF_ctauErr', 'Bu_DTFL0_M', 'Bu_DTFL0_MERR',
        'Bu_DIRA_OWNPV', 'eventNumber' # 'runNumber', # 'Bu_DTF_Lambda0_M'
    ]
    
    # Add trigger branches if they are defined
    try:
        particle_branches += trigger_branches
    except NameError:
        pass  # trigger_branches is not defined
    
    return particle_branches


def truth_branches(particles):
    h1, h2, h3 = particles
    particle_branches = [
        f'{h1}_TRUEID', f'{h2}_TRUEID', f'{h3}_TRUEID', 'Lp_TRUEID', 'Lpi_TRUEID', 'L0_TRUEID', 'Bu_TRUEID'
        f'{h1}_MC_MOTHER_ID', f'{h2}_MC_MOTHER_ID', f'{h3}_MC_MOTHER_ID', 'Lp_MC_MOTHER_ID', 'Lpi_MC_MOTHER_ID', 'L0_MC_MOTHER_ID'
    ]
    if "h" in particles[1]:
        particle_branches += ['p_MC15TuneV1_ProbNNp_corr', 'h1_MC15TuneV1_ProbNNk_corr', 'h2_MC15TuneV1_ProbNNk_corr']
    else:
        particle_branches += ['p1_MC15TuneV1_ProbNNp_corr', 'p2_MC15TuneV1_ProbNNp_corr', 'p3_MC15TuneV1_ProbNNp_corr']
    return particle_branches



def reshape_var(arrs):
    """
    Safely reshape arrays, checking if they are multidimensional first
    """
    # List of variables to reshape
    vars_to_reshape = [
        "Bu_DTFL0_M", "Bu_DTFL0_MERR", "Bu_DTF_decayLength", "Bu_DTF_decayLengthErr",
        "Bu_DTF_Lambda0_decayLength", "Bu_DTF_Lambda0_decayLengthErr", "Bu_DTF_ctau",
        "Bu_DTF_ctauErr", "Bu_DTF_status", "Bu_DTF_chi2", "Bu_DTF_nDOF", 
        "Bu_DTFL0_chi2", "Bu_DTFL0_nDOF", "Bu_DTFL0_ctau", "Bu_DTFL0_ctauErr"
    ]
    
    for var in vars_to_reshape:
        if var in arrs:
            # Check if array is multidimensional
            try:
                shape = arrs[var].shape
                if len(shape) > 1 and shape[1] > 0:
                    # Only reshape if it's 2D or higher
                    arrs[var] = arrs[var][:, 0]
                # If 1D, leave it as is
            except (AttributeError, IndexError) as e:
                # If we can't get the shape or there's another error, print a message and continue
                print(f"Warning: Could not reshape {var}: {e}")
                continue

    return None

def reshape(arrs, varlist: list):
    
    for var in varlist:
        arrs[var] = arrs[var][:, 0]
    
    return arrs

