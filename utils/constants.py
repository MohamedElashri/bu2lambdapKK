"""
    Constant objects to keep consistent in analysis
"""

maglist = ["MD", "MU"]
tracklist = ["LL", "DD"]
yearlist = ["16", "17", "18"]

trackcutstr = dict()
trackcutstr["LL"] = "Lp_TRACK_Type==3&&Lpi_TRACK_Type==3"
trackcutstr["DD"] = "Lp_TRACK_Type==5&&Lpi_TRACK_Type==5"

dataPath = "/eos/lhcb/user/m/melashri/data/bu2LpKK/RD/reduced"
mcPath = "/eos/lhcb/wg/BnoC/Bu2LambdaPPP/MC/DaVinciTuples/restripped.MC"
mcCorrPath = "/eos/lhcb/user/m/melashri/data/bu2LpKK/MC/PIDCorrection"



''' Trigger cuts '''
trigcut = ("(Bu_L0Global_TIS|Bu_L0HadronDecision_TOS)"
            "&(Bu_Hlt1TrackMVADecision_TOS|Bu_Hlt1TwoTrackMVADecision_TOS)"
            "&(Bu_Hlt2Topo2BodyDecision_TOS|Bu_Hlt2Topo3BodyDecision_TOS|Bu_Hlt2Topo4BodyDecision_TOS)")


''' Truth match '''
truth = ("&(abs(Bu_TRUEID)==521) & (abs(L0_TRUEID)==3122)"
         "& (abs(Lp_TRUEID)==2212) & (abs(Lpi_TRUEID)==211)"
         "& (abs(Lp_MC_MOTHER_ID)==3122) & (abs(Lpi_MC_MOTHER_ID)==3122)")

truthpkk = truth + ("& (abs(p_TRUEID)==2212) & (abs(h1_TRUEID)==321) & (abs(h2_TRUEID)==321)"
                    "& (abs(p_MC_MOTHER_ID)==521) & (abs(L0_MC_MOTHER_ID)==521)"
                    "& ( ((abs(h1_MC_MOTHER_ID)==521) & (abs(h2_MC_MOTHER_ID)==521)) | ((abs(h2_MC_MOTHER_ID)==521) & (abs(h1_MC_MOTHER_ID)==521)))")


truthppbar = truth + ("& (abs(p1_TRUEID)==2212) & (abs(p2_TRUEID)==2212) & (abs(p3_TRUEID)==2212)"
                      "& (abs(p1_MC_MOTHER_ID)==521) & (abs(p2_MC_MOTHER_ID)==521) & (abs(p3_MC_MOTHER_ID)==521) & (abs(L0_MC_MOTHER_ID)==521)")

truthEtacPPbar = truth + ("& (abs(p1_TRUEID)==2212) & (abs(p2_TRUEID)==2212) & (abs(p3_TRUEID)==2212)"
                      "& (abs(L0_MC_MOTHER_ID)==521)")

truthJpsiK = truth + ("& (abs(p_TRUEID)==2212) & (abs(h1_TRUEID)==321) & (abs(h2_TRUEID)==321)"
                      "& (abs(p_MC_MOTHER_ID)==443) & (abs(L0_MC_MOTHER_ID)==443)"
                      "& ( ((abs(h1_MC_MOTHER_ID)==521) & (abs(h2_MC_MOTHER_ID)==443)) | ((abs(h2_MC_MOTHER_ID)==521) & (abs(h1_MC_MOTHER_ID)==443)))")

truthEtacK = truth + ("& (abs(p_TRUEID)==2212) & (abs(h1_TRUEID)==321) & (abs(h2_TRUEID)==321)"
                      "& (abs(p_MC_MOTHER_ID)==441) & (abs(L0_MC_MOTHER_ID)==441)"
                      "& ( ((abs(h1_MC_MOTHER_ID)==521) & (abs(h2_MC_MOTHER_ID)==441)) | ((abs(h2_MC_MOTHER_ID)==521) & (abs(h1_MC_MOTHER_ID)==441)))")


# # cut strings
# cutstrs = dict()
# cutstrs["signal_pre_DD"] = "& (Bu_DTF_chi2<30) & (Bu_FDCHI2_OWNPV>175) & (Bu_IPCHI2_OWNPV<10) & (Bu_PT>3000) "\
#                            "& (p1_MC15TuneV1_ProbNNp>0.05) & (p2_MC15TuneV1_ProbNNp>0.05) & (p3_MC15TuneV1_ProbNNp>0.05) "\
#                            "& (abs(L0_M-1115.6)<6) & (Lp_ProbNNp>0.2) & (L0_FDCHI2_ORIVX>45) & (mp1p3<2850) & (mp2p3<2850)"

# cutstrs["normal_pre_DD"] = "& (Bu_DTF_chi2<30) & (Bu_FDCHI2_OWNPV>175) & (Bu_IPCHI2_OWNPV<10) & (Bu_PT>3000)"\
#                            "& (p_MC15TuneV1_ProbNNp>0.05) & ((h1_MC15TuneV1_ProbNNk*h2_MC15TuneV1_ProbNNk)>0.05) "\
#                            "& (abs(L0_M-1115.6)<6) & (Lp_ProbNNp>0.2) & (L0_FDCHI2_ORIVX>45)"

# cutstrs["signal_pre_LL"] = cutstrs["signal_pre_DD"] + " & (L0_FDCHI2_OWNPV>50) & ((L0_ENDVERTEX_Z-Bu_ENDVERTEX_Z)>20)"
# cutstrs["normal_pre_LL"] = cutstrs["normal_pre_DD"] + " & (L0_FDCHI2_OWNPV>50) & ((L0_ENDVERTEX_Z-Bu_ENDVERTEX_Z)>20)"



# # final selection
# cutstrs["signal_final_LL"] = "& (p1_MC15TuneV1_ProbNNp*p2_MC15TuneV1_ProbNNp*p3_MC15TuneV1_ProbNNp>0.60) & (log(Bu_IPCHI2_OWNPV)<1.60)"
# cutstrs["signal_final_DD"] = "& (p1_MC15TuneV1_ProbNNp*p2_MC15TuneV1_ProbNNp*p3_MC15TuneV1_ProbNNp>0.70) & (log(Bu_IPCHI2_OWNPV)<1.80)"



# cutstrs["signal_tighter_LL"] = "& (p1_MC15TuneV1_ProbNNp*p2_MC15TuneV1_ProbNNp*p3_MC15TuneV1_ProbNNp>0.60) & (log(Bu_IPCHI2_OWNPV)<1.60)"
# cutstrs["signal_tighter_DD"] = "& (p1_MC15TuneV1_ProbNNp*p2_MC15TuneV1_ProbNNp*p3_MC15TuneV1_ProbNNp>0.70) & (log(Bu_IPCHI2_OWNPV)<1.80)"
# cutstrs["signal_final_MC_LL"] = cutstrs["signal_final_LL"].replace("ProbNNp", "ProbNNp_corr")
# cutstrs["signal_final_MC_DD"] = cutstrs["signal_final_DD"].replace("ProbNNp", "ProbNNp_corr")

# # final selection for normalization channel
# cutstrs["normal_final_LL"] = "& (p_MC15TuneV1_ProbNNp*h1_MC15TuneV1_ProbNNk*h2_MC15TuneV1_ProbNNk>0.60) & (log(Bu_IPCHI2_OWNPV)<1.60)"
# cutstrs["normal_final_DD"] = "& (p_MC15TuneV1_ProbNNp*h1_MC15TuneV1_ProbNNk*h2_MC15TuneV1_ProbNNk>0.70) & (log(Bu_IPCHI2_OWNPV)<1.80)"
# cutstrs["normal_tighter_LL"] = "& (p_MC15TuneV1_ProbNNp*h1_MC15TuneV1_ProbNNk*h2_MC15TuneV1_ProbNNk>0.60) & (log(Bu_IPCHI2_OWNPV)<1.60)"
# cutstrs["normal_tighter_DD"] = "& (p_MC15TuneV1_ProbNNp*h1_MC15TuneV1_ProbNNk*h2_MC15TuneV1_ProbNNk>0.70) & (log(Bu_IPCHI2_OWNPV)<1.80)"
# cutstrs["normal_final_MC_LL"] = "& (p_MC15TuneV1_ProbNNp_corr*h1_MC15TuneV1_ProbNNk_corr*h2_MC15TuneV1_ProbNNk_corr>0.6) & (log(Bu_IPCHI2_OWNPV)<1.6)"
# cutstrs["normal_final_MC_DD"] = "& (p_MC15TuneV1_ProbNNp_corr*h1_MC15TuneV1_ProbNNk_corr*h2_MC15TuneV1_ProbNNk_corr>0.70) & (log(Bu_IPCHI2_OWNPV)<1.8)"

# cutstrs["normal_nominal_LL"] = "& (p_MC15TuneV1_ProbNNp*h1_MC15TuneV1_ProbNNk*h2_MC15TuneV1_ProbNNk>0.50) & (log(Bu_IPCHI2_OWNPV)<1.70)"
# cutstrs["normal_nominal_DD"] = "& (p_MC15TuneV1_ProbNNp*h1_MC15TuneV1_ProbNNk*h2_MC15TuneV1_ProbNNk>0.55) & (log(Bu_IPCHI2_OWNPV)<2.00)"
# cutstrs["signal_nominal_LL"] = "& (p1_MC15TuneV1_ProbNNp*p2_MC15TuneV1_ProbNNp*p3_MC15TuneV1_ProbNNp>0.5) & (log(Bu_IPCHI2_OWNPV)<1.7)"
# cutstrs["signal_nominal_DD"] = "& (p1_MC15TuneV1_ProbNNp*p2_MC15TuneV1_ProbNNp*p3_MC15TuneV1_ProbNNp>0.55) & (log(Bu_IPCHI2_OWNPV)<2.0)"





