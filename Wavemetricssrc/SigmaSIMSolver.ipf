#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

//Procedure to calculate sigma_SIM for a series of cyclic data.
//Authors: CEP Talbot, OG Reed

//Calls the Sigma_SIM solver for each cycle forming a wave containing the Sigma_SIM of each plot
function CyclesSigmaSIM(cycles, comp)
	variable cycles
	string comp
	
	Make/O/N=(cycles) $(comp+"_SigmaSIMS")
	Make/O/N=(cycles) $(comp+"_StrainSIMS")
	wave Sigma_SIMA = $(comp+"_SigmaSIMS")
	wave Strain_SIMA = $(comp+"_StrainSIMS")
	Make/O/N=(cycles) cycleno
	wave cycleno
	variable startindex = 30 //This accounts for the tail of the previous cycle being included which affects cycles with large amount of recovery
	
	variable i = 0 //Dummy Variable
	variable sigmasimindex
	for(i=0; i<(cycles); i+=1)
		wave stresswave = $(comp+"_StressCycle_"+num2str(i+1))
		wave strainwave = $(comp+"_StrainCycle_"+num2str(i+1))
		Sigma_SIMA[i] = round(FindSigmaSIM3(stresswave, strainwave, comp, startindex))
		Strain_SIMA[i] = FindStrainSIM(strainwave, sigmasimindex)
		cycleno[i] = i+1
	endfor
	
	killwindows()
	Display Sigma_SIMA vs cycleno
	SetAxis left 0,200;DelayUpdate
	SetAxis bottom 0,11;DelayUpdate
	ModifyGraph nticks(bottom)=10,minor=1,sep(left)=10;DelayUpdate
	Label left "σ\\BSIM";DelayUpdate
	Label bottom "Cycle Number"
	ModifyGraph mode=3,rgb=(0,0,0)
	ModifyGraph mrkThick=1
	ModifyGraph width=432,height={Aspect,1}
	ModifyGraph gfSize=0
	ModifyGraph axOffset(left)=-3, axOffset(bottom)=-1
	ModifyGraph tick=2,mirror=1,axThick=2,standoff=0
	//CurveFit Power Sigma_SIMA /D
	
end


// Function to calculate sigma_SIM for each cycle. This one uses the intersection of two lines on the plot
function FindSigmaSIM1(stresswave, strainwave, comp, startindex)
	wave stresswave, strainwave
	string comp //AlloyComposition
	variable startindex
	variable length = numpnts(stresswave)/2
	
	Make/N=(length-startindex) stresswave_2
	Make/N=(length-startindex) strainwave_2
	
	Duplicate/O/R=[startindex,length] stresswave, stresswave_2 //Stresswave and strainwave include the tail end of the previous cycle, this cuts that out
	Duplicate/O/R=[startindex,length] strainwave, strainwave_2
	
	Differentiate stresswave_2 /X=strainwave_2 /D=DifferentiatedWave	//Find the R2 values as a function of stress
	Smooth 500, DifferentiatedWave
	wavestats/Q DifferentiatedWave
	variable definepeak = V_max-0.6*(V_max-V_min)
	
	FindPeak /M=(definepeak)/N/Q/P DifferentiatedWave
	//FindPeak /N/Q/P DifferentiatedWave
	
	if(V_flag == 0)
		variable inflection = V_PeakLoc
	else
		inflection = numpnts(stresswave_2)
	endif
	
	make/O/N=2 fit1
	make/O/N=2 fit2

	CurveFit/Q line stresswave_2[inflection-20,inflection] /X=strainwave_2 /D 
	wave W_coef = W_coef
	duplicate/O W_coef, fit1
	
	CurveFit/Q line stresswave_2[30,60] /X=strainwave_2 /D 
	duplicate/O W_coef, fit2

	variable A = fit1[0]-fit2[0]
	variable B = fit1[1]-fit2[1]
	variable root = -A/B 
	//print inflection, root
	variable sigmasimindex = fit1[0]+fit1[1]*root
	//print sigmasimindex
	//killwaves strainwave_2, stresswave_2, DifferentiatedWave, fit1, fit2, coefs, rootfinder
	killvariables root, 
	killvariables inflection, 
	killvariables V_chisq, 
	killvariables V_endChunk, 
	killvariables V_endLayer, 
	killvariables V_endRow, 
	killvariables V_Flag, 
	killvariables V_iterations, 
	killvariables V_LeadingEdgeLoc,
	killvariables V_nheld, 
	killvariables V_npnts, 
	killvariables V_nterms, 
	killvariables v_numINFs, 
	killvariables V_numNaNs, 
	killvariables V_PeakLoc, 
	killvariables V_PeakVal, 
	killvariables V_PeakWidth, 
	killvariables V_Pr, 
	killvariables V_q, 
	killvariables V_Rab, 
	killvariables V_r2, 
	killvariables V_siga, 
	killvariables V_sigb, 
	killvariables V_startChunk, 
	killvariables V_startCol, 
	killvariables V_startLayer, 
	killvariables V_startRow, 
	killvariables V_TrailingEdgeLoc
	
	return sigmasimindex

end

// This one again is the intersection of two lines, done slightly differently
function FindSigmaSIM2(stresswave, strainwave, comp, startindex)
	wave stresswave, strainwave
	string comp //AlloyComposition
	variable startindex
	variable length = numpnts(stresswave)/2
	
	Make/N=(length-startindex) stresswave_2
	Make/N=(length-startindex) strainwave_2
	make/n=(numpnts(stresswave_2)) Gradients
	
	Duplicate/O/R=[startindex,length] stresswave, stresswave_2 //Stresswave and strainwave include the tail end of the previous cycle, this cuts that out
	Duplicate/O/R=[startindex,length] strainwave, strainwave_2
	
	variable j
	variable skip = 20
	for (j=skip;j<=numpnts(stresswave_2);j+=1)
		CurveFit/Q line stresswave_2[0,j] /X=strainwave_2 /D 
		wave W_coef = W_coef
		Gradients[j-skip] = W_coef[0]
	endfor

	differentiate Gradients /D=DiffGradient
	smooth 25, DiffGradient
	wavestats/Q/R=(skip,numpnts(diffgradient)) DiffGradient
	variable maxchangeingrad = V_maxloc
	
	make/O/N=2 fit1
	make/O/N=2 fit2

	CurveFit/Q line stresswave_2[maxchangeingrad+50,maxchangeingrad] /X=strainwave_2 /D 
	wave W_coef = W_coef
	duplicate/O W_coef, fit1
	
	CurveFit/Q line stresswave_2[30,60] /X=strainwave_2 /D 
	duplicate/O W_coef, fit2

	variable A = fit1[0]-fit2[0]
	variable B = fit1[1]-fit2[1]
	variable root = -A/B 
	//print maxchangeingrad, root
	variable sigmasimindex = fit1[0]+fit1[1]*root
	//print sigmasimindex
	//killwaves strainwave_2, stresswave_2, DifferentiatedWave, fit1, fit2, coefs, rootfinder
	killvariables root, 
	killvariables inflection, 
	killvariables V_chisq, 
	killvariables V_endChunk, 
	killvariables V_endLayer, 
	killvariables V_endRow, 
	killvariables V_Flag, 
	killvariables V_iterations, 
	killvariables V_LeadingEdgeLoc,
	killvariables V_nheld, 
	killvariables V_npnts, 
	killvariables V_nterms, 
	killvariables v_numINFs, 
	killvariables V_numNaNs, 
	killvariables V_PeakLoc, 
	killvariables V_PeakVal, 
	killvariables V_PeakWidth, 
	killvariables V_Pr, 
	killvariables V_q, 
	killvariables V_Rab, 
	killvariables V_r2, 
	killvariables V_siga, 
	killvariables V_sigb, 
	killvariables V_startChunk, 
	killvariables V_startCol, 
	killvariables V_startLayer, 
	killvariables V_startRow, 
	killvariables V_TrailingEdgeLoc
	
	return sigmasimindex

end


// This uses a set deviation in stress away from the fitted line to the first few points
function FindSigmaSIM3(stresswave, strainwave, comp, startindex)
	wave stresswave, strainwave
	string comp // Alloy Composition
	variable startindex
	variable length = numpnts(stresswave)/2
	variable j, sigmasimindex, dev = 5
	
	// Stresswave and strainwave include the tail end of the previous cycle, this cuts that out
	wave stresswave_2,strainwave_2
	Make/O/N=(length-startindex) stresswave_2
	Make/O/N=(length-startindex) strainwave_2
	Duplicate/O/R=[startindex,length] stresswave, stresswave_2
	Duplicate/O/R=[startindex,length] strainwave, strainwave_2
	
	CurveFit/Q line stresswave_2[10,55] /X=strainwave_2 /D
	wave W_coef
	make/O/N=2 fit1 = W_coef
	wave fit1adj
	make/O/N=2 fit1adj = {fit1[0],fit1[1]+fit1[0]}
	
	// Make a wave which gives the difference between the fit and real curve
	wave diff
	Make/O/N=(length-startindex) diff
	for (j=20;j<numpnts(stresswave_2);j+=1)
		diff[j] = fit1[0] + fit1[1]*strainwave_2[j] - stresswave_2[j]
		//print diff[j]
		if (diff[j] > dev && diff[j-2] > dev && diff[j-4] > dev && diff[j-6] > dev && diff[j-8] > dev)
			sigmasimindex = stresswave_2[j-4]
			break
		endif
	endfor
	
	//killwaves strainwave_2, stresswave_2, diff
	print sigmasimindex
	return sigmasimindex

end

// This finds the sigmasim (and yield strength) by fitting a straight line to the linear region, 
// defined by p1 and p2, offsetting by a given amount and then finding the intersection
function FindSigmaSIMenv(p1, p2, offset, ywave, xwave)
	variable p1, p2, offset
	wave ywave, xwave

	variable i

	CurveFit/Q line ywave[p1,p2] /X=xwave /D
	wave W_coef
	make/O/N=(numpnts(xwave)) fitadjy
	make/O/N=(numpnts(xwave)) diff
	
	
	for (i=0;i<(numpnts(xwave));i+=1)
		fitadjy[i] = W_coef[0]+(xwave[i]-offset)*W_coef[1]
		diff[i] = ywave[i]-fitadjy[i]
	endfor
	
	findlevel /P /Q diff, 0
	print "Strain = " + num2str(xwave[V_LevelX])
	print "Stress = " + num2str(ywave[V_LevelX])
	
end



function FindStrainSIM(strainwave, sigmasimindex)
	wave strainwave
	variable sigmasimindex
	
	if(sigmasimindex == 0)
		return NaN
	endif

	variable Strain_SIM = strainwave[sigmasimindex]
	
	return Strain_SIM
	
end




function killwindows() //copied from the WaveMettrics engineer post on the forums

string list = winlist("*", ";", "WIN:2")
variable i

for (i=0;i<itemsinlist(list); i+=1)

string windowname = stringfromlist (i,list, ";")

killwindow $windowname

endfor


end