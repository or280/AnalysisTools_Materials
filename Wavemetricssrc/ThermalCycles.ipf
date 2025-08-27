#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// Script for analysing thermal cycles, and works for DSC, SXRD or DIC.
// Includes functions for splitting cycles and plotting seperately, and then finding transformation temperatures, enthalpies and hysteresis widths.
// Authors: OG Reed
// Date created: 25/05/2023

// Calls the function to split and plot. 
// Note that this only really works for DSC and SXRD as the DIC has more variable lengths.
function Thermalcycles(ywave, temp, cycles)
	wave ywave, temp
	variable cycles
	
	variable i
	variable length = numpnts(ywave)
	variable interval = length/cycles
	print interval
	
	for (i=1;i<=cycles;i+=1)
		Duplicate/O/R=[(i-1)*interval, i*interval] ywave, $(nameofwave(ywave)+"_"+num2str(i))
		Duplicate/O/R=[(i-1)*interval, i*interval] Temp, $("Temp_"+num2str(i))
	endfor
	
	//Plot data onto graph
	display $(nameofwave(ywave)+"_1") vs $("Temp_1")
	
	if(cycles > 1)
		for (i=2;i<=cycles;i+=1)
			appendtograph $(nameofwave(ywave)+"_"+num2str(i)) vs $("Temp_"+num2str(i))
		endfor
	endif

	//General Graph Aesthetics, easily changed
	label bottom "Temperature / C"
	label left "y"
	ModifyGraph lsize=2
	SetAxis/A
	ModifyGraph mirror=1,axThick=2,standoff=0
	ModifyGraph tick=2
	ModifyGraph width=1500,height=800
	ModifyGraph axOffset(left)=-3,axOffset(bottom)=-1

end



// Procedure to calculate the intersect point of two lines fitted to sections of a plot.
// Used to find temperatures from any given plot.

// Calls the intersect solver, given the 4 points defining the two lines that intersect.
function IntersectorARB(ywave,xwave,cycle,d1,d2,d3,d4)
	variable d1,d2,d3,d4,cycle
	wave ywave,xwave

	variable interval = numpnts($(nameofwave(ywave)+"_1"))
	
	d1 = d1+interval*(cycle-1)
	d2 = d2+interval*(cycle-1)
	d3 = d3+interval*(cycle-1)
	d4 = d4+interval*(cycle-1)
	
	CurveFit/Q line ywave[d1,d2] /X=xwave /D 
	wave W_coef = W_coef
	wave W_sigma
	duplicate/O W_coef, fit1
	//print fit1
	
	CurveFit/Q line ywave[d3,d4] /X=xwave /D 
	duplicate/O W_coef, fit2
	//print fit2

	variable A = fit1[0]-fit2[0]
	variable B = fit1[1]-fit2[1]
	variable answer = -A/B
	
	print ("x value = " + num2str(answer))
	print ("y value = " + num2str(fit1[0]+fit1[1]*answer))
	killwaves fit1, fit2, W_coef, W_sigma
	
end



// Function to find the average intrinsic hysteresis width of an SXRD cycle.
// To do this it finds the difference of points either side of the loop and finds the average seperation.
function intrinsichys(cycle,length)
	variable cycle, length
	
	variable midvalue, midp1, midp2, accu, i, width, offset
	wave p1area, p1area_1, p1area_2, temp
	
	if(cycle==1)
		midvalue = wavemax(p1area_1)/2
		findlevel /EDGE=1 /P /q p1area midvalue
		midp1 = v_levelx
		print midp1
		findlevel /EDGE=2 /P /q p1area midvalue
		midp2 = v_levelx
		//midp2 = 1212
		print midp2
		
		accu = 0
		for (i=-length/2;i<=length/2;i+=1)
			accu+= temp[midp2+i]-temp[midp1+i]
		endfor
		
		width = accu/(length+1)
		print width
		
	endif
	
	if(cycle==2)
		offset = numpnts(p1area_1)
		midvalue = wavemax(p1area_2)/2
		findlevel /EDGE=1 /P /q p1area_2 midvalue
		//midp1 = v_levelx
		midp1 = 92
		findlevel /EDGE=2 /P /q p1area_2 midvalue
		//midp2 = v_levelx
		midp2 = 313
		
		accu = 0
		for (i=-length/2;i<=length/2;i+=1)
			accu+= temp[offset+midp2+i]-temp[offset+midp1+i]
		endfor
		
		width = accu/(length+1)
		print width
		
	endif
	
end



//Calls the enthalpy solver, which finds the area inside a DSC peak, and hence the heat of enthalpy.
function Enthalpy(cycle,rate,d1,d2,d3,d4)
	variable rate,d1,d2,d3,d4,cycle
	wave Heatflow, Temp, W_coef, W_fitConstants
	variable i,d2n,d3n
	variable peakarea, bkgarea, Finalarea
	
	variable interval = numpnts($(nameofwave(Heatflow)+"_1"))
	
	d1 = d1+interval*(cycle-1)
	d2 = d2+interval*(cycle-1)
	d3 = d3+interval*(cycle-1)
	d4 = d4+interval*(cycle-1)

	Duplicate/O/R=[d1,d4] Heatflow, mask
	Duplicate/O/R=[d1,d4] Temp, tempsec
	
	//Remove the peak from the mask wave such that it is just the background. Then fit to this.
	d2n=d2-d1
	d3n=d3-d1
	
	for (i=0;i<=numpnts(mask);i+=1)
		if(i>d2n && i<d3n)
			mask[i] = NaN
		endif
	endfor
	
	CurveFit /Q poly_XOffset 3, mask /X=tempsec /D
	wave W_coef, W_fitConstants, W_sigma, fit_mask, W_ParamConfidenceInterval
	
	//Find the areas underneath the background between d2 and d3, and then the area underneath the heatflow wave between d2 and d3
	Duplicate/O/R=[d2,d3] Heatflow, heattofit
	Duplicate/O/R=[d2,d3] Temp, temptofit
	Duplicate/O/R=[d2,d3] Heatflow, bkg
	
	for (i=0;i<numpnts(temptofit);i+=1)
		bkg[i] = W_coef[0]+W_coef[1]*(temptofit[i]-W_fitConstants[0])+W_coef[2]*(temptofit[i]-W_fitConstants[0])^2
	endfor
	
	peakarea = areaXY(temptofit,heattofit)
	bkgarea = areaXY(temptofit,bkg)
	
	Finalarea = (peakarea - bkgarea)/(rate/60)
	print Finalarea
	
	killwaves W_coef,W_fitConstants,W_sigma,fit_mask,W_ParamConfidenceInterval,tempsec,mask
end