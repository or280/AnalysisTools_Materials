#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later
#include <WindowBrowser>

//Code to plot ratchet or cyclic tensile tests, separating a single stress and strain file into separate waves for easy analysis
//Authors: CEP Talbot, N Church, OG Reed

// Zeroes Strain
function ZeroStrain(strainwave)
	wave strainwave
	
	Duplicate strainwave, Strain
	Strain -= strainwave[0]
	
end

// Separates ratchet test into individual waves for each cycle, cycles is number of cycles performed
// If a ratchet test, set cycles to 0 and the script will calculate the number of cycles
function PlotCycles(strainwave1, stresswave, cycles)
	wave strainwave1, stresswave
	variable cycles
	
	Duplicate/O strainwave1, strainwave //now zeros strain in main function overwriting previous strain wave
	strainwave -= strainwave1[0]
	
	// Caluclates no. of cycles for a ratchet test. Change the dividing no. if stress steps are different
	if (cycles == 0)
		cycles = floor((wavemax(stresswave)-5)/25)
		print cycles
	endif
	
	wave cyclepositions = FindCycles(stresswave, cycles)
	
	variable i
	string NewStressWave = "Stresscycle"
	string NewStrainWave = "Straincycle"
	Make/O/N=(cycles+1) Cycleno
	Make/O/N=(cycles+1) MinStrain
	Make/O/N=(cycles+1) MaxStrain
	Make/O/N=(cycles+1) DeltaStrain
	Make/O/N=(cycles+1) MaxStress
	Make/O/N=(cycles+1) MinStress
	wave Cycleno, MaxStrain, MinStrain, DeltaStrain, MaxStress, MinStress
	
	// Separates stress waves and strain waves independently into separate waves by copying from the large stress and strain files 
	for (i=1;i<cycles+1; i+=1)
		Duplicate/O/R=[cyclepositions[i-1], cyclepositions[i]] stresswave, $(NewStressWave+num2str(i))
		Duplicate/O/R=[cyclepositions[i-1], cyclepositions[i]] strainwave, $(NewStrainWave+num2str(i))
	
	
		//Plot data onto graph and edit graph
		if (i<2)
			display $(NewStressWave+num2str(i)) vs $(NewStrainWave+num2str(i))
			label bottom "Strain / %"
			label left "Tensile Stress / MPa"
		endif
		
		if (i>1)
			appendtograph $(NewStressWave+num2str(i)) vs $(NewStrainWave+num2str(i))
		endif
		
		//General Graph Aesthetics, easily changed
		ApplyColourTableToTopGraph("blue")
		ModifyGraph lsize=1.5
		ModifyGraph tick=2,mirror=1,standoff=0
		SetAxis left 0,900
		SetAxis bottom 0,4.5
		ModifyGraph gFont="Times New Roman", gfSize=11
		ModifyGraph width=300,height=300
		ModifyGraph margin(left)=34,margin(bottom)=28,margin(right)=8,margin(top)=8
		
	endfor
	
	// Finds the strains at the top and end of each cycle
	
	Cycleno[0] = 0
	MinStrain[0] = 0
	MaxStrain[0] = NaN
	MinStress[0] = 25
	MaxStress[0] = 25
	DeltaStrain[0] = NaN
	
	for(i=1; i<cycles+1; i+=1)
		Cycleno[i] = i
		MaxStrain[i] = WaveMax($(NewStrainWave+num2str(i)))
		MinStrain[i] = strainwave[cyclepositions[i]]	
		MaxStress[i] = WaveMax($(NewStressWave+num2str(i)))
		MinStress[i] = stresswave[cyclepositions[i]]
		DeltaStrain[i] = MaxStrain[i]-MinStrain[i]	
		
		killvariables V_Flag
		killvariables V_row
		killvariables V_value
		killvariables start
		killvariables finish
		killvariables target
															
	endfor
	
	Duplicate/O DeltaStrain, ZeroedDeltaStrain
	ZeroedDeltaStrain -= DeltaStrain[1]

	// Plotting various other variables
	
	PlotStrains(Minstrain, maxstrain, deltastrain, maxstress, cycleno)
	PlotEnv(Maxstress, Maxstrain)
	PlotArea(cycles)
	
	killwaves strainwave
	
end


//Colours traces progressively based on a designated colourtable, taken from online.
Function ApplyColourTableToTopGraph(ctabname)
    String ctabname

    String graphName = WinName(0, 1)
    if (strlen(graphName) == 0)
        return -1
    endif

    Variable numTraces = ItemsInList(TraceNameList(graphName,";",3))

    if (numTraces <= 0)
        return -1
    endif
   
    Variable denominator= numTraces-1
    if( denominator < 1 )
        denominator= 1    // avoid divide by zero, use just the first color for 1 trace
    endif

    ColorTab2Wave $ctabname // creates M_colors
    Wave M_colors
    //reverse M_colors
    Variable numRows= DimSize(M_colors,0)
    Variable red, green, blue
    Variable i, index
    for(i=0; i<numTraces; i+=1)
        index = round(i/denominator * (numRows-40))  // spread entire color range over all traces.
        //ModifyGraph/W=$graphName rgb[numTraces-i]=(M_colors[index][0], M_colors[index][1], M_colors[index][2]) //Use if wanting reverse colour table
        ModifyGraph/W=$graphName rgb[i]=(M_colors[index][0], M_colors[index][1], M_colors[index][2]) //Use if wanting standard colour table
        
    endfor
    return 0
End


// Finds the indices of the points where a new cycle starts in order to separate the waves in SeparateCycles function. Returns wave with indices of the stresses where each cycle begins.
function/WAVE FindCycles(stresswave, cycles)
	wave stresswave
	variable cycles
	variable length = numpnts(stresswave)
	
	Make/O/N=(cycles+1) cyclepositions
	
	variable i
	variable cyclecount
	for(i=0; i<length; i+=1)
	
		if(stresswave[i] < 26) // 25MPa being the chosen lower limit for the tests, change this if value is different
			cyclepositions[cyclecount] = i
			cyclecount += 1
			i += 50 // Adds 50 to index to skip to a stress sufficiently high to be above 25MPa in the next cycle
			
		endif
				
	endfor
	
	cyclepositions[cycles+1] = length
	
	return cyclepositions
	
end

// Find and plot the hysteresis area of each cycle
function PlotArea(cycles)
	variable cycles
	
	wave HysArea, Cycleno
	Make/O/N=(cycles) HysArea
	
	HysArea[0] = NaN
	
	variable i = 1
	for(i=1; i<(cycles+1); i+=1)
		
		Integrate/meth=1 $("Stresscycle"+num2str(i))/X=$("Straincycle"+num2str(i))/D=$("Area"+num2str(i))
	
		wave Integral = $("Area"+num2str(i))
		HysArea[i] = Integral[numpnts(Integral)]
		
		killwaves Integral
		
	endfor
	
	HysArea = HysArea/100
	Display HysArea vs Cycleno
	Label left "Hysteresis Area / MJ m\\S-3"
	Label bottom "Cycle Number"
	ModifyGraph lsize=1.5
	ModifyGraph tick=2,mirror=1,standoff=0
	SetAxis left 0,3
	SetAxis bottom 0,35
	ModifyGraph gFont="Times New Roman", gfSize=11
	ModifyGraph width=300,height=300
	ModifyGraph margin(left)=34,margin(bottom)=28,margin(right)=8,margin(top)=8
	
End


function PlotStrains(Minstrain, maxstrain, deltastrain, maxstress, cycleno)
	wave Minstrain, maxstrain, deltastrain, maxstress, cycleno

	Display MinStrain vs Maxstress
	appendtograph deltastrain vs Maxstress
	//appendtograph $(Comp+"_DeltaStrain") vs CyclesCount		
	ModifyGraph lsize=1.5
	SetAxis bottom 0,900
	SetAxis left -0.1,3
	ModifyGraph mirror=1, standoff=0, tick=2
	ModifyGraph gFont="Times New Roman", gfSize=11
	ModifyGraph width=300,height=300
	ModifyGraph margin(left)=34,margin(bottom)=28,margin(right)=8,margin(top)=8
	ModifyGraph lstyle(DeltaStrain)=3,rgb(DeltaStrain)=(26214,0,0)
	//ModifyGraph rgb($(Comp+"_MinStrain"))=(1,16019,65535)
	//label bottom "Maximum Tensile Stress (MPa)"
	label bottom "Max Stress / MPa"
	label left "Strain / %"
	Legend/C/N=text0/J/F=0/A=LT/X=5.00/Y=5.00 "\\s(#0) Accumulated Strain\r\\s(#1) Delta Strain\r"
	
end


// Generate and plot the envelope
function Plotenv(MaxStress, Maxstrain)
	wave MaxStress, MaxStrain

	Interpolate2/T=2/N=500/E=2/Y=MaxStress_CS/X=MaxStrain_CS MaxStrain,MaxStress
	
	display MaxStress_CS vs MaxStrain_CS
	label bottom "Strain / %"
	label left "Tensile Stress / MPa"
	ModifyGraph lsize=1.5
	ModifyGraph tick=2,mirror=1,standoff=0
	SetAxis left 0,900
	SetAxis bottom 0,4.5
	ModifyGraph gFont="Times New Roman", gfSize=11
	ModifyGraph width=300,height=300
	ModifyGraph margin(left)=34,margin(bottom)=28,margin(right)=8,margin(top)=8
	
end