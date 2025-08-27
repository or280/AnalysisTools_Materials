#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// Matches up the frames from Linkam with that from DIC or SXRD for example
// Author: OG Reed
// Date created: 12/10/2023

// Calls the function to match up the force (and hence stress) from the linkam
// ywave becomes strain if matching with DIC, and peakarea if matching with SXRD
function Forcematch(Force,LinkTime,GDATime,ywave,a,b)
	wave force, linktime, GDAtime, ywave
	variable a,b
	string ywname = nameofWave(ywave)

	make /N=(numpnts(force)) /O Stress
	wave Stress
	
	variable i
	for (i=0;i<(numpnts(force));i+=1)
		Stress[i] = force[i]/(a*b)
	endfor
	
	interpolate2 /I=3 /X=GDATime linktime, stress
	wave Stress_CS
	
	variable j
	for (j=0;j<(numpnts(Stress_CS));j+=1)
		if (Stress_CS[j] < 0)
			Stress_CS[j] = NaN
		elseif (Stress_CS[j] > 1000)
			Stress_CS[j] = NaN
		endif
	endfor
	
	display Stress_CS vs ywave
	ModifyGraph axThick=1.5
	ModifyGraph tick=2,mirror=1,standoff=0
	ModifyGraph gFont="Times New Roman",gfSize=11
	ModifyGraph lsize=1.5
	Label left "Stress / MPa"
	
	strswitch(ywname)
		case "Strain":
			Label bottom "Strain / %"
			print("Max strain = " + num2str(wavemax(ywave)))
			print("Max stress = " + num2str(wavemax(Stress_CS)))
			print("End strain = " + num2str(ywave[numpnts(ywave)]))
		
		break
		
		case "Peakarea":
			Label left "Peakarea / a.u."
		
		break
		
	endswitch
	
end

//Matches up the temp from the linkam
//Again, ywave is strain in DIC and peakarea in SXRD
function Tempmatch(Temp,LinkTime,GDATime,ywave)
	wave temp, linktime, GDAtime, ywave
	string ywname = nameofWave(ywave)
	
	interpolate2 /I=3 /X=GDATime linktime, temp
	wave Temp_CS
	
	variable j
	for (j=0;j<(numpnts(Temp_CS));j+=1)
		if (Temp_CS[j] < -155)
			Temp_CS[j] = NaN
		elseif (Temp_CS[j] > 255)
			Temp_CS[j] = NaN
		endif
	endfor
	
	//For DIC
	strswitch(ywname)
		case "Strain":
			make /N=(numpnts(ywave)) /O Strain_adj
		
			variable k
			for (k=0;k<(numpnts(Temp_CS));k+=1)
				Strain_adj[k] = ywave[k]-(0.0008*(Temp_CS[k]-25))
			endfor
		
			display Strain_adj vs Temp_CS
			Label left "Strain / %"
			print("Recovery strain = " + num2str(-wavemin(Strain_adj, pnt2x(Strain_adj,1),pnt2x(Strain_adj,600))))
			print("TWSME strain = " + num2str(wavemax(Strain_adj, pnt2x(Strain_adj,900),pnt2x(Strain_adj,1500))-wavemin(Strain_adj, pnt2x(Strain_adj,1),pnt2x(Strain_adj,900))))
		
		break 
		
		case "SXRD":
			display ywave vs Temp_CS
			Label left "Peakarea / a.u."
			
		break
		
	endswitch
	
	ModifyGraph axThick=1.5
	ModifyGraph tick=2,mirror=1,standoff=0
	ModifyGraph gFont="Times New Roman",gfSize=11
	ModifyGraph lsize=1.5
	Label bottom "Temp / C"
	
end


//Finds the SIM value given the frame at which SIM occured (i.e. converts between frame and acutal value based on interpolation above).
function simfromframe(SIMframes)
	wave SIMframes
	wave Stress_CS
	
	make /N=(numpnts(SIMframes)) /O SSIM
	make /N=(numpnts(SIMframes)) /O Cycle
	wave SSIM, Cycle
	
	variable i
	for (i=0;i<(numpnts(SIMframes));i+=1)
		SSIM[i] = Stress_CS[SIMframes[i]]
		Cycle[i] = i+1
	endfor
	
end

function sumfromframe(SUMframes)
	wave SUMframes
	wave Stress_CS
	
	make /N=(numpnts(SUMframes)) /O SSUM
	make /N=(numpnts(SUMframes)) /O Cycle
	wave SSUM, Cycle
	
	variable i
	for (i=0;i<(numpnts(SUMframes));i+=1)
		SSUM[i] = Stress_CS[SUMframes[i]]
		Cycle[i] = i+1
	endfor
	
end