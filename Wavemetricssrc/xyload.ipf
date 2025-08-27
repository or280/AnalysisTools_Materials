#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// Loads all SXRD files into igor as waves d0, d1 etc.
// Authors: NG Jones, CEP Talbot, O Reed

// Loads in a given type of file, given by type in the function
Function loadfiles(expno, lo, hi, type)
	string expno, type
	variable lo,hi
	variable i
	string filename, datname,pathname, filenum
	string base = expno+"_ascii"
	
	pathinfo home
	NewPath/O/q rawpath, S_Path+base

	strswitch(type)
		case "xy":
		
		for (i=lo; i<=hi; i+=1)
		
			// Adds preceeding zeros to file name if needed
			if (i <10)
				filenum="0000"+num2str(i)
			elseif (i>9  &&  i<100)
				filenum="000"+num2str(i)
			elseif (i>99  && i <1000)
				filenum="00"+num2str(i)
			elseif (i>999  && i <10000)
				filenum="0"+num2str(i)
			elseif (i>9999  && i <100000)
				filenum=num2str(i)
			elseif (i>99999 && i<1000000)
				filenum="0"+num2str(i)
			else
				filenum=num2str(i)
			endif

			filename=expno+"_"+filenum+".xy"
			//filename=expno+"_["+num2str(i)+",;1679,;1475]_"+filenum+".xy"
	
			if (i==lo)
				LoadWave/q /G/p=rawpath/n=datin /V={"\t"," $",0,0}/L={0,0,0,0,1} /B="c=1,f=0;c=1,f=-1,n='_skip_';" filename
				wave datin0
				duplicate/O datin0 ang
				killwaves datin0
			endif
			
				datname="d"+num2str(i)
				loadwave /q /G/p=rawpath/N=datin /L={0,0,0,1,0}/V={"\t"," $",0,0} /B="c=1,f=0,n='_skip_';c=1,f=-1;" filename
				wave datin0
				duplicate/O datin0 $datname
				killwaves datin0
				
			endfor
		
		break
		
		case "chi":
			filename=expno+filenum+".chi"

			if (i==lo)
				LoadWave/q /G/p=rawpath/n=datin /V={" "," $",0,0}/L={0,4,0,0,1} /B="c=1,f=0;c=1,f=-1,n='_skip_';" filename
				duplicate datin0 ang
				killwaves datin0
			endif
		
				datname="d"+num2str(i)
				loadwave /q /G/p=rawpath/N=datin /L={0,4,0,1,0}/V={" "," $",0,0} /B="c=1,f=0,n='_skip_';c=1,f=-1;" filename
				duplicate datin0 $datname
				killwaves datin0
				
		break 
		
		endswitch
end


// Same as above for xy files but now corrects for if there are missing frames by duplicating the previous one
Function xyloadNov22(expno, lo, hi)
	string expno
	variable lo,hi
	variable i, j
	string filename, datname,pathname, filenum
	string prevfilenum = "00000"
	string base = expno+"_FR"
	variable reset = 0
	wave datin0
	
	pathinfo home
	NewPath/O/q rawpath, S_Path+base
	
	make/n=0 MissingPoints

	for(j=lo; j<=hi; j+=1)
	for (i=0; reset==0 && j<=hi; i+=1)
	
//	SECTION START: adds preceeding zeros to file name if needed
	if (i <10)
			filenum="0000"+num2str(i)
		elseif (i>9  &&  i<100)
			filenum="000"+num2str(i)
		elseif (i>99  && i <1000)
			filenum="00"+num2str(i)
		elseif (i>999  && i <10000)
			filenum="0"+num2str(i)
		elseif (i>9999  && i <100000)
			filenum=num2str(i)
//		elseif (i>99999 && i<100000)
//		filenum="0"+num2str(i)
		else
			filenum=num2str(i)
		endif
			
	filename=expno+"_["+num2str(j)+",;1679,;1475]_"+filenum+".xy"

//	END OF SECTION

		//Test to see if file is there, due to missing files from some experiments
		variable ref
		Open/r/p=rawpath/z ref filename 
		
		if(V_Flag != 0)
			//print filename
			filename=expno+"_["+num2str(j-1)+",;1679,;1475]_"+prevfilenum+".xy"
			
			//Saves missing filenum to a wave for tracking 
			Redimension/N=(numpnts(MissingPoints)+1) MissingPoints
			MissingPoints[numpnts(MissingPoints)] = j
			i=-1
		else
			reset=0
			prevfilenum = filenum
		endif
		
		close/A
		
		//Load in file
		if (i==lo)
			LoadWave/q /G/p=rawpath/n=datin /V={"\t"," $",0,0}/L={0,0,0,0,1} /B="c=1,f=0;c=1,f=-1,n='_skip_';" filename
			duplicate datin0 ang
			killwaves datin0
		endif
			datname="d"+num2str(j)
			loadwave /q /G/p=rawpath/N=datin /L={0,0,0,1,0}/V={"\t"," $",0,0} /B="c=1,f=0,n='_skip_';c=1,f=-1;"filename
			duplicate datin0 $datname
			killwaves datin0
			j = j+1
	endfor
	endfor
end


// Function to adjust for the ring current if it hasn't been done in processing
// Author: OG Reed

function currentadj(lo,hi,current)
wave current
variable lo, hi
variable i
string datname

	for (i=lo; i<=hi; i+=1)
		datname = "d"+num2str(i)
		wave ref = $datname
		
		ref *= current[i]
		ref /= 300
	endfor
end
		



// Not sure exactly what these do??
function cakeload(base)
	string base
	string filename, datname
	variable binsize,numcols,i
	wave /z datin0
	filename=base+".spr"
	NewPath/O/q rawpath
	LoadWave /m /q /p=rawpath /J /N=datin /K=1 /V={" "," $",0,0}/L={0,1,0,1,0} filename	
	cakesplit()
end

function cakesplit()
	wave /z base
	variable binsize,numcols,i
	string datname
	wave datin0
		
	binsize = 360/dimsize(datin0,0)
	numcols = dimsize(datin0,1)
	
	for (i=0;i<=(360/binsize);i+=1)
		datname="d"+num2str(i)
		make /o/n=(numcols) tempwave
		tempwave= datin0[i] [p]
		duplicate /o tempwave $datname
		killwaves tempwave
	endfor
	killwaves datin0
end


