#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

//Procedure to crop off values from the bottom of a wave if they are smaller than a given value.
//Authors: OG Reed
//Date created: 14/12/2022

//Calls the cropper function which crops any value less than crop off the wave "thingtocrop".
function Cropper(thingtocrop,crop,goesto)
	wave thingtocrop
	variable crop,goesto
	variable i, j
	
	for (i=0; i<numpnts(thingtocrop); i+=1)
		if(thingtocrop[i] < crop)
			//for (j=i; j<=numpnts(thingtocrop); j+=1)
				thingtocrop[i] = goesto
			//endfor
		endif		
	endfor
end


Function IntCropper(thingtocrop,crop)
	wave thingtocrop
	variable crop
	variable i, j, k
	
	for (j=1; j<=20; j+=1)
		for (i=1; i<=numpnts(thingtocrop); i+=1)
			if(abs(thingtocrop[i+1] - thingtocrop[i]) > crop)
				thingtocrop[i] = nan
			endif		
			if(thingtocrop[i-1] == 0 && thingtocrop[i+1] == 0)
				thingtocrop[i] = nan
			endif	
		endfor
	endfor
	
	for (k=1; k<=numpnts(thingtocrop); k+=1)		
		if(thingtocrop[k-1] == 0 && thingtocrop[k+1] == 0)
			thingtocrop[k] = nan
		endif
	endfor
	
	if(thingtocrop[1] == 0)
		thingtocrop[0] = nan
	endif
	
	if(thingtocrop[numpnts(thingtocrop)-2] == 0)
		thingtocrop[numpnts(thingtocrop)] = 0
	endif
	
end
	