#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later


function valueatfraction(fraction, list)
	wave list
	variable fraction
	
	sort list, list
	
	variable length, answer
	
	length = numpnts(list)
	answer = list[(length*fraction)]
	
	print  answer
	
end

