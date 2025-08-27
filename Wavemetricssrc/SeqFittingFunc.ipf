‾#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// This contains the fitting functions for "SeqFit"
// It fits the required function in the range specified when calling seqfit.
// Authors: NG Jones, OG Reed
// Date created: 2012


//______________________________________
// Fits gauss peak within a range from centre
//______________________________________
function gaussaroundcentre(data, param, x, centre, range)
	wave data, param, x
	variable centre, range
	variable rangelow, rangehigh, start, stop

	rangelow=centre-range
	rangehigh=centre+range
	
	findlevel /q x rangelow
	start=v_levelx
	findlevel /q x rangehigh
	stop=v_levelx
			
	// Change h=... below to change which parameters are fixed (for bkg etc.)
	funcfit /q /h="00011000" onegaus param data[start,stop] /x=x[start,stop] /D
	
end			

//___________________________________________
// Fits voigt peak within a range from centre
//___________________________________________
function voigtaroundcentre(data, param, x, centre,range)
	wave data, param, x
	variable centre, range
	variable rangelow, rangehigh, start, stop
	
	rangelow=centre-range
	rangehigh=centre+range
	
	findlevel /q x rangelow
	start=v_levelx
	findlevel /q x rangehigh
	stop=v_levelx

	// Change h=... below to change which parameters are fixed (for bkg etc.)
	Funcfit /q /h="000110000" onevoigt param data[start,stop] /x=x[start,stop] /d
	
end

			
//___________________________________________
// Fits double gauss peak within a range from centre
//___________________________________________
function twogaussaroundcentre(data, param, x, centre, range)
	wave data, param, x
	variable centre, range
	variable rangelow, rangehigh, start, stop
	wave areas, param
	
	rangelow=centre-range
	rangehigh=centre+range
	
	findlevel /q x rangelow
	start=v_levelx
	findlevel /q x rangehigh
	stop=v_levelx
	
	// Change h=... below to change which parameters are fixed (for bkg etc.)
	funcfit /q /h="00011000000" twogaus param data[start,stop] /x=x[start,stop] /D

end

//___________________________________________
// Fits double voigt peak within a range from centre
//___________________________________________
function twovoigtaroundcentre(data, param, x, centre,range)
	wave data, param, x
	variable centre, range
	variable rangelow, rangehigh, start, stop
	wave T_Constraints
	
	rangelow=centre-range
	rangehigh=centre+range
	
	findlevel /q x rangelow
	start=v_levelx
	findlevel /q x rangehigh
	stop=v_levelx

	// Change h=... below to change which parameters are fixed (for bkg etc.)
	Funcfit /q /h="0001100000000" twovoigt param data[start,stop] /x=x[start,stop] /D
	
end


//___________________________________________
// Fits triple gauss peak within a range from centre
//___________________________________________
function threegaussaroundcentre(data, param, x, centre, range)
	wave data, param, x
	variable centre, range
	variable rangelow, rangehigh, start, stop
	wave areas, param
	
	rangelow=centre-range
	rangehigh=centre+range
	
	findlevel /q x rangelow
	start=v_levelx
	findlevel /q x rangehigh
	stop=v_levelx
	
	// Change h=... below to change which parameters are fixed (for bkg etc.)
	funcfit /q /h="00011000000000" threegaus param data[start,stop] /x=x[start,stop] /D

	
end

//___________________________________________
// Fits four gauss peaks within a range from centre
//___________________________________________
function fourgaussaroundcentre(data, param, x, centre, range)
	wave data, param, x
	variable centre, range
	variable rangelow, rangehigh, start, stop
	wave areas, param
	
	rangelow=centre-range
	rangehigh=centre+range
	
	findlevel /q x rangelow
	start=v_levelx
	findlevel /q x rangehigh
	stop=v_levelx
	
	// Change h=... below to change which parameters are fixed (for bkg etc.)
	funcfit /q /h="00011000000000000" fourgaus param data[start,stop] /x=x[start,stop] /D

end	

//___________________________________________
// Fits 4 voigt peaks within a range from centre
//___________________________________________
function fourvoigtaroundcentre(data, param, x, centre,range)
	wave data, param, x
	variable centre, range
	variable rangelow, rangehigh, start, stop
	
	rangelow=centre-range
	rangehigh=centre+range
	
	findlevel /q x rangelow
	start=v_levelx
	findlevel /q x rangehigh
	stop=v_levelx

	// Change h=... below to change which parameters are fixed (for bkg etc.)
	Funcfit /q /h="000110000000000000000" fourvoigt param data[start,stop] /x=x[start,stop] /d
	
end
