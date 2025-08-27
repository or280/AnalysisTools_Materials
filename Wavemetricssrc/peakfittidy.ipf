#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// Moves all fits and variables from a sequential peak fit into a single folder.
// Authors: NG Jones

Function PeakFitTidy(peak, lo, hi)
	string peak
	variable lo, hi
	variable i, j
	
	NewDataFolder/O $(peak+"_Fit")
	
	variable V_startRow,V_startLayer,V_startCol,V_startChunk,V_numNaNs,V_numINFs,V_nterms,V_npnts,V_nheld,V_endRow,V_endLayer,V_endCol,V_endChunk,V_chisq
	MoveVariable V_startRow, :$(peak+"_Fit"):
	MoveVariable V_startLayer, :$(peak+"_Fit"):
	MoveVariable V_startCol, :$(peak+"_Fit"):
	MoveVariable V_startChunk, :$(peak+"_Fit"):
	MoveVariable V_numNaNs, :$(peak+"_Fit"):
	MoveVariable V_numINFs, :$(peak+"_Fit"):
	MoveVariable V_nterms, :$(peak+"_Fit"):
	MoveVariable V_npnts, :$(peak+"_Fit"):
	MoveVariable V_nheld, :$(peak+"_Fit"):
	MoveVariable V_endRow, :$(peak+"_Fit"):
	MoveVariable V_endLayer, :$(peak+"_Fit"):
	MoveVariable V_endCol, :$(peak+"_Fit"):
	MoveVariable V_endChunk, :$(peak+"_Fit"):
	MoveVariable V_chisq, :$(peak+"_Fit"):
	
	wave xwave, W_sigma, W_coef, p1width, p1pos, p1shape, p1area, p2width, p2pos, p2shape, p2area, p1werr, p1perr, p1aerr, param, input, info, Temp_1, Temp_2, p1area_1, p1area_2
	MoveWave xwave, :$(peak+"_Fit"):
	MoveWave W_sigma, :$(peak+"_Fit"):
	MoveWave W_coef, :$(peak+"_Fit"):
	MoveWave p1width, :$(peak+"_Fit"):
	MoveWave p1pos, :$(peak+"_Fit"):
	MoveWave p1shape, :$(peak+"_Fit"):
	MoveWave p1area, :$(peak+"_Fit"):
	MoveWave p2width, :$(peak+"_Fit"):
	MoveWave p2pos, :$(peak+"_Fit"):
	MoveWave p2shape, :$(peak+"_Fit"):
	MoveWave p2area, :$(peak+"_Fit"):
	MoveWave p1werr, :$(peak+"_Fit"):
	MoveWave p1perr, :$(peak+"_Fit"):
	MoveWave p1aerr, :$(peak+"_Fit"):
	MoveWave param, :$(peak+"_Fit"):
	MoveWave input, :$(peak+"_Fit"):
	MoveWave info, :$(peak+"_Fit"):
	MoveWave Temp_1, :$(peak+"_Fit"):
	MoveWave Temp_2, :$(peak+"_Fit"):
	MoveWave p1area_1, :$(peak+"_Fit"):
	MoveWave p1area_2, :$(peak+"_Fit"):
	
	
	for(i=lo; i<=hi; i+=1)
		wave data = $("fit_d"+num2str(i))
		MoveWave data, :$(peak+"_Fit"):
	endfor
end
