#pragma TextEncoding = "MacRoman"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// List of peak fitting (and other important) functions with multiple peaks and backgrounds
// Authors: NG Jones, OG Reed
// Date created: 2012


//_______________________________________
// Gaussian with polynomial background
// NG Jones - January 2013
//_______________________________________
function onegaus(param,x): FitFunc
	wave param
	variable x
	variable bkg,p1
	
	// bkg - param 0-4
	// p1 - param 5-7
	
	bkg = param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
	p1 = param[5]*exp(-((x-param[6])^2/(2*param[7]^2)))
	return bkg+p1
	
end

//__________________________________________
// Voigt Function
// O. Reed - Nov 2023
//__________________________________________
function onevoigt(param,x): FitFunc
	wave param
	variable x
	variable bkg, voigtX, voigtY
	
	//param[0-4] = background
	//param[5] = area
	//param[6] = x0
	//param[7] = gw (FWHM)
	//param[8] = shape (Lw/Gw)
	
	bkg = param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
	voigtX = 2*sqrt(ln(2))*(x-param[6])/param[7]
	voigtY = sqrt(ln(2))*param[8]
	
	return bkg + (param[5]/param[7])*2*sqrt(ln(2)/pi)*VoigtFunc(voigtX, voigtY)
	
end

//______________________________
// TwoGaus fitting, 4 order poly bkg
// NGJ - Sept 2013
//______________________________
function twogaus(param,x): FitFunc
	wave param
	variable x
	variable bkg,p1,p2
	
	// bkg - param 0-4
	// p1 - param 5-7
	// p2 - param 8 -10
	
	bkg = param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
	p1 = param[5]*exp(-((x-param[6])^2/(2*param[7]^2)))
	p2 = param[8]*exp(-((x-param[9])^2/(2*param[10]^2)))
	
	return bkg+p1+p2
end


//________________________________
// TwoVoigt fitting, 4 order poly pkg
// O. Reed - April 2024
//________________________________
function twovoigt(param,x): FitFunc
	wave param
	variable x
	variable bkg, voigtX1, voigtY1, voigtX2, voigtY2
	
	//param[0-4] = background
	//param[5] = area 1
	//param[6] = x0 1
	//param[7] = gw (FWHM) 1
	//param[8] = shape (Lw/Gw) 1
	//param[9] = area 2
	//param[10] = x0 2
	//param[11] = gw (FWHM) 2
	//param[12] = shape (Lw/Gw) 2
	
	bkg = param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
	voigtX1 = 2*sqrt(ln(2))*(x-param[6])/param[7]
	voigtY1 = sqrt(ln(2))*param[8]
	voigtX2 = 2*sqrt(ln(2))*(x-param[10])/param[11]
	voigtY2 = sqrt(ln(2))*param[12]
	
	return bkg + (param[5]/param[7])*2*sqrt(ln(2)/pi)*VoigtFunc(voigtX1, voigtY1) + (param[9]/param[11])*2*sqrt(ln(2)/pi)*VoigtFunc(voigtX2, voigtY2)

end


//_______________________________
// ThreeGaus fitting, 4 order poly bkg
// NGJ - Sept 2013
//_______________________________
function threegaus(param,x): FitFunc
	wave param
	variable x
	variable bkg,p1,p2,p3
	
	// bkg - param 0-4
	// p1 - param 5-7
	// p2 - param 8 -10
	// p3 - param 11 - 13
	
	bkg = param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
	p1 = param[5]*exp(-((x-param[6])^2/(2*param[7]^2)))
	p2 = param[8]*exp(-((x-param[9])^2/(2*param[10]^2)))
	p3 = param[11]*exp(-((x-param[12])^2/(2*param[13]^2)))
	
	return bkg+p1+p2+p3
end


//________________________________
// ThreeVoigt fitting, 4 order poly pkg
// O. Reed - April 2024
//________________________________
function threevoigt(param,x): FitFunc
	wave param
	variable x
	variable bkg,p1,p2,p3
	
	// bkg - param 0-4
	// p1 - param 5-8
	// p2 - param 9-12
	// p3 - param 13-16
	// voigt order - int, width,pos,shape
	
	bkg = param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
	p1 = param[5]*VoigtFunc(param[6]*(x-param[7]),param[8])
	p2 = param[9]*VoigtFunc(param[10]*(x-param[11]),param[12])
	p3= param[13]*VoigtFunc(param[14]*(x-param[15]),param[16])
	
	return bkg+p1+p2+p3
end


//_______________________________
// FourGaus fitting, 4 order poly bkg
// NGJ - Sept 2013
//_______________________________
function fourgaus(param,x): FitFunc
	wave param
	variable x
	variable bkg,p1,p2,p3, p4
	
	// bkg - param 0-4
	// p1 - param 5-7
	// p2 - param 8 -10
	// p3 - param 11 - 13
	// p4 - param 14 - 16

	bkg = param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
	p1 = param[5]*exp(-((x-param[6])^2/(2*param[7]^2)))
	p2 = param[8]*exp(-((x-param[9])^2/(2*param[10]^2)))
	p3 = param[11]*exp(-((x-param[12])^2/(2*param[13]^2)))
	p4 = param[14]*exp(-((x-param[15])^2/(2*param[16]^2)))
	
	return bkg+p1+p2+p3+p4
end

//________________________________
// FourVoigt fitting, 4 order poly pkg
// O. Reed - April 2024
//________________________________
function fourvoigt(param,x): FitFunc
	wave param
	variable x
	variable bkg, voigtX1, voigtY1, voigtX2, voigtY2, voigtX3, voigtY3, voigtX4, voigtY4, p1, p2, p3, p4
	
	//param[0-4] = background
	//param[5] = area 1
	//param[6] = x0 1
	//param[7] = gw (FWHM) 1
	//param[8] = shape (Lw/Gw) 1
	//param[9] = area 2
	//param[10] = x0 2
	//param[11] = gw (FWHM) 2
	//param[12] = shape (Lw/Gw) 2
	//param[13] = area 3
	//param[14] = x0 3
	//param[15] = gw (FWHM) 3
	//param[16] = shape (Lw/Gw) 3
	//param[17] = area 4
	//param[18] = x0 4
	//param[19] = gw (FWHM) 4
	//param[20] = shape (Lw/Gw) 4
	
	bkg = param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
	voigtX1 = 2*sqrt(ln(2))*(x-param[6])/param[7]
	voigtY1 = sqrt(ln(2))*param[8]
	voigtX2 = 2*sqrt(ln(2))*(x-param[10])/param[11]
	voigtY2 = sqrt(ln(2))*param[12]
	voigtX3 = 2*sqrt(ln(2))*(x-param[14])/param[15]
	voigtY3 = sqrt(ln(2))*param[16]
	voigtX4 = 2*sqrt(ln(2))*(x-param[18])/param[19]
	voigtY4 = sqrt(ln(2))*param[20]
	
	p1 = (param[5]/param[7])*2*sqrt(ln(2)/pi)*VoigtFunc(voigtX1, voigtY1)
	p2 = (param[9]/param[11])*2*sqrt(ln(2)/pi)*VoigtFunc(voigtX2, voigtY2)
	p3 = (param[13]/param[15])*2*sqrt(ln(2)/pi)*VoigtFunc(voigtX3, voigtY3)
	p4 = (param[17]/param[19])*2*sqrt(ln(2)/pi)*VoigtFunc(voigtX4, voigtY4)
	
	return bkg + p1 + p2 + p3 + p4 

end

//_______________________________________
// Gaussian with exponential background
// Created - NGJ January 2013
//_______________________________________
function singgausexp(param,x) : FitFunc
	wave param
	variable x
	variable p1, bkg
	
	// bkg - param 0-3
	// p1 - param 4-6
	
	bkg = param[0]+param[1]*exp(-(x-param[3])/param[2])
	p1 = param[4]*exp(-((x-param[5])/param[6])^2)
	return bkg+p1
	
end

	
//_______________________________________
// Inverse quadratic (i.e. y = x^-0.5). e.g. for grain size to yield strength correlation
// OG Reed - May 2024
//_______________________________________
function invquad(param,x): FitFunc
	wave param
	variable x
	variable p1
	
	p1 = param[0]+param[1]*(x^-0.5)
	return p1
	
end
