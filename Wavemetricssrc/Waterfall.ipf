#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// Makes a waterfall plot
// Authors: NG Jones, CEP Talbot

// Calls function to make a waterfall plot, lo is first wave and hi is last. 
function makewaterfall(lo,hi,step)
	variable lo, hi,step
	variable wnum,pts,i,dnum
	string dname
	wave ang
	//wave dwave			
	pts = numpnts(ang)
	wnum = floor((hi-lo)/step)

	make /O/N=(pts,wnum+1) dwater
	
	for ( i=lo; i<=hi; i +=step)
		dname= "d"+num2istr(i)
		duplicate /o $dname dwave
		dwater[][(i-lo)/step]= dwave [p]
		killwaves dwave
	endfor
	
	NewWaterfall dwater vs {ang,*}
	ModifyWaterfall angle=90, axlen=.5
	ModifyGraph rgb=(34952,34952,34952), nticks(left)=0, nticks(right)=0, axRGB(left)=(65535,65535,65535),axrgb(right)=(65535,655353,65535)
	Duplicate/O dwater,mat1ColorIndex
	mat1ColorIndex=y
	//ModifyGraph zColor(dwater)={mat1ColorIndex,*,*,coldwarm}
	//ModifyGraph width=432, height=648
	ModifyGraph mode=7,usePlusRGB=1,hbFill=2,plusRGB=(65535,65535,65535)
	
	// Standard graph editing
	SetAxis right 0,wnum
	Label bottom "2θ / °"
	Label left "Intensity / a.u."
	ModifyGraph gFont="Times New Roman",gfSize=11
	ModifyGraph tick(bottom)=2
	
	killwaves mat1colorindex
	
end

