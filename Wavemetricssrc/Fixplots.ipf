#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// Script for standardising plots
// Author: OG Reed
// Date created: 12/10/2023


function fixlineplot()
	ModifyGraph axThick=1.5
	ModifyGraph margin(left)=34,margin(bottom)=34,margin(right)=8,margin(top)=8,width=212.598,height={Aspect,1}
	ModifyGraph tick=2,mirror=1,standoff=0
	ModifyGraph fSize=11,font="Times New Roman"
	ModifyGraph lsize=1.5
end

function fixmarkplot()
	ModifyGraph axThick=1.5
	ModifyGraph margin(left)=34,margin(bottom)=34,margin(right)=8,margin(top)=8,width=212.598,height={Aspect,1}
	ModifyGraph tick=2,mirror=1,standoff=0
	ModifyGraph fSize=11,font="Times New Roman"
	ModifyGraph mode=3,marker=19,msize=3,mrkThick=0.5, rgb=(1,4,52428)
end
