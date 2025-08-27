#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// Sequential plotting function in reverse. Need to do a single fit already to get to W_coef. 
// Centre and range give the range over which the funciton will be fitted.
// The calculated co-efficients can be used as the initial guess for the next wave, or the same initial guess can be used each time.
// Authors: NG Jones, OG Reed
// Date created: 2012

Function SeqFitR(ws,wf,func,centre,range)
	variable ws, wf, centre, range
	string func
	variable datpnts, datnum, wc, peakarea1, bkgarea
	string datname, fitname
	wave input, ang, W_coef
	variable V_chisq
	
	newnotebook /n=fit /k=1 /f=0
	datpnts=(ws-wf)+1
	make /o /n=1000 bgfit
	duplicate ang xwave
	duplicate /o W_coef input
	duplicate /o W_coef param
	// Info gives the bounds over which the function will be fitted
	make /o /n=2 info={centre,range}

	for (wc=ws; wc>wf; wc -=1)
		datname="d"+num2istr(wc)
		datnum=ws-wc
		seqfitvisr(ws,wc,xwave,datname,info)
		
		strswitch(func)
		
		case "onegauss":
			
			if(wc==ws)
				make /o /n=(datpnts) p1pos,p1width,p1area,chisq
				notebook fit, text="File,B1,B2,B3,B4,B5,Amplitude, Position, Width, Area,EB1,EB2,EB3,EB4 ,EB5 ,Ea, Ep, Ew\r";
				param = input
			endif
			
			//peak fitting bit
			gaussaroundcentre($datname, param, xwave, info[0],info[1])
			fitname="fit_"+datname
		
			// Old method for finding area under peak. More accurate to use eqn
			//make /n=1000 pwork
			//setscale x, (info[0]-info[1]),(info[0]+info[1]),pwork
			//pwork=param[5]*exp(-((x-param[6])^2/(2*param[7]^2)))
			//peakarea=area(pwork)
			//killwaves pwork
			
			peakarea1=param[5]*param[7]/0.3989
						
			p1pos[datnum]=param[6]; p1width[datnum]=param[7]; p1area[datnum]=peakarea1;
			
			notebook fit, text=""+num2istr(wc)+","
			notebook fit, text =""+num2str(param[0])+",";
			notebook fit, text =""+num2str(param[1])+",";
			notebook fit, text =""+num2str(param[2])+",";
			notebook fit, text =""+num2str(param[3])+",";
			notebook fit, text =""+num2str(param[4])+",";
			notebook fit, text=""+num2str(param[5])+",";
			notebook fit, text=""+num2str(param[6])+",";
			notebook fit, text=""+num2str(param[7])+",";
			notebook fit, text=""+num2str(peakarea1)+"\r";
			
			// Inlcude line below if you want to reset the guess for each pattern
			param = input

		break
			
		// Add more functions in here if needed (can just copy and paste from SeqFit)
			
		endswitch
								
	endfor
	
end