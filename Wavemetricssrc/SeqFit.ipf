#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// Sequential plotting function. Need to do a single fit already to get to W_coef. 
// Centre and range give the range over which the funciton will be fitted.
// The calculated co-efficients can be used as the initial guess for the next wave, or the same initial guess can be used each time.
// Authors: NG Jones, OG Reed
// Date created: 2012


Function SeqFit(ws,wf,func,p1,p2)
	variable ws, wf, p1, p2
	string func
	variable datpnts, datnum, wc, bkgarea, peakarea1, peakarea2, peakarea3, peakarea4
	variable widthl1, widthg1, widthv1, widthl2, widthg2, widthv2, widthl3, widthg3, widthv3, widthl4, widthg4, widthv4
	string datname, fitname
	wave input, ang
	variable V_chisq
	
	//newnotebook /n=fit /k=1 /f=0
	datpnts=(wf-ws)+1
	make /o /n=1000 bgfit
	duplicate ang xwave
	//Make/D/N=9/o W_coef = {0,0,0,0,0,1,1.96,0.03,0.04}
	wave W_coef
	duplicate /o W_coef input
	duplicate /o W_coef param
	// Info gives the bounds over which the function will be fitted
	make /o /n=2 info={p1+(0.5*(p2-p1)),0.5*(p2-p1)}
	
	
	// Runs the fitting over all waves/pattens between ws and wf
	for (wc=ws; wc<=wf; wc +=1)
		datname="d"+num2istr(wc)
		//seqfitvis(ws,wc,datname,info)
		datnum=wc-ws
		
		// Looks at the function selected and performs the appropriate fit and calculations
		strswitch(func)	
		
		case "onegauss":
			
			if(wc==ws)
				make /o /n=(datpnts) p1pos,p1width,p1area,chisq
				//notebook fit, text="File,B1,B2,B3,B4,B5,Amplitude, Position, Width, Area,EB1,EB2,EB3,EB4 ,EB5 ,Ea, Ep, Ew\r";
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
			
			//notebook fit, text=""+num2istr(wc)+","
			//notebook fit, text =""+num2str(param[0])+",";
			//notebook fit, text =""+num2str(param[1])+",";
			//notebook fit, text =""+num2str(param[2])+",";
			//notebook fit, text =""+num2str(param[3])+",";
			//notebook fit, text =""+num2str(param[4])+",";
			//notebook fit, text=""+num2str(param[5])+",";
			//notebook fit, text=""+num2str(param[6])+",";
			//notebook fit, text=""+num2str(param[7])+",";
			//notebook fit, text=""+num2str(peakarea1)+"\r";
			
			// Inlcude line below if you want to reset the guess for each pattern
			param = input
			
		break
		
								
		case "onevoigt":
			if(wc==ws)
				make /o /n=(datpnts) p1pos,p1width,p1area,p1shape
				param=input
			endif
			
			voigtaroundcentre($datname, param, ang, info[0],info[1])
			fitname="fit_"+datname

			peakarea1 = param[5]
			widthl1 = param[7]*param[8]
			widthg1 = param[7]
			widthv1 = widthl1/2 + sqrt((widthl1^2)/4 + widthg1^2)
		
			p1pos[datnum]=param[6]; p1width[datnum]=widthv1; p1area[datnum]=peakarea1; p1shape[datnum]=param[8]
			
			// Inlcude line below if you want to reset the guess for each pattern
			param = input
			
		break
		
		
		
		case "twoguass":
			if(wc==ws)
				make /o /n=(datpnts) p1pos,p1width,p1area,p2pos,p2width,p2area
				notebook fit, text="File, P1Intensity, P1Position, P1Width, P1Area, P2Intensity, P2Position, P2Width, P2Area\r";
				param=input
			endif
		
			twogaussaroundcentre($datname, param, ang, info[0],info[1])
			fitname="fit_"+datname
			
			peakarea1=param[5]*param[7]/0.3989
			peakarea2=param[8]*param[10]/0.3989
			
			p1pos[datnum]=param[6]; p1width[datnum]=param[7]; p1area[datnum]=peakarea1
			p2pos[datnum]=param[9]; p2width[datnum]=param[10]; p2area[datnum]=peakarea2

 
			notebook fit, text=""+num2str(wc)+","
			notebook fit, text =""+num2str(param[5])+",";
			notebook fit, text=""+num2str(param[6])+",";
			notebook fit, text=""+num2str(param[7])+",";
			notebook fit, text=""+num2str(peakarea1)+",";
			notebook fit, text=""+num2str(param[8])+",";
			notebook fit, text=""+num2str(param[9])+",";
			notebook fit, text=""+num2str(param[10])+",";
			notebook fit, text=""+num2str(peakarea2)+"\r";
			
			// Inlcude line below if you want to reset the guess for each pattern
			param = input
			
		break
		
		case "twovoigt":
			if(wc==ws)
				make /o /n=(datpnts) p1pos,p1width,p1area,p1shape,p2pos,p2width,p2area,p2shape
				param=input
			endif
			
			twovoigtaroundcentre($datname, param, ang, info[0],info[1])
			fitname="fit_"+datname
			
			peakarea1 = param[5]
			widthl1 = param[7]*param[8]
			widthg1 = param[7]
			widthv1 = widthl1/2 + sqrt((widthl1^2)/4 + widthg1^2)
			peakarea2 = param[9]
			widthl2 = param[11]*param[12]
			widthg2 = param[11]
			widthv2 = widthl2/2 + sqrt((widthl2^2)/4 + widthg2^2)
		
			p1pos[datnum]=param[6]; p1width[datnum]=widthv1; p1area[datnum]=peakarea1; p1shape[datnum]=param[8]
			p2pos[datnum]=param[10]; p2width[datnum]=widthv2; p2area[datnum]=peakarea2; p2shape[datnum]=param[12]
			
			// Inlcude line below if you want to reset the guess for each pattern
			param = input
			
		break
		
		case "threeguass":
			if(wc==ws)
				make /o /n=(datpnts) p1pos,p1width,p1area,p2pos,p2width,p2area,p3pos,p3width,p3area
				notebook fit, text="File, P1Intensity, P1Position, P1Width, P1Area, P2Intensity, P2Position, P2Width, P2Area, P3Intensity, P3Position, P3Width, P3Area\r";
				param=input
			endif
		
			threegaussaroundcentre($datname, param, ang, info[0],info[1])
			fitname="fit_"+datname
			
			peakarea1=param[5]*param[7]/0.3989
			peakarea2=param[8]*param[10]/0.3989
			peakarea3=param[11]*param[13]/0.3989
			
			p1pos[datnum]=param[6]; p1width[datnum]=param[7]; p1area[datnum]=peakarea1
			p2pos[datnum]=param[9]; p2width[datnum]=param[10]; p2area[datnum]=peakarea2
			p3pos[datnum]=param[12]; p3width[datnum]=param[13]; p3area[datnum]=peakarea3

 
			notebook fit, text=""+num2str(wc)+","
			notebook fit, text =""+num2str(param[5])+",";
			notebook fit, text=""+num2str(param[6])+",";
			notebook fit, text=""+num2str(param[7])+",";
			notebook fit, text=""+num2str(peakarea1)+",";
			notebook fit, text=""+num2str(param[8])+",";
			notebook fit, text=""+num2str(param[9])+",";
			notebook fit, text=""+num2str(param[10])+",";
			notebook fit, text=""+num2str(peakarea2)+",";
			notebook fit, text=""+num2str(param[11])+",";
			notebook fit, text=""+num2str(param[12])+",";
			notebook fit, text=""+num2str(param[13])+",";
			notebook fit, text=""+num2str(peakarea3)+"\r";
			
			
			// Inlcude line below if you want to reset the guess for each pattern
			param = input
			
		break
		
		
		case "fourguass":
			if(wc==ws)
				make /o /n=(datpnts) p1pos,p1width,p1area,p2pos,p2width,p2area,p3pos,p3width,p3area,p4pos,p4width,p4area
				notebook fit, text="File, P1Intensity, P1Position, P1Width, P1Area, P2Intensity, P2Position, P2Width, P2Area, P3Intensity, P3Position, P3Width, P3Area, P4Intensity, P4Position, P4Width, P4Area\r";
				param=input
			endif
		
			fourgaussaroundcentre($datname, param, ang, info[0],info[1])
			fitname="fit_"+datname
			
			peakarea1=param[5]*param[7]/0.3989
			peakarea2=param[8]*param[10]/0.3989
			peakarea3=param[11]*param[13]/0.3989
			peakarea4=param[14]*param[16]/0.3989
			
			p1pos[datnum]=param[6]; p1width[datnum]=param[7]; p1area[datnum]=peakarea1
			p2pos[datnum]=param[9]; p2width[datnum]=param[10]; p2area[datnum]=peakarea2
			p3pos[datnum]=param[12]; p3width[datnum]=param[13]; p3area[datnum]=peakarea3
			p4pos[datnum]=param[15]; p4width[datnum]=param[16]; p4area[datnum]=peakarea4

 
			notebook fit, text=""+num2str(wc)+","
			notebook fit, text =""+num2str(param[5])+",";
			notebook fit, text=""+num2str(param[6])+",";
			notebook fit, text=""+num2str(param[7])+",";
			notebook fit, text=""+num2str(peakarea1)+",";
			notebook fit, text=""+num2str(param[8])+",";
			notebook fit, text=""+num2str(param[9])+",";
			notebook fit, text=""+num2str(param[10])+",";
			notebook fit, text=""+num2str(peakarea2)+",";
			notebook fit, text=""+num2str(param[11])+",";
			notebook fit, text=""+num2str(param[12])+",";
			notebook fit, text=""+num2str(param[13])+",";
			notebook fit, text=""+num2str(peakarea3)+",";
			notebook fit, text=""+num2str(param[14])+",";
			notebook fit, text=""+num2str(param[15])+",";
			notebook fit, text=""+num2str(param[16])+",";
			notebook fit, text=""+num2str(peakarea4)+"\r";
			
			
			// Inlcude line below if you want to reset the guess for each pattern
			param = input
			
		break
 
 
 		case "fourvoigt":
 		
			if(wc==ws)
				make /o /n=(datpnts) p1pos,p1width,p1area,p1shape,p2pos,p2width,p2area,p2shape,p3pos,p3width,p3area,p3shape,p4pos,p4width,p4area,p4shape
				param=input
			endif
			
			fourvoigtaroundcentre($datname, param, ang, info[0],info[1])
			fitname="fit_"+datname
			
			peakarea1 = param[5]; widthl1 = param[7]*param[8]; widthg1 = param[7]; widthv1 = widthl1/2 + sqrt((widthl1^2)/4 + widthg1^2)
			peakarea2 = param[9]; widthl2 = param[11]*param[12]; widthg2 = param[11]; widthv2 = widthl2/2 + sqrt((widthl2^2)/4 + widthg2^2)
			peakarea3 = param[13]; widthl3 = param[15]*param[16]; widthg3 = param[15]; widthv3 = widthl3/2 + sqrt((widthl3^2)/4 + widthg3^2)
			peakarea4 = param[17]; widthl4 = param[19]*param[20]; widthg4 = param[19]; widthv4 = widthl4/2 + sqrt((widthl4^2)/4 + widthg4^2)
		
			p1pos[datnum]=param[6]; p1width[datnum]=widthv1; p1area[datnum]=peakarea1; p1shape[datnum]=param[8]
			p2pos[datnum]=param[10]; p2width[datnum]=widthv2; p2area[datnum]=peakarea2; p2shape[datnum]=param[12]
			p3pos[datnum]=param[14]; p3width[datnum]=widthv3; p3area[datnum]=peakarea3; p3shape[datnum]=param[16]
			p4pos[datnum]=param[18]; p4width[datnum]=widthv4; p4area[datnum]=peakarea4; p4shape[datnum]=param[20]
			
			// Inlcude line below if you want to reset the guess for each pattern
			//param = input
			
		break
		
		
 		endswitch
 		
 	endfor
 	
end
 
 
 
function Findpeakpos(x1,x2,func)
	variable x1,x2
	string func
	wave W_coef

	strswitch(func)	
		
	case "onevoigt":
	
		make /o /n=(7) positions
		wave positions
		FuncFit/H="000110000" onevoigt W_coef :HR:AR:d423[x1,x2] /X=:HR:AR:ang /D
		positions[0] = W_coef[6]
		FuncFit/H="000110000" onevoigt W_coef :HR_HTRR:A:d532[x1,x2] /X=:HR_HTRR:A:ang /D
		positions[1] = W_coef[6]
		FuncFit/H="000110000" onevoigt W_coef :HR_HTRR:B:d676[x1,x2] /X=:HR_HTRR:B:ang /D
		positions[2] = W_coef[6]
		FuncFit/H="000110000" onevoigt W_coef :HR_HTRR:F:d762[x1,x2] /X=:HR_HTRR:F:ang /D
		positions[3] = W_coef[6]
		FuncFit/H="000110000" onevoigt W_coef :HR_HTRR:H:d541[x1,x2] /X=:HR_HTRR:H:ang /D
		positions[4] = W_coef[6]
		FuncFit/H="000110000" onevoigt W_coef :HR_HTRR:I:d504[x1,x2] /X=:HR_HTRR:I:ang /D
		positions[5] = W_coef[6]
		FuncFit/H="000110000" onevoigt W_coef :HR_HTRR:J:d685[x1,x2] /X=:HR_HTRR:J:ang /D
		positions[6] = W_coef[6]
		
	break
 	
 	case "twovoigt":
 	
 		make /o /n=(14) positions
		wave positions
		FuncFit/H="0001100000000" twovoigt W_coef :HR:AR:d423[x1,x2] /X=:HR:AR:ang /D
		positions[0] = W_coef[6]
		positions[1] = W_coef[10]
		FuncFit/H="0001100000000" twovoigt W_coef :HR_HTRR:A:d532[x1,x2] /X=:HR_HTRR:A:ang /D
		positions[2] = W_coef[6]
		positions[3] = W_coef[10]
		FuncFit/H="0001100000000" twovoigt W_coef :HR_HTRR:B:d676[x1,x2] /X=:HR_HTRR:B:ang /D
		positions[4] = W_coef[6]
		positions[5] = W_coef[10]
		FuncFit/H="0001100000000" twovoigt W_coef :HR_HTRR:F:d762[x1,x2] /X=:HR_HTRR:F:ang /D
		positions[6] = W_coef[6]
		positions[7] = W_coef[10]
		FuncFit/H="0001100000000" twovoigt W_coef :HR_HTRR:H:d541[x1,x2] /X=:HR_HTRR:H:ang /D
		positions[8] = W_coef[6]
		positions[9] = W_coef[10]
		FuncFit/H="0001100000000" twovoigt W_coef :HR_HTRR:I:d504[x1,x2] /X=:HR_HTRR:I:ang /D
		positions[10] = W_coef[6]
		positions[11] = W_coef[10]
		FuncFit/H="0001100000000" twovoigt W_coef :HR_HTRR:J:d685[x1,x2] /X=:HR_HTRR:J:ang /D
		positions[12] = W_coef[6]
		positions[13] = W_coef[10]
 	
 	break
 	
 	case "fourvoigt":
 	
 		make /o /n=(28) positions
		wave positions
		FuncFit fourvoigt W_coef :HR:AR:d423[x1,x2] /X=:HR:AR:ang /D
		positions[0] = W_coef[6]
		positions[7] = W_coef[10]
		positions[14] = W_coef[14]
		positions[21] = W_coef[18]
		FuncFit fourvoigt W_coef :HR_HTRR:A:d532[x1,x2] /X=:HR_HTRR:A:ang /D
		positions[1] = W_coef[6]
		positions[8] = W_coef[10]
		positions[15] = W_coef[14]
		positions[22] = W_coef[18]
		FuncFit fourvoigt W_coef :HR_HTRR:B:d676[x1,x2] /X=:HR_HTRR:B:ang /D
		positions[2] = W_coef[6]
		positions[9] = W_coef[10]
		positions[16] = W_coef[14]
		positions[23] = W_coef[18]
		FuncFit fourvoigt W_coef :HR_HTRR:F:d762[x1,x2] /X=:HR_HTRR:F:ang /D
		positions[3] = W_coef[6]
		positions[10] = W_coef[10]
		positions[17] = W_coef[14]
		positions[24] = W_coef[18]
		FuncFit fourvoigt W_coef :HR_HTRR:H:d541[x1,x2] /X=:HR_HTRR:H:ang /D
		positions[4] = W_coef[6]
		positions[11] = W_coef[10]
		positions[18] = W_coef[14]
		positions[25] = W_coef[18]
		FuncFit fourvoigt W_coef :HR_HTRR:I:d504[x1,x2] /X=:HR_HTRR:I:ang /D
		positions[5] = W_coef[6]
		positions[12] = W_coef[10]
		positions[19] = W_coef[14]
		positions[26] = W_coef[18]
		FuncFit fourvoigt W_coef :HR_HTRR:J:d685[x1,x2] /X=:HR_HTRR:J:ang /D
		positions[6] = W_coef[6]
		positions[13] = W_coef[10]
		positions[20] = W_coef[14]
		positions[27] = W_coef[18]

 	
 	break
 	
 	endswitch
 	
 	print positions
 
end


//Not sure what these two do!!???

  //_____________________________
//Background fitter
//_____________________________

function Findbackground(datwave,ang, param)
	wave datwave, ang, param
	variable mrange, mcent, mlow, mhigh, fitrange
	
//_______________
// Set ranges here:
	mrange=20
	fitrange=5
//_______________
	
	findlevel /p /q ang param[6]
	mcent=round(v_levelx)
	mlow =mcent-mrange
	mhigh=mcent+mrange
	
	duplicate /o ang mask
	mask=1
	mask[mlow,mhigh]=0
	
	curvefit/q line datwave[(mlow-fitrange),(mhigh+fitrange)] /X=ang /m=mask
	
end	

