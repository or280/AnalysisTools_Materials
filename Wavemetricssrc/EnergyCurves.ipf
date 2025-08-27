#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later


function plotG(dHexp, dSexp, Texp)
	wave dHexp, dSexp, Texp

	wave Temp, dH, dS, dG, GM, GA, W_coef, dHparam, dSparam
	variable i
	Make/O/N=(400) Temp
	Make/O/N=(400) dH
	Make/O/N=(400) dS
	Make/O/N=(400) dG
	Make/O/N=(400) GM
	Make/O/N=(400) GA
	
	Make/O/N=(2) W_coef
	Make/O/N=(2) dHparam
	Make/O/N=(2) dSparam
	CurveFit/M=0 line, dHexp/X=Texp/D
	dHparam = W_coef
	K0 = 0
	CurveFit/h="10"/M=0 line, dSexp/X=Texp/D
	dSparam = W_coef
	
	for (i=0;i<400; i+=1)
		Temp[i] = i
		
		dH[i] = dHparam[0] + dHparam[1]*i
		dS[i] = dSparam[0] + dSparam[1]*i
		dG[i] = dH[i] - i*dS[i]*0.001
		
		GM[i] = 0 - i*i*0.0005
		GA[i] = GM[i]+dG[i]
	endfor
	
	display GM vs Temp
	appendtograph GA vs Temp
	
	wave fit_dHexp, fit_dSexp, dSparam, dHparam, W_coef, W_sigma
	killwaves fit_dHexp, fit_dSexp, dSparam, dHparam, W_coef, W_sigma
	
end


function plotG2(Hm, Ha, Sm, Sa, Mp, Ap)
	variable Hm, Ha, Sm, Sa, Mp, Ap

	Mp = Mp + 273
	Ap = Ap + 273

	wave Temp, dH, dS, dG, GM, GA
	variable i
	Make/O/N=(600) Temp
	Make/O/N=(600) dH
	Make/O/N=(600) dS
	Make/O/N=(600) dG
	Make/O/N=(600) GM
	Make/O/N=(600) GA
	
	variable Sdiff, Hdiff, Tdiff, Trecdiff, Tlndiff, Trecsqdiff, cleft, cright, c, a
	Sdiff = Sa-Sm
	Hdiff = Ha-Hm
	Tdiff = Ap-Mp
	Trecdiff = 1/Ap - 1/Mp
	Tlndiff = ln(Ap) - ln(Mp)
	Trecsqdiff = 1/(Ap^2) - 1/(Mp^2)
	cleft = Sdiff-(Hdiff*Tlndiff)/Tdiff
	cright = (Trecdiff*Tlndiff)/Tdiff - 2*Trecsqdiff
	c = cleft/cright
	a = (Hdiff + c*Trecdiff)/Tdiff
	print a, c
	
	for (i=1;i<600; i+=1)
		Temp[i] = i
		
		dH[i] = Hm + a*(i-Mp) - c*((1/i)-(1/Mp))
		dS[i] = Sm + a*(ln(i)-ln(Mp)) - 2*c*((1/i^2)-(1/Mp^2))
		dG[i] = dH[i] - i*dS[i]
		
		GM[i] = 0.01*i - 0.05*i*ln(i)
		GA[i] = GM[i]+dG[i]
	endfor
	
	display GM vs Temp
	appendtograph GA vs Temp
	
end


// Function defining the einstien model of the heat capacity with temperature
function heatcap(param,x): FitFunc
	wave param
	variable x
	variable C
	
	// max value - param 0
	// sharpness - param 1
	
	C = param[0] * (param[1]/(x+273.15))^2 * (exp(param[1]/(x+273.15))/(exp(param[1]/(x+273.15))-1)^2)
	return C
	
end


function plotG3(Ca0, Ca1, Cm0, Cm1, T0)
	variable Ca0, Ca1, Cm0, Cm1, T0

	wave Temp, CM, CA, HA, HM, SM, SA, GM, GA
	variable i, dH0
	Make/O/N=(600) Temp
	Make/O/N=(600) CM
	Make/O/N=(600) CA
	Make/O/N=(600) CMoverT
	Make/O/N=(600) CAoverT
	Make/O/N=(600) HM
	Make/O/N=(600) HA
	Make/O/N=(600) SM
	Make/O/N=(600) SA
	Make/O/N=(600) GM
	Make/O/N=(600) GA
	
	for (i=1;i<600; i+=1)
		Temp[i] = i
		CA[i] = 6 * Ca0 * (Ca1/i)^2 * (exp(Ca1/i)/(exp(Ca1/i)-1)^2)
		CM[i] = 6 * Cm0 * (Cm1/i)^2 * (exp(Cm1/i)/(exp(Cm1/i)-1)^2)
		
		CAoverT[i] = CA[i]/i
		CMoverT[i] = CM[i]/i
	endfor
	
	integrate /T CA /D = HA
	integrate /T CM /D = HM
	integrate /T CAoverT /D = SA
	integrate /T CMoverT /D = SM
	
	for (i=1;i<600; i+=1)
		GA[i] = HA[i] - i*SA[i]
		GM[i] = HM[i] - i*SM[i]
	endfor
	
	dH0 = GM[T0] - GA[T0]
	GM = GM - dH0
	HM = HM - dH0
	
	print "dH @ T = 0K is: " + num2str(dH0)
	
	display CM vs Temp
	appendtograph CA vs Temp
	ModifyGraph rgb(CM)=(1,4,52428)
	
	display GM vs Temp
	appendtograph GA vs Temp
	ModifyGraph rgb(GM)=(1,4,52428)
	
end


function rougefitting(dH0, A, x0, w, dHm, dHa, dSm, dSa, Mp, Ap)
	variable dH0, A, x0, w, dHm, dHa, dSm, dSa, Mp, Ap

	Mp = Mp + 273
	Ap = Ap + 273
	
	wave Temp, dC, dCoverT, dH, dS
	variable i
	Make/O/N=(600) Temp
	Make/O/N=(600) dC
	Make/O/N=(600) dCoverT
	Make/O/N=(600) dH
	Make/O/N=(600) dS
	
	for (i=1;i<600; i+=1)
		Temp[i] = i
		dC[i] = A * exp(-(ln(i/x0)/w)^2)
		dCoverT[i] = dC[i]/i
	endfor
	
	integrate /T dC /D = dH
	dH = dH + dH0 
	integrate /T dCoverT /D = dS
	
	variable errorsq = 0
	errorsq += (dHm - dH[Mp])^2
	errorsq += (dHa - dH[Ap])^2
	errorsq += (dSm - dS[Mp])^2
	errorsq += (dSa - dS[Ap])^2
	
	//print errorsq
	return errorsq
	
end


function dCfit(dHm, dHa, dSm, dSa, Mp, Ap)
	variable dHm, dHa, dSm, dSa, Mp, Ap

	Mp = Mp + 273
	Ap = Ap + 273
	
	variable dH0,A,x0,w
	
	variable dH01 = 4, dH02 = 5
	variable A1 = 0.14, A2 = 0.16
	variable x01 = 65, x02 = 75
	variable w1 = 0.13, w2 = 0.15
	variable sens = 10
	
	variable i=0, j=0, k=0, l=0
	make/O/N=(sens,sens,sens,sens) errors
	for (dH0=dH01;dH0<dH02; dH0+=(dH02-dH01)/sens)
		for (A=A1;A<A2; A+=(A2-A1)/sens)
			for (x0=x01;x0<x02; x0+=(x02-x01)/sens)
				for (w=w1;w<w2; w+=(w2-w1)/sens)
					errors[i][j][k][l] = rougefitting(dH0, A, x0, w, dHm, dHa, dSm, dSa, Mp, Ap)
					l+=1
				endfor
				k+=1
				l=0
			endfor
			j+=1
			k=0
			l=0
		endfor
		i+=1
		j=0
		k=0
		l=0
	endfor
	
	variable dH0ans, Aans, x0ans, wans
	wavestats/Q errors
	dH0ans = dH01 + V_minRowLoc*(dH02-dH01)/sens
	Aans = A1 + V_minColLoc*(A2-A1)/sens
	x0ans = x01 + V_minLayerLoc*(x02-x01)/sens
	wans = w1 + V_minChunkLoc*(w2-w1)/sens
	
	print "dH0 = " + num2str(dH0ans)
	print "A = " + num2str(Aans)
	print "x0 = " + num2str(x0ans)
	print "w = " + num2str(wans)
	

end


function plotG4(dH0,A,x0,w)
	variable dH0, A, x0, w

	wave Temp, CM, CA, dC, CMoverT, CaoverT, HA, HM, SM, SA, GM, GA
	variable i
	Make/O/N=(600) Temp
	Make/O/N=(600) CM
	Make/O/N=(600) CA
	Make/O/N=(600) dC
	Make/O/N=(600) CMoverT
	Make/O/N=(600) CAoverT
	Make/O/N=(600) HM
	Make/O/N=(600) HA
	Make/O/N=(600) SM
	Make/O/N=(600) SA
	Make/O/N=(600) GM
	Make/O/N=(600) GA

	for (i=1;i<600; i+=1)
		Temp[i] = i
		
		CM[i] = 0.468 * (250/i)^2 * (exp(250/i)/(exp(250/i)-1)^2)
		dC[i] = A * exp(-(ln(i/x0)/w)^2)
		CA[i] = CM[i] + dC[i]
		
		CAoverT[i] = CA[i]/i
		CMoverT[i] = CM[i]/i
	endfor

	integrate /T CA /D = HA
	integrate /T CM /D = HM
	HM = HM - dH0
	integrate /T CAoverT /D = SA
	integrate /T CMoverT /D = SM
	
	for (i=1;i<600; i+=1)
		GA[i] = HA[i] - i*SA[i]
		GM[i] = HM[i] - i*SM[i]
	endfor
	
	display CM vs Temp
	appendtograph CA vs Temp
	ModifyGraph rgb(CM)=(1,4,52428)
	
	display GM vs Temp
	appendtograph GA vs Temp
	ModifyGraph rgb(GM)=(1,4,52428)
	
end