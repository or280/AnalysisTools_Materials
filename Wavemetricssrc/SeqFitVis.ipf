#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// 
// Authors: NG Jones, OG Reed
// Date created: 2012

//_________________________________________________
// Sequential plotting for data visualisation
//_________________________________________________
 function seqfitvis (ws, wc, datname,info)
 	variable ws,wc
 	string datname
 	wave info
 	variable pcent, tol, rl, rh
  	string olddata, oldfit
  	wave param,bgfit, ang
 	
 	pcent=info[0]
 	tol=info[1]
 	olddata="d"+num2istr(wc-1)
 	oldfit="fit_d"+num2istr(wc-1)
 	
 		if(wc==ws)
 			display $datname vs ang
	 		setaxis bottom (pcent-1.5*tol),(pcent+1.5*tol)
 			findlevel /q ang (pcent-tol)
			rl=v_levelx
			findlevel /q ang (pcent+tol)
 			rh=v_levelx
			wavestats /q /r=(rl,rh) $datname
			setaxis left v_min, 1.1*v_max
// 		setaxis bottom 10.4,10.9;setaxis left 0,60
 			ModifyGraph width=600,height={Aspect,0.75}
 			setscale x,(pcent-1.2*tol),(pcent+1.2*tol), bgfit
 			bgfit= param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
 			appendtograph bgfit
 			ModifyGraph mode(bgfit)=0,rgb(bgfit)=(3,52428,1)
 			
 		else
 			appendtograph $datname vs ang
 			bgfit= param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
 			findlevel /q ang (pcent-tol)
			rl=v_levelx
			findlevel /q ang (pcent+tol)
			rh=v_levelx
			wavestats /q /r=(rl,rh) $datname
 			setaxis left v_min, (1.1*V_max)
 			removefromgraph $olddata
 			removefromgraph $oldfit
 		endif
 		
 		modifygraph mode=3,msize=3, rgb=(0,0,0)
 		ModifyGraph mode(bgfit)=0,rgb(bgfit)=(3,52428,1)
 end 	
 
 
 
 //_________________________________________________
// Sequential plotting for data visualisation
//_________________________________________________
 function seqfitvisr (ws, wc, xwave, datname,info)
 	variable ws,wc
 	wave xwave
 	string datname
 	wave info
 	variable pcent, tol, rl, rh
  	string olddata, oldfit
  	wave bgfit, param
 	
 	pcent=info[0]
 	tol=info[1]
 	olddata="d"+num2istr(wc+1)
 	oldfit="fit_d"+num2istr(wc+1)
 	 
 		if(wc==ws)
 			display $datname vs xwave
 			setaxis bottom (pcent-1.5*tol),(pcent+1.5*tol)
 			findlevel /q xwave (pcent-tol)
			rl=v_levelx
			findlevel /q xwave (pcent+tol)
 			rh=v_levelx
			wavestats /q /r=(rl,rh) $datname
			setaxis left v_min, 1.1*v_max
 			ModifyGraph width=600,height={Aspect,0.75}
 			Modifygraph mode($datname)=4
 			setscale x, (pcent-tol), (pcent+tol),bgfit
 			bgfit=param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
 			appendtograph bgfit

 				
 			
 		else
 			bgfit=param[0]+param[1]*x+param[2]*x^2+param[3]*x^3+param[4]*x^4
 			appendtograph $datname vs xwave
 			findlevel /q xwave (pcent-tol)
			rl=v_levelx
			findlevel /q xwave (pcent+tol)
			rh=v_levelx
			wavestats /q /r=(rl,rh) $datname
 			setaxis left v_min, (1.1*V_max)
 			removefromgraph $olddata
 			removefromgraph $oldfit
 		endif
 		
 		modifygraph mode=3,msize=3, rgb=(0,0,0)
 end 	
 
 
 