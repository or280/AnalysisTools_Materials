#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later



function removezero(frame)
	 wave frame
	 variable row = dimsize(frame,0)
	 variable col = dimsize(frame,1)
	 variable i,j
	 
	 print row
	 print col
	 
	 for(i=0;i<row;i+=1)
	 	for(j=0;j<col;j+=1)
	 		if(frame[i][j] == 0)
	 			frame[i][j] = NaN
	 		endif
	 	endfor
	 endfor
	 
end


function/WAVE rowavgwave(frame)
	wave frame
	variable row = dimsize(frame,0)
	variable col = dimsize(frame,1)
	variable i, j, valuecounter, rowsum
	
	wave rowavg
	
	for(i=0;i<row;i+=1)
		valuecounter = 0
		rowsum = 0
		
		for(j=0;j<col;j+=1)
			if(frame[i][j] != 0)
	 			valuecounter += 1
	 			rowsum += frame[i][j]
	 		endif
	 	endfor
	 	
		rowavg[i] = rowsum/valuecounter
		
		return rowavg
		
	endfor

end


function allrowavgs(frames)
	variable frames
	variable i
	wave f1
	
	wave rowavgs
	Make/O/N=(dimsize(f1,0),frames) rowavgs
	
	for(i=1;i<frames+1;i+=1)
		wave data = $("f"+num2str(i))
		wave data2 = rowavgwave(data)
		rowavgs[][i] = data2
	endfor
	
end