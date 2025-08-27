#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

//Procedure to kill multiple graphs and tables that are getting annoying.
//Authors: OG Reed
//Date created: 14/12/2022

Function KillGraphs(startnum, endnum)
    Variable startnum, endnum
    Variable i
    String WindowName
    
    for(i=startnum; i<=(endnum); i+=1)
        WindowName="Graph"+num2str(i)
        KillWindow $WindowName
    endfor
    
End

Function KillTables(startnum, endnum)
    Variable startnum, endnum
    Variable i
    String WindowName
    
    for(i=startnum; i<=(endnum); i+=1)
        WindowName="Table"+num2str(i)
        KillWindow $WindowName
    endfor
    
End