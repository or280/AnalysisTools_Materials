#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// Calculator for using PTMT to evaluate and perform habit plane analysis for the beta to adp transformation
// CEPT Jul '22, edited by OG Reed

// How to run: 	1) Call RunPTMT(a0, a, b, c) with a0 being beta lp and abc the adp lp's
// Everything will print to the console

// Refer to Lieberman, Weschler and Read, JAP 26, 473 (1955)
function PTMTTiNb(a0, a, b, c)
	variable a0, a, b, c
	
	FindF(a0, a, b, c)
	
	wave F, Fstar
	DiagonaliseF(F, Fstar)
	
	wave Fd
	variable xdydratio = SolveXdYd(Fd)

	FindNormalAxis(xdydratio)
	
	wave n_d
	FindDeformationMatrix()
	shearangle()
	twinplane()
	transformationmatrices()
	orientationrelationships(a, b, c)
	
end

//First, calculate the matrix Fd, which is the diagonal of the symmetric part of the tensor F. F relates the transformation
//tensor of the parent and martensite phases with the need to maintain an invariant plane strain. 

function FindF(a0, a, b, c)
	variable a0, a, b, c //LP's of beta and adp phases respectively
	
	variable eta_1 = (sqrt(2)*(b+c))/(4*a0)
	variable eta_2 = (sqrt(2)*(-b+c))/(4*a0)
	variable eta_3 = a/a0
	
	make/O/N=(3,3) T1 = {{eta_1,eta_2,0},{eta_2,eta_1,0},{0,0,eta_3}}
	make/O/N=(3,3) T2 = {{eta_1,0,eta_2},{0,eta_3,0},{eta_2,0,eta_1}}
	make/O/N=(3,3) Phi = {{0.9999,0.009585,-0.01034},{-0.01034,0.9970,-0.07608},{0.009585,0.07618,0.9970}}
	
	variable sizeofjwave = 50000
	Make/O/N=(sizeofjwave) jwave; SetScale/I x 0, 1, jwave; jwave = x
	
	variable i = 0
	make/O/N=(numpnts(jwave)) Solutions
	Make/O/N=(numpnts(jwave)) ModSolutions
	
	for(i=0; i<numpnts(jwave); i+=1)
	
		Solutions[i] = FindDeterminantF(T1, T2, Phi, jwave[i]) //Calculate a series of values to solve for x graphically
		ModSolutions[i] = Abs(Solutions[i])
	
	endfor
	
	WaveStats/Q ModSolutions //FindMinima to find x
	make/o/n=1 TwinFrac = 1-jwave[V_minloc]
	matrixOP/O F = (1-TwinFrac[0])*T1 + TwinFrac[0]*Phi x T2
	matrixOP/O Fstar = F^t
	
end

//Evaluates the determinant of F to allow graphical solutions

Function FindDeterminantF(T1, T2, Phi, j) //Calculates Det(FF*-I), where F = (1-j)T1+j*Phi x T2 and j is the volume fraction of variants
	wave T1, T2, Phi
	variable j

		matrixOP/O F = (1-j)*T1 + j*Phi x T2
		matrixOP/O Fstar = F^t
		matrixOP/O Det = Det((F x Fstar) - Identity(3))
		variable Solution = Det[0]
		
		return Solution
	
end

//Diagonalises FFstar allowing Fd to be found

function DiagonaliseF(F, Fstar)
	wave F, Fstar
	
	MatrixOP/O FstarF = Fstar x F
	MatrixEigenV/SYM/EVEC FstarF //Computes the eigenvalues allowing diagonalisation
	wave W_eigenValues, M_eigenVectors
	make/O/N=3 EigenValues = real(W_eigenValues)
	make/O/N=(3,3) Fd = {{EigenValues[0],0,0},{0,EigenValues[1],0},{0,0,EigenValues[2]}}
	MatrixOP/O Fd = sqrt(Fd)
	make/O/N=(3,3) EigenVectors = M_eigenVectors

end

//Uses Eigenvalues to calculate the xd/yd ratio which enables habit plane calculations

function SolveXdYd(Fd)
	wave Fd
	wave EigenValues, EigenVectors, sortvalues
	
	make/O/N=3 sortvalues={Fd[0][0],Fd[1][1],Fd[2][2]}
	print sortvalues
	sort/R sortvalues,sortvalues
	
	print sortvalues
	
	if(sortvalues[1] == Fd[0][0])
			matrixOP/O EigenValues = rotateRows(EigenValues,2)
			matrixOP/O Fd = rotateRows(Fd,2)
			matrixOP/O Fd = rotateCols(Fd,2)
			matrixOP/O EigenVectors = rotateCols(EigenVectors,2)
	elseif(sortvalues[1] == Fd[1][1])
			matrixOP/O EigenValues = rotateRows(EigenValues,1)
			matrixOP/O Fd = rotateRows(Fd,1)
			matrixOP/O Fd = rotateCols(Fd,1)
			matrixOP/O EigenVectors = rotateCols(EigenVectors,1)
			print ("B")
	endif
	

	variable Lambda1 = Fd[0][0]
	variable Lambda2 = Fd[1]	[1]
	variable Lambda3 = Fd[2][2]
	print ("λ1 = "+num2str(Lambda1)+", λ2 = "+num2str(Lambda2)+", λ3 = "+num2str(Lambda3)+".")
	
	Variable xdydratio = sqrt((1-lambda2^2)/(lambda1^2-1))
	print ("x/y deformation ratio  -  Should be close to 1 = "+num2str(xdydratio))
	return xdydratio

end

//Finds the Habit Plane normal

function FindNormalAxis(xdydratio)
	variable xdydratio
	wave EigenVectors

	make/O/n=3 axis = {1, -xdydratio, 0}
	
	
	make/O/n=3 n_d = 1/sqrt((1+xdydratio^2))*axis
	
	MatrixOP/O InterfaceNormal = EigenVectors x n_d
	print ("Interface normal (w.r.t. beta) = {"+num2str(InterfaceNormal[0])+", "+num2str(InterfaceNormal[1])+", "+num2str(InterfaceNormal[2])+"}")

	
end

function FindDeformationMatrix()
wave n_d, InterfaceNormal, F
wave EigenVectors

	make/o/n=3 kd = {0,0,1}	
	cross/Dest=sd kd, n_d
	
	//Calculate shear vector
	MatrixOP/o shear = EigenVectors x sd
	print ("Direction of shear, s = {"+num2str(shear[0])+", "+num2str(shear[1])+", "+num2str(shear[2])+"}")
	
	//Find the rotation component of the total deformation E
	//Find two vectors which lie within the interface plane
	make/o/n=3 id = {1,0,0}
	make/o/n=3 jd = {0,1,0}
	
	cross/Dest=q1 id, InterfaceNormal 
	cross/Dest=p1 jd, InterfaceNormal //Vectors q1 and p1 are such that they lie in the plane
	
	MatrixOP/O q2 = F x q1
	MatrixOP/O p2 = F x p1 //The action of F on q1 and p1 will enable the rotation to be evaluated
	
	//Use q1 and p1 to calculate psi such that Er = Psi x F x r = r, (q1-q2)x(p1-p2)/(q1-q2).(p1-p2) = tan(psi1/2)*uo where uo is the rotation direction and psi1 the rotation magnitude
	
	make/n=3/o QDiff = q1-q2
	make/n=3/o PDiff = p1-p2
	make/n=3/o PSum = p1+p2
	
	//Find the rotation utilising Euler's rotational theorem 
	cross/Dest=Numer QDiff,PDiff
	MatrixOP/O Denom = QDiff.PSum
	Make/n=3/O EulerAngle = Numer/Denom[0] //Calculates tan(psi_1/2)*u_o where psi_1 is the magnitude of rotation and u_o the normalised matrix
	Make/n=1/O RotationMag = sqrt(EulerAngle[0]^2+EulerAngle[1]^2+EulerAngle[2]^2)
	variable Psi1 = 2*atan(RotationMag[0]) //Calculates the magnitude of the rotation
	Make/n=3/O u = EulerAngle/RotationMag[0] //Calculates the normalised rotation axis, u0
	KillWaves  Numer, Denom, QDiff, PDiff, PSum, RotationMag
	
	//Convert rotation axis and direction to a rotation tensor, Psi, through common solution see https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle for details 
	make/o/n=(3,3) u_x = {{0,u[2],-u[1]},{-u[2],0,u[0]},{u[1],-u[0],0}} //Calculates cross product matrix of u
	MatrixOP/O Psi = cos(psi1)*Identity(3)+sin(psi1)*u_x+(1-cos(psi1))*OuterProduct(u,u) 
	
	//Find deformation tensor, DefTensor = Psi x F (E in the paper)
	MatrixOP/O DefTensor = Psi x F
	
end

function shearangle()
wave DefTensor, InterfaceNormal

	MatrixOP/O En = DefTensor x InterfaceNormal
	Make/o/n=3 MagEN = sqrt(En[0]^2+En[1]^2+En[2]^2)
	Make/o/n=3 nprime = En/MagEn
	MatrixOP/O ndotn = InterfaceNormal.nprime
	make/o/n=1 AngleofShear = acos(ndotn)*(180/pi)
	make/o/n=1 MagofShear = 1-MagEN
	print ("Angle of Shear = "+num2str(AngleofShear[0])+"°. The magnitude of shear = "+num2str(MagofShear[0])+".")

end

function twinplane()
wave DefTensor
	
	make/O/n=2 v1 = {1,0,0}
	make/O/n=2 v2 = {0,1/sqrt(2),1/sqrt(2)}
	MatrixOP/O v1prime = DefTensor x v1
	MatrixOP/O v2prime = DefTensor x v2
	
	Cross/DEST=TwinningPlane v1prime, v2prime
	print ("The twin plane = {"+num2str(TwinningPlane[0])+","+num2str(TwinningPlane[1])+","+num2str(TwinningPlane[2])+"}")
	killwaves v1, v2, v1prime, v2prime
	
end

function InterfaceAngle()
variable i, j, k //Direction of ideal beta interface
wave InterfaceNormal	
	
	make/n=3/o direction = {1,1,1}
	MatrixOP/O AbsInterfaceNormal = Abs(InterfaceNormal)
	WaveStats/Q AbsInterfaceNormal
	direction[V_MaxLoc] = 2
	MatrixOP/O HabitDot = AbsInterfaceNormal.direction
	make/n=1/O MagIN = sqrt(InterfaceNormal[0]^2+InterfaceNormal[1]^2+InterfaceNormal[2]^2)
	make/n=1/O MagD = sqrt(Direction[0]^2+Direction[1]^2+Direction[2]^2)
	make/n=1/O IAngle = (acos(HabitDot[0]/(MagIN[0]*MagD[0])))*(180/pi)
	print IAngle
	Killwaves AbsInterfaceNormal
	
end

function TransformationMatrices() //Calculates full transformation matrices between the two bases
wave T1, T2, Psi, Phi

	MatrixOP/O M1 = Psi //x T1 //M1 is the distortions to which twin region 1 is subjected
	MatrixOP/O M2 = Psi x Phi //x T2
	Make/o/n=(3,3) GeneralRotation = {{1/sqrt(2),1/sqrt(2),0},{-1/sqrt(2),1/sqrt(2),0},{0,0,1}}
	Make/o/n=(3,3) GeneralRotation2 = {{1/sqrt(2),0,1/sqrt(2)},{0,1,0},{-1/sqrt(2),0,1/sqrt(2)}} //Maps cubic system to the two twin variants
	MatrixOP/O Theta = (M1 x GeneralRotation^T)^T //True mapping taking into account the distortion
	MatrixOP/O Omega = (M2 x GeneralRotation2^T)^T //These convert cubic vector to orthorhombic vector
	MatrixOP/O ThetaInv = Inv(Theta)
	MatrixOP/O OmegaInv = Inv(Omega) //These convert orthorhombic vector to cubic vector
	
end

function OrientationRelationships(a, b, c)
variable a, b, c //ADP Lattice parameters
wave Theta, ThetaInv, Omega, OmegaInv

	//Twin 1: 45degree rotation of i and j about k (see Fig. 7 Liebermann et al.): {100} Planes
	Print "Orientation relationship between {100}α″ in Variant 1"
	//(1-10)b and (100)adp
	make/n=3/o ADPWave = {1,0,0}
	MatrixOP/O ADPWave_b = ThetaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, -1, 0}
	variable ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between (1-10)β and (100)α″ = "+num2str(ang)+"°")
	
	//(110)b and (010)adp
	make/n=3/o ADPWave = {0,1,0}
	MatrixOP/O ADPWave_b = ThetaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, 1, 0}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between (110)β and (010)α″ = "+num2str(ang)+"°")
	
	//(001)b and (001)adp
	make/n=3/o ADPWave = {0,0,1}
	MatrixOP/O ADPWave_b = ThetaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {0, 0, 1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between (001)β and (001)α″ = "+num2str(ang)+"°")
	
	//Twin 2: 45degree rotation of i and k about k (see Fig. 7 Liebermann et al.): {100} Planes
	Print "Orientation relationship between {100}α″ in Variant 2"
	//(10-1)b and (100)adp
	make/n=3/o ADPWave = {1,0,0}
	MatrixOP/O ADPWave_b = OmegaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, 0, -1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between (10-1)β and (100)α″ = "+num2str(ang)+"°")
	
	//(010)b and (010)adp
	make/n=3/o ADPWave = {0,1,0}
	MatrixOP/O ADPWave_b = OmegaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {0,1,0}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between (010)β and (010)α″ = "+num2str(ang)+"°")
	
	//(101)b and (001)adp
	make/n=3/o ADPWave = {0,0,1}
	MatrixOP/O ADPWave_b = OmegaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1,0,1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between (101)β and (001)α″ = "+num2str(ang)+"°")
	
	//Twin 1: 45degree rotation of i and j about k (see Fig. 7 Liebermann et al.): <110> Directions
	//The basis  is given by u = bi+cj+ak where a,b and c are the adp lattice parameters
	//[111]b and [011]adp
	Print "Orientation relationship between [111]α″ in Variant 1"
	make/n=3/o ADPWave = {0*b,1*c,1*a}
	ADPWave = (1/sqrt(0*b^2+1*c^2+1*a^2))*ADPWave
	MatrixOP/O ADPWave_b = ThetaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, 1, 1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [111]β and [011]α″ = "+num2str(ang)+"°")
	
	//[11-1]b and [01-1]adp
	make/n=3/o ADPWave = {0*b,1*c,-1*a}
	ADPWave = (1/sqrt(0*b^2+1*c^2+(-1*a)^2))*ADPWave
	MatrixOP/O ADPWave_b = ThetaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, 1, -1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [11-1]β and [01-1]α″ = "+num2str(ang)+"°")
	
	//[1-11]b and [101]adp
	make/n=3/o ADPWave = {1*b,0*c,1*a}
	ADPWave = (1/sqrt(1*b^2+0*c^2+1*a^2))*ADPWave
	MatrixOP/O ADPWave_b = ThetaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, -1, 1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [1-11]β and [101]α″ = "+num2str(ang)+"°")
	
	//[1-1-1]b and [10-1]adp
	make/n=3/o ADPWave = {1*b,0*c,-1*a}
	ADPWave = (1/sqrt(1*b^2+0*c^2+(-1*a)^2))*ADPWave
	MatrixOP/O ADPWave_b = ThetaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, -1, -1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [1-1-1]β and [10-1]α″ = "+num2str(ang)+"°")
	
	//[100]b and [110]adp
	make/n=3/o ADPWave = {1*b,1*c,0*a}
	ADPWave = (1/sqrt(1*b^2+1*c^2+0*a^2))*ADPWave
	MatrixOP/O ADPWave_b = ThetaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, 0, 0}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [100]β and [110]α″ = "+num2str(ang)+"°")
	
	//[0-10]b and [1-10]adp
	make/n=3/o ADPWave = {1*b,-1*c,0*a}
	ADPWave = (1/sqrt(1*b^2+(-1*c)^2+0*a^2))*ADPWave
	MatrixOP/O ADPWave_b = ThetaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {0, -1, 0}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [0-10]β and [1-10]α″ = "+num2str(ang)+"°")
	
	//Twin 2: 45degree rotation of i and j about k (see Fig. 7 Liebermann et al.): <110> Directions
	//The basis  is given by u = bi+aj+ck where a,b and c are the adp lattice parameters
	//[111]b and [011]adp
	Print "Orientation relationship between [111]α″ in Variant 2"
	make/n=3/o ADPWave = {0*b,1*a,1*c}
	ADPWave = (1/sqrt(0*b^2+1*a^2+1*c^2))*ADPWave
	MatrixOP/O ADPWave_b = OmegaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, 1, 1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [111]β and [011]α″ = "+num2str(ang)+"°")
	
	//[11-1]b and [110]adp
	make/n=3/o ADPWave = {1*b,1*a,0*c}
	ADPWave = (1/sqrt(1*b^2+1*a^2+0*c^2))*ADPWave
	MatrixOP/O ADPWave_b = OmegaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, 1, -1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [11-1]β and [110]α″ = "+num2str(ang)+"°")
	
	//[1-11]b and [0-11]adp
	make/n=3/o ADPWave = {0*b,-1*a,1*c}
	ADPWave = (1/sqrt(0*b^2+(-1*a)^2+1*c^2))*ADPWave
	MatrixOP/O ADPWave_b = OmegaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, -1, 1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [1-11]β and [0-11]α″ = "+num2str(ang)+"°")
	
	//[1-1-1]b and [1-10]adp
	make/n=3/o ADPWave = {1*b,-1*a,0*c}
	ADPWave = (1/sqrt(1*b^2+(-1*a)^2+0*c^2))*ADPWave
	MatrixOP/O ADPWave_b = OmegaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, -1, -1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [1-1-1]β and [1-10]α″ = "+num2str(ang)+"°")
	
	//[100]b and [101]adp
	make/n=3/o ADPWave = {1*b,0*a,1*c}
	ADPWave = (1/sqrt(1*b^2+0*a^2+1*c^2))*ADPWave
	MatrixOP/O ADPWave_b = OmegaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {1, 0, 0}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [100]β and [101]α″ = "+num2str(ang)+"°")
	
	//[00-1]b and [10-1]adp
	make/n=3/o ADPWave = {1*b,0*a,-1*c}
	ADPWave = (1/sqrt(1*b^2+0*a^2+(-1*c)^2))*ADPWave
	MatrixOP/O ADPWave_b = OmegaInv x ADPWave // Converts vector in adp system to that in cubic
	make/n=3/o CubeWave = {0, 0, -1}
	ang = AngleBetweenDir(CubeWave, ADPWave_b)
	print ("Angle between [00-1]β and [10-1]α″ = "+num2str(ang)+"°")
	
end

function AngleBetweenDir(CubeDir, OrthDir) //CubeDir and OrthDir should be in the same basis
wave CubeDir, OrthDir
	
	variable Mag1 = sqrt(CubeDir[0]^2+CubeDir[1]^2+CubeDir[2]^2)
	variable Mag2 = sqrt(OrthDir[0]^2+OrthDir[1]^2+OrthDir[2]^2)
	MatrixOP/O DotProduct = CubeDir.OrthDir
	variable Ang = (180/pi)*acos(DotProduct[0]/(Mag1*Mag2))
	
	killwaves DotProduct
	return Ang

end