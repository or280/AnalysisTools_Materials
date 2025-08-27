#pragma TextEncoding = "UTF-8"
#pragma rtGlobals=3				// Use modern global access method and strict wave access
#pragma DefaultTab={3,20,4}		// Set default tab width in Igor Pro 9 and later

// Calculator to determine the PTMT values for the transformation of B2 to B19' in NiTi alloys.
// This calculator doesn't consider the twinning (or ratio thereof) because this ratio will very depending on the twins forming. I may add this at a later stage.
// a0 is parameter of B2 and a, b, c, g are the parameters of the B19' phase.
// Output is the transformation matrix between the two phases, which then indicates other things about the transformation

// Authors: OG Reed, July 2023

// Calls function to find matrices and key parameters
function PTMTNiTi(a0, a, b, c, g0)
	variable a0, a, b, c, g0
	variable detU, g
	
	g = g0*pi/180
	
	FindU(a0, a, b, c, g)
	wave Uwave
	matrixOP/O U = Uwave
	
	MatrixEigenV/SYM/EVEC Uwave //Computes the eigenvalues allowing diagonalisation
	wave W_eigenValues, M_eigenVectors
	make/O/N=3 EigenValues = real(W_eigenValues)
	make/O/N=(3,3) Ud = {{EigenValues[0],0,0},{0,EigenValues[1],0},{0,0,EigenValues[2]}}
	detU = Eigenvalues[0] * Eigenvalues[1] * Eigenvalues[2]
	
	print U
	print "Eigenvalues = " + num2str(Eigenvalues[0]) + ", " + num2str(Eigenvalues[1]) + ", " + num2str(Eigenvalues[2])
	print "detU = " + num2str(detU)

end
	
	
// Function to find the transformation matrix for a cubic to monoclinic transformation given the lattice parameters.
function FindU(a0, a, b, c, g)
	variable a0, a, b, c, g
	variable gamma0,epsilon0,alpha0,delta0
	
	gamma0 = (a*(sqrt(2)*a+c*sin(g)))/(a0*sqrt(2*a^2+c^2+2*sqrt(2)*a*c*sin(g)))
	epsilon0 = (a*c*cos(g))/(sqrt(2)*a0*sqrt(2*a^2+c^2+2*sqrt(2)*a*c*sin(g)))
	alpha0 = ((c*(c+sqrt(2)*a*sin(g))/sqrt(2*a^2+c^2+2*sqrt(2)*a*c*sin(g)))+b)/(2*sqrt(2)*a0)
	delta0 = ((c*(c+sqrt(2)*a*sin(g))/sqrt(2*a^2+c^2+2*sqrt(2)*a*c*sin(g)))-b)/(2*sqrt(2)*a0)
	
	make/O/N=(3,3) Uwave = {{gamma0,epsilon0,epsilon0},{epsilon0,alpha0,delta0},{epsilon0,delta0,alpha0}}
	
end

