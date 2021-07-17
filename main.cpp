/* pseudospectral rk4-integrator -- KdV equation
 * compile using for example: g++ main.cpp -o run -lfftw3 -std=c++11 -O3
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <cstdlib> // random numbers //
#include <fftw3.h>


// discretization points in real space 
const int N = 384;
// constant PI
const double PI = 4.0 * atan(1.0);
// current time for the model and time step length
double systemtime = 0.0, dtime = 5.0e-5;
// system size
double systemsize = 30.0 * PI;

// whether or not the Fourier transformation plans have been allocated
bool rk4_initialized = false;
// real space data array + dummy used in the rk4 scheme
double rdata[N], rdata_working[N];
// fftw transformation plans
fftw_complex cdata[N / 2 + 1], cdata_working[N / 2 + 1];
fftw_plan rk4_tofourier, rk4_toreal, rk4_tofourier_working, rk4_toreal_working;

void rwritetofile( double rwarray[N], int number );
void cwritetofile( fftw_complex cwarray[N / 2 + 1] );
void tofourierspace( fftw_plan ptofourierspace, double input[N], fftw_complex output[N / 2 + 1] );
void singletofourierspace( double input[N], fftw_complex output[N / 2 + 1] );
void torealspace( fftw_plan ptorealspace, fftw_complex input[N / 2 + 1], double output[N] );
void singletorealspace( fftw_complex input[N / 2 + 1], double output[N] );
void rcopyarray( int ilength, double input[], double output[]);
void ccopyarray( int ilength, fftw_complex input[], fftw_complex output[]);
void derivative( fftw_complex input[N / 2 + 1], fftw_complex output[N / 2 + 1] );

void pde( fftw_plan ptofourierspace, fftw_plan ptorealspace, double rinput[N], fftw_complex cinput[N / 2 + 1], fftw_complex output[N / 2 + 1] );
void rk4step(int nsteps);

void initializeSoliton(int ilength, double rinput[N], double routput[N], double velocity, double position);

const double nonlinparameter = 1.0f;

int main(int argc, char *argv[]){
	// initializing //
	initializeSoliton(N, rdata, rdata, 2.0, 6.0*PI);
	initializeSoliton(N, rdata, rdata, 1.0, 12.0*PI);
	initializeSoliton(N, rdata, rdata, 0.5, 18.0*PI);
	initializeSoliton(N, rdata, rdata, 0.25, 24.0*PI);
	singletofourierspace( rdata, cdata );
	rwritetofile(rdata, 0);
	cwritetofile(cdata);
	
	for (int mi = 0; mi < 4000; mi++){
		rk4step(2000);
		rwritetofile(rdata, mi + 1);
		std::cout << mi << " " << rdata[0] << std::endl;
	}
	
	if (rk4_initialized) {
		fftw_destroy_plan(rk4_tofourier);
		fftw_destroy_plan(rk4_tofourier_working);
		fftw_destroy_plan(rk4_toreal);
		fftw_destroy_plan(rk4_toreal_working);
	}
	return 0;
}
void initializeSoliton(int ilength, double rinput[N], double routput[N], double velocity, double position) {
	double position0 = std::fmod(position, systemsize);
	if (position0 < 0.0) position0 +=systemsize;
	for (int i = 0; i < ilength; i++){
		routput[i] = rinput[i] + 3.0 * velocity/nonlinparameter  * (
						1.0 / pow(cosh(0.5 * sqrt(fabs(velocity)) * ((double)i * systemsize / (double)ilength - position0)), 2)
						+ 1.0 / pow(cosh(0.5 * sqrt(fabs(velocity)) * ((double)i * systemsize / (double)ilength - position0 - systemsize)), 2)
						+ 1.0 / pow(cosh(0.5 * sqrt(fabs(velocity)) * ((double)i * systemsize / (double)ilength - position0 + systemsize)), 2)
						);
	}
}
void rk4step(int nsteps){
	fftw_complex cdata_initial[N / 2 + 1];
	fftw_complex k1[N / 2 + 1], k2[N / 2 + 1], k3[N / 2 + 1], k4[N / 2 + 1];
	
	if (!rk4_initialized) {
		rk4_tofourier = fftw_plan_dft_r2c_1d(N, rdata, cdata, FFTW_ESTIMATE);
		rk4_toreal = fftw_plan_dft_c2r_1d(N, cdata, rdata, FFTW_ESTIMATE);
		rk4_tofourier_working = fftw_plan_dft_r2c_1d(N, rdata_working, cdata_working, FFTW_ESTIMATE);
		rk4_toreal_working = fftw_plan_dft_c2r_1d(N, cdata_working, rdata_working, FFTW_ESTIMATE);
		rk4_initialized = true;
	}
	for (int ri = 0; ri < nsteps; ri++) {
		// dealiasing //
		/*for(int ei = N / 2 - 10; ei < N / 2 + 1; ei++) {
			cdata[ei][0] = 0.0;
			cdata[ei][1] = 0.0;
		}*/
		ccopyarray( N, cdata, cdata_working);
		ccopyarray( N, cdata, cdata_initial);
		
		// k1 - step //
		pde(rk4_tofourier_working, rk4_toreal_working, rdata_working, cdata_working, k1);
		
		// k2 - step //
		for(int ri = 0; ri < N / 2 + 1; ri++) {
			cdata_working[ri][0] = cdata_initial[ri][0] + dtime / 2.0 * k1[ri][0];
			cdata_working[ri][1] = cdata_initial[ri][1] + dtime / 2.0 * k1[ri][1];
		}
		//torealspace( rk4_toreal_working, cdata_working, rdata_working );
		pde(rk4_tofourier_working, rk4_toreal_working, rdata_working, cdata_working, k2);
		
		// k3 - step //
		for(int ri = 0; ri < N / 2 + 1; ri++) {
			cdata_working[ri][0] = cdata_initial[ri][0] + dtime / 2.0 * k2[ri][0];
			cdata_working[ri][1] = cdata_initial[ri][1] + dtime / 2.0 * k2[ri][1];
		}
		//torealspace( rk4_toreal_working, cdata_working, rdata_working );
		pde(rk4_tofourier_working, rk4_toreal_working, rdata_working, cdata_working, k3);
		
		// k4 - step //
		for(int ri = 0; ri < N / 2 + 1; ri++) {
			cdata_working[ri][0] = cdata_initial[ri][0] + dtime * k3[ri][0];
			cdata_working[ri][1] = cdata_initial[ri][1] + dtime * k3[ri][1];
		}
		//torealspace( rk4_toreal_working, cdata_working, rdata_working );
		pde(rk4_tofourier_working, rk4_toreal_working, rdata_working, cdata_working, k4);
		
		// execute step //
		for(int ri = 0; ri < N / 2 + 1; ri++) {
			cdata[ri][0] = cdata_initial[ri][0] + dtime / 6.0 * (k1[ri][0] + 2.0 * k2[ri][0] + 2.0 * k3[ri][0] + k4[ri][0]);
			cdata[ri][1] = cdata_initial[ri][1] + dtime / 6.0 * (k1[ri][1] + 2.0 * k2[ri][1] + 2.0 * k3[ri][1] + k4[ri][1]);
		}
	}
	torealspace( rk4_toreal, cdata, rdata );
	systemtime += ((double) nsteps) * dtime;
	return;
}
void pde( fftw_plan ptofourierspace, fftw_plan ptorealspace, double rinput[N], fftw_complex cinput[N / 2 + 1], fftw_complex output[N / 2 + 1] ){
	double rgradient[N];
	fftw_complex cinputsquared[N / 2 + 1];
	fftw_complex cbackup[N / 2 + 1];
	
	// make backup
	ccopyarray( N, cinput, cbackup );
	// calculate square of field in real space
	torealspace( ptorealspace, cinput, rinput );
	for(int ei = 0; ei < N; ei++) {
		rinput[ei] = rinput[ei] * rinput[ei];
	}
	// transform back
	tofourierspace( ptofourierspace, rinput, cinput );
	ccopyarray( N, cinput, cinputsquared );
	// restore backup
	ccopyarray( N, cbackup, cinput );

	for(int ei = 0; ei < N / 2 + 1; ei++) {
		output[ei][0] = -pow(2.0 * PI / systemsize * (double)ei, 3) * cinput[ei][1] + 0.5 * nonlinparameter * 2.0 * PI / systemsize * (double)ei * cinputsquared[ei][1];
		output[ei][1] = pow(2.0 * PI / systemsize * (double)ei, 3) * cinput[ei][0] - 0.5 * nonlinparameter * 2.0 * PI / systemsize * (double)ei * cinputsquared[ei][0];
	}
	return;
}

/* ###############################################################################
   ######################    Supporting fftw3 functions     ######################
   ############################################################################### */

void derivative( fftw_complex input[N / 2 + 1], fftw_complex output[N / 2 + 1] ){
	for(int di = 0; di < N / 2 + 1; di++) {
		output[di][0] = -(2.0 * PI / systemsize * (double)di) * input[di][1];
		output[di][1] = (2.0 * PI / systemsize * (double)di) * input[di][0];
	}
	return;
}
void tofourierspace( fftw_plan ptofourierspace, double input[N], fftw_complex output[N / 2 + 1] ){
	// normalize as well //
	
	fftw_execute(ptofourierspace);
	for(int mi = 0; mi < N / 2 + 1; mi++) {
		output[mi][0] /= (double)N;
		output[mi][1] /= (double)N;
	}
	
	return;
}
void singletofourierspace( double input[N], fftw_complex output[N / 2 + 1] ){
	// normalize as well //
	fftw_plan ptemp;
	double inputbackup[N];
	
	rcopyarray( N, input, inputbackup);
	ptemp = fftw_plan_dft_r2c_1d(N, input, output, FFTW_ESTIMATE);
	rcopyarray( N, inputbackup, input);
	fftw_execute(ptemp);
	for(int mi = 0; mi < N / 2 + 1; mi++) {
		output[mi][0] /= (double)N;
		output[mi][1] /= (double)N;
	}
	fftw_destroy_plan(ptemp);
	
	return;
}
void torealspace( fftw_plan ptorealspace, fftw_complex input[N / 2 + 1], double output[N] ){
	// make copy of fftwcomplex beforehand since the stored data is lost in the process and revert to input afterwards //
	fftw_complex inputbackup[N / 2 + 1];
	for(int mi = 0; mi < N / 2 + 1; mi++) {
		inputbackup[mi][0] = input[mi][0];
		inputbackup[mi][1] = input[mi][1];
	}
	fftw_execute(ptorealspace);
	for(int mi = 0; mi < N / 2 + 1; mi++) {
		input[mi][0] = inputbackup[mi][0];
		input[mi][1] = inputbackup[mi][1];
	}
	return;
}
void singletorealspace( fftw_complex input[N / 2 + 1], double output[N] ){
	// make copy of fftwcomplex beforehand since the stored data is lost in the process and revert to input afterwards //
	fftw_plan ptemp;
	fftw_complex inputbackup[N / 2 + 1];
	
	ptemp = fftw_plan_dft_c2r_1d(N, inputbackup, output, FFTW_ESTIMATE);
	for(int mi = 0; mi < N / 2 + 1; mi++) {
		inputbackup[mi][0] = input[mi][0];
		inputbackup[mi][1] = input[mi][1];
	}
	fftw_execute(ptemp);
	fftw_destroy_plan(ptemp);
	
	return;
}
void rwritetofile( double rwarray[N], int number){
	std::ofstream outputfile;
	std::ostringstream filename;
	static int wi = 0;
	int icount;

        std::stringstream ss;
        ss << "data/output" << std::setfill('0') << std::setw(5) << number << ".dat";
	outputfile.open(ss.str());
	for (icount = 0; icount < N; icount++){
		outputfile << icount << "\t" << std::scientific << std::setprecision(15) << rwarray[icount]<< std::endl;
	}
	outputfile.close();
	wi++;
	
	return;
}
void cwritetofile( fftw_complex cwarray[N / 2 + 1] ){
	std::ofstream outputfile;
	std::ostringstream filename;
	static int wi = 0;
	int icount;
 
	outputfile.open("data/output.dat");
	for (icount = 0; icount < N / 2 + 1; icount++){
		outputfile << icount << "\t" 
		<< std::scientific << std::setprecision(15) << sqrt(cwarray[icount][0]*cwarray[icount][0] 
							+ cwarray[icount][1]*cwarray[icount][1]) 
		<< std::endl;
	}
	outputfile.close();
	wi++;
	
	return;
}
void rcopyarray( int ilength, double input[], double output[]){
	std::copy ( input, input+ilength, output );
	return;
}
void ccopyarray( int ilength, fftw_complex input[], fftw_complex output[]){
	for(int mi = 0; mi < ilength / 2 + 1; mi++) {
		output[mi][0] = input[mi][0];
		output[mi][1] = input[mi][1];
	}
	return;
}