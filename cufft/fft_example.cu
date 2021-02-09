#include <cufft.h>
#include <complex>
#include <vector>
#include <iostream>
#include "../common/timers.hpp"
#include "../common/error_macros.h"
#define N_DIM 8192
#define N_PRINTS 30



namespace dft_contants 
{
    const std::complex<double> PI {{ 3.14159, 0 }};
    const std::complex<double> IM_UNIT {{0, 1}};

    const double FREQ = 3.0;

}
namespace dft_functions 
{
    std::vector<std::complex<double>> dft(const std::vector<std::complex<double> > &x) 
    {
        std::vector<std::complex<double> > dft(x.size()); // output dft
        double xN = double(x.size()); // samples 

        std::complex<double> temp = {{0, 0 }};  // temp accumulator

        for (int k = 0; k < x.size(); ++k) {
            for (int n = 0; n < x.size(); ++n) {
                temp = {{double(-1*2*k*n) / xN, 0 }};
                dft[k] += x[n] * exp(dft_contants::IM_UNIT * dft_contants::PI * temp); // divide by N to normalize
            }
        }
        return dft; 
    }

    void generate_fake_signal(std::vector<std::complex<double> > &x, size_t dim) 
    {
        double delta = (dft_contants::PI.real() / dft_contants::FREQ);
        for (int i = 0; i < dim; ++i) {
            x.push_back(cos(i * delta));
        }
    }

    void vector_to_cufftComplex(const std::vector<std::complex<double> > &x, cufftComplex **complx, size_t Nx) 
    {
        (*complx) = (cufftComplex*) malloc(sizeof(cufftComplex) * Nx);

        for (int i  = 0; i < Nx; ++i)
        {
            (*complx)[i].x = x[i].real();
            (*complx)[i].y = x[i].imag();
        }
    }
}


int main() {
    int i;
    
    helpers::CPUTimer timer;

    std::vector<std::complex<double> > samples;
    dft_functions::generate_fake_signal(samples, N_DIM);

    cufftHandle plan = 0;
    cufftComplex *complexSamples, *complexSamples_d, *complexFreq;

    complexFreq = (cufftComplex *) malloc( sizeof (cufftComplex) * N_DIM);

    dft_functions::vector_to_cufftComplex(samples, &complexSamples, N_DIM);

  

    printf("Initial samples:\n");

    for(i=0; i < N_PRINTS; ++i) {
        printf(" %2.4f\n", samples[i]);
    }
    printf("...\n");

    timer.start();
    std::vector<std::complex<double>> complexFreq_h = dft_functions::dft(samples);
    double elapsed = timer.stop();
    printf("Naive impl. output samples:\n");

    for(i=0; i < N_PRINTS; ++i) {
        printf("  %d: (%2.4f, %2.4f)\n", i + 1, complexFreq_h[i].real(),
        complexFreq_h[i].imag());
    }
    printf("...\n");
    printf("Elapsed on CPU: %f \n", elapsed);

    // setup cuFFT plan
    CHECK_CUFFT(cufftPlan1d(&plan, N_DIM, CUFFT_C2C, 1));
    CHECK(cudaMalloc((void**) &complexSamples_d, sizeof(cufftComplex) * N_DIM));

    CHECK(cudaMemcpy(complexSamples_d, complexSamples, sizeof(cufftComplex) * N_DIM, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    CHECK_CUFFT(cufftExecC2C(plan, complexSamples_d, complexSamples_d, CUFFT_FORWARD));
    cudaEventRecord(stop);
    
    CHECK(cudaMemcpy(complexFreq, complexSamples_d, sizeof(cufftComplex) * N_DIM, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float elapsed_d = 0;
    cudaEventElapsedTime(&elapsed_d, start, stop);

    printf("Fourier coefficients: \n");
    for(i=0; i < N_PRINTS; ++i) {
        printf("  %d: (%2.4f, %2.4f)\n", i + 1, complexFreq[i].x,
               complexFreq[i].y);
    }
    printf("... \n");
    printf("Elapsed on Cuda: %f \n", elapsed_d);


    free(complexSamples);
    free(complexFreq);

    CHECK(cudaFree(complexSamples_d));
    CHECK_CUFFT(cufftDestroy(plan));

}