#ifndef OPTIMISER_H
#define OPTIMISER_H

#include <stdio.h>

void initialise_optimiser(double learning_rate, int batch_size, int total_epochs);
void run_optimisation(void);

void initialise_optimiser_2(double learning_rate_0,double learning_rate_N, double momentum_coeff, int batch_size, int total_epochs,int run_ID);
void run_optimisation_2(void);

void initialise_optimiser_3(double learning_rate, double rms_coeff, double delta, int batch_size, int total_epochs, int run_ID);
void run_optimisation_3(void);

double evaluate_objective_function(unsigned int sample);
double evaluate_objective_function_reduced(unsigned int sample);

void numerical_gradient_check(unsigned int sample);

#endif /* OPTMISER_H */
