#include "optimiser.h"
#include "mnist_helper.h"
#include "neural_network.h"
#include "math.h"

// Function declarations
void update_parameters(unsigned int batch_size);
void update_parameters_3(unsigned int batch_size);
void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy);

// Optimisation parameters
unsigned int log_freq = 30000; // Compute and print accuracy every log_freq iterations

unsigned int run_ID;

// Paramters passed from command line arguments
unsigned int num_batches;
unsigned int batch_size;
unsigned int total_epochs;
double learning_rate;

//part 2
double learning_rate_0;
double learning_rate_N;
double momentum_coeff;

//part 3 
double learning_rate;
double rms_coeff;
double delta;


void print_training_stats(unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy){
    printf("Epoch: %u,  Total iter: %u,  Mean Loss: %0.12f,  Test Acc: %f\n", epoch_counter, total_iter, mean_loss, test_accuracy);
}

void print_training_stats_2(unsigned int epoch_counter, double learning_rate, unsigned int total_iter, double mean_loss, double test_accuracy) {
    printf("Epoch: %u, Learning Rate: %0.4f,  Total iter: %u,  Mean Loss: %0.12f,  Test Acc: %f\n", epoch_counter,learning_rate, total_iter, mean_loss, test_accuracy);
}

/*
void write_to_file_training_stats(const char* file_name_nn, unsigned int epoch_counter, unsigned int total_iter, double mean_loss, double test_accuracy) {

    FILE* file_nn_output;
    file_nn_output = fopen(file_name_nn, "w");
    fprintf(file_nn_output, "Epoch_Counter Total_Iter Mean_Loss Test_Accuracy \n");

}*/

void initialise_optimiser(double cmd_line_learning_rate, int cmd_line_batch_size, int cmd_line_total_epochs){
    batch_size = cmd_line_batch_size;
    learning_rate = cmd_line_learning_rate;
    total_epochs = cmd_line_total_epochs;
    
    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with paramters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tlearning_rate = %f\n\n",
           total_epochs, batch_size, num_batches, learning_rate);
}

void initialise_optimiser_2(double cmd_line_learning_rate_0, double cmd_line_learning_rate_N, double cmd_line_momentum_coeff,
                            int cmd_line_batch_size, int cmd_line_total_epochs, int cmd_line_run_ID) {
    batch_size = cmd_line_batch_size;
    learning_rate_0 = cmd_line_learning_rate_0;
    learning_rate_N = cmd_line_learning_rate_N;
    momentum_coeff = cmd_line_momentum_coeff;
    total_epochs = cmd_line_total_epochs;
    run_ID = cmd_line_run_ID;

    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with parameters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tinitial_learning_rate = %f\n\tfinal_learning_rate = %f\n\tmomentum_coefficient = %f \n\n",
        total_epochs, batch_size, num_batches, learning_rate_0,learning_rate_N,momentum_coeff);
}

void initialise_optimiser_3(double cmd_line_learning_rate, double cmd_line_rms_coeff, double cmd_line_delta,
    int cmd_line_batch_size, int cmd_line_total_epochs, int cmd_line_run_ID) {
    batch_size = cmd_line_batch_size;
    learning_rate = cmd_line_learning_rate;
    rms_coeff = cmd_line_rms_coeff;
    delta = cmd_line_delta;
    total_epochs = cmd_line_total_epochs;
    run_ID = cmd_line_run_ID;

    num_batches = total_epochs * (N_TRAINING_SET / batch_size);
    printf("Optimising with parameters: \n\tepochs = %u \n\tbatch_size = %u \n\tnum_batches = %u\n\tlearning_rate = %f\n\trms_coefficient = %f\n\tdelta_constant = %f \n\n",
        total_epochs, batch_size, num_batches, learning_rate, rms_coeff, delta);
}

void run_optimisation(void){
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;

    /*
    FILE* file_nn_output;
    char filename[0x100];
    snprintf(filename, sizeof(filename), "nu_%0.3f_b%u_ep%u.txt", learning_rate, batch_size, total_epochs);
    file_nn_output = fopen(filename, "w");
    fprintf(file_nn_output, "Epoch_Counter Total_Iter Mean_Loss Test_Accuracy \n");
    */

   // FILE* file_gradient_check;
    //char filename_2[0x100];
    //snprintf(filename_2, sizeof(filename), "gradient_check_nu%0.3f_b%u.txt", learning_rate, batch_size);
    //file_gradient_check = fopen(filename_2, "w");
    //fprintf(file_gradient_check, "Analytical_dw Forward_dw \n");

    FILE* file_gradient_check;
    char filename_num[0x100];
    snprintf(filename_num, sizeof(filename_num), "gradient_check_nu%0.3f.txt", learning_rate);
    file_gradient_check = fopen(filename_num, "w");
    //fclose(file_gradient_check);
    fprintf(file_gradient_check, "Weight_term Analytical_dw Forward_dw \n");

    // Run optimiser - update parameters after each minibatch
    for (int i=0; i < num_batches; i++){
        for (int j = 0; j < batch_size; j++){
            // Evaluate accuracy on testing set (expensive, evaluate infrequently)
            if (total_iter % log_freq == 0 || total_iter == 0){
                if (total_iter > 0){
                    mean_loss = mean_loss/((double) log_freq);
                }
                test_accuracy = evaluate_testing_accuracy();
                print_training_stats(epoch_counter, total_iter, mean_loss, test_accuracy);
                //fprintf(file_nn_output, "%d %d %0.60f %0.60f \n", epoch_counter, total_iter,mean_loss,test_accuracy);
                // Reset mean_loss for next reporting period
                mean_loss = 0.0;
            }
            // Evaluate forward pass and calculate gradients
            obj_func = evaluate_objective_function(training_sample);
            mean_loss+=obj_func;
            // Update iteration counters (reset at end of training set to allow multiple epochs)
            if ((training_sample == 1) && (batch_size == 1)) {
                printf("gradient check");
                numerical_gradient_check(training_sample);
            }

            total_iter++;
            training_sample++;

            // On epoch completion:
            if (training_sample == N_TRAINING_SET){
                training_sample = 0;
                epoch_counter++;
            }
        }
        // Update weights on batch completion
        update_parameters(batch_size);
    }
    // Print final performance
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss/((double) log_freq)), test_accuracy);
}

//void numerical_gradient_check(unsigned int sample, double num_step, struct)

void numerical_gradient_check(unsigned int sample) {
//check using literature 
//update each weight and do this once, reset the weights though and use a reduced format for evluate objectvie function
//also potentially check converge by exploring much longer epochs 

    double L_original_3_O;
    double L_original_2_3;
    double L_original_1_2;
    double L_original_I_1;

    double L_update_3_O;
    double L_update_2_3;
    double L_update_1_2;
    double L_update_I_1;

    double w_original_3_O;
    double w_original_2_3;
    double w_original_1_2;
    double w_original_I_1;

    double numerical_dw_3_O;
    double numerical_dw_2_3;
    double numerical_dw_1_2;
    double numerical_dw_I_1;

    double d = 0.000001;

    w_original_3_O = w_L3_LO[1][1].w;
    L_original_3_O = evaluate_objective_function_reduced(sample);
    //perform small perturbation to w
    w_L3_LO[1][1].w = w_L3_LO[1][1].w + d;
    //calculate loss from perturbed value
    L_update_3_O = evaluate_objective_function_reduced(sample);
    numerical_dw_3_O = (L_update_3_O - L_original_3_O) / (d);
    printf("numerical w_L3_LO[1][1] gradient %0.8f \n",numerical_dw_3_O);
    printf("Analytical w_L3_LO[1][1] gradient %0.8f \n", w_L3_LO[1][1].dw);
    w_L3_LO[1][1].w = w_original_3_O;
    
    w_original_2_3 = w_L2_L3[1][1].w;
    L_original_2_3 = evaluate_objective_function_reduced(sample);
    w_L2_L3[1][1].w = w_L2_L3[1][1].w + d;
    L_update_2_3 = evaluate_objective_function_reduced(sample);
    numerical_dw_2_3 = (L_update_2_3 - L_original_2_3) / (d);
    printf("numerical w_L2_L3[1][1] gradient %0.8f \n", numerical_dw_2_3);
    printf("Analytical w_L2_L3[1][1] gradient %0.8f \n", w_L2_L3[1][1].dw);
    w_L2_L3[1][1].w = w_original_2_3;

    w_original_1_2 = w_L1_L2[1][1].w;
    L_original_1_2 = evaluate_objective_function_reduced(sample);
    w_L1_L2[1][1].w = w_L1_L2[1][1].w + d;
    L_update_1_2 = evaluate_objective_function_reduced(sample);
    numerical_dw_1_2 = (L_update_1_2 - L_original_1_2) / (d);
    printf("numerical w_L1_L2[1][1] gradient %0.8f \n", numerical_dw_1_2);
    printf("Analytical w_L1_L2[1][1] gradient %0.8f \n", w_L1_L2[1][1].dw);
    w_L1_L2[1][1].w = w_original_1_2;

    w_original_I_1 = w_LI_L1[1][1].w;
    L_original_I_1 = evaluate_objective_function_reduced(sample);
    w_LI_L1[1][1].w = w_LI_L1[1][1].w + d;
    L_update_I_1 = evaluate_objective_function_reduced(sample);
    numerical_dw_I_1 = (L_update_I_1 - L_original_I_1) / (d);
    printf("numerical w_LI_L1[1][1] gradient %0.8f \n", numerical_dw_I_1);
    printf("Analytical w_LI_L1[1][1] gradient %0.8f \n", w_LI_L1[1][1].dw);
    w_LI_L1[1][1].w = w_original_I_1;

    /*
    FILE* file_gradient_check;
    char filename_num[0x100];
    snprintf(filename_num, sizeof(filename_num), "gradient_check_nu%0.3f.txt", learning_rate);
    file_gradient_check = fopen(filename_num, "w");
    //fclose(file_gradient_check);
    fprintf(file_gradient_check, "Weight_term Analytical_dw Forward_dw \n");
    */
    //fprintf(file_gradient_check, "Weight_term Analytical_dw Forward_dw \n");
    //fprintf(file_gradient_check, "w_LI_L1[0][0] %0.8f %0.8f \n", w_LI_L1[0][0].dw, numerical_dw_I_1);
    //fprintf(file_gradient_check, "w_L1_L2[0][0] %0.8f %0.8f \n", w_L1_L2[0][0].dw, numerical_dw_1_2);
    //fprintf(file_gradient_check, "w_L2_L3[0][0] %0.8f %0.8f \n", w_L2_L3[0][0].dw, numerical_dw_2_3);
    //fprintf(file_gradient_check, "w_L3_LO[0][0] %0.8f %0.8f \n", w_L3_LO[0][0].dw, numerical_dw_3_O);

}
void run_optimisation_2(void) {
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;
    double alpha;


    FILE* file_nn_output;
    char filename[0x100];
    snprintf(filename, sizeof(filename), "nu0_%0.3f_nuN_%0.3f_mc%0.3f_b%u_ep%u_r%u.txt", learning_rate_0, learning_rate_N,momentum_coeff, batch_size, total_epochs,run_ID);
    file_nn_output = fopen(filename, "w");
    fprintf(file_nn_output, "Epoch_Counter Total_Iter Mean_Loss Test_Accuracy \n");

    learning_rate = learning_rate_0;

    // Run optimiser - update parameters after each minibatch
    for (int i = 0; i < num_batches; i++) {
        for (int j = 0; j < batch_size; j++) {
            // Evaluate accuracy on testing set (expensive, evaluate infrequently)
            if (total_iter % log_freq == 0 || total_iter == 0) {
                if (total_iter > 0) {
                    mean_loss = mean_loss / ((double)log_freq);
                }
                test_accuracy = evaluate_testing_accuracy();
                print_training_stats_2(epoch_counter,learning_rate, total_iter, mean_loss, test_accuracy);
                fprintf(file_nn_output, "%d %d %0.60f %0.60f \n", epoch_counter, total_iter, mean_loss, test_accuracy);
                // Reset mean_loss for next reporting period
                mean_loss = 0.0;
            }
            // Evaluate forward pass and calculate gradients
            obj_func = evaluate_objective_function(training_sample);
            mean_loss += obj_func;
            // Update iteration counters (reset at end of training set to allow multiple epochs)
            total_iter++;
            training_sample++;
            // On epoch completion:
            if (training_sample == N_TRAINING_SET) {
                training_sample = 0;
                epoch_counter++;
                //Learning rate decay 
                alpha = ((float)epoch_counter)/((float)total_epochs);
                printf("alpha: %0.3f \n",alpha);
                learning_rate = learning_rate_0 * (1 - alpha) + alpha*learning_rate_N;
                printf("learning rate: %0.3f \n",learning_rate);
            }
        }
        // Update weights on batch completion
        update_parameters(batch_size);
    }
    // Print final performance
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss / ((double)log_freq)), test_accuracy);
}

void run_optimisation_3(void) {
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;
    //double alpha;

    FILE* file_nn_output;
    char filename[0x100];
    snprintf(filename, sizeof(filename), "nu%0.3f_rms%0.3f_d%0.9f_b%u_ep%u_r%u.txt", learning_rate, rms_coeff, delta, batch_size, total_epochs, run_ID);
    file_nn_output = fopen(filename, "w");
    fprintf(file_nn_output, "Epoch_Counter Total_Iter Mean_Loss Test_Accuracy \n");

    // Run optimiser - update parameters after each minibatch
    for (int i = 0; i < num_batches; i++) {
        for (int j = 0; j < batch_size; j++) {
            // Evaluate accuracy on testing set (expensive, evaluate infrequently)
            if (total_iter % log_freq == 0 || total_iter == 0) {
                if (total_iter > 0) {
                    mean_loss = mean_loss / ((double)log_freq);
                }
                test_accuracy = evaluate_testing_accuracy();
                print_training_stats_2(epoch_counter, learning_rate, total_iter, mean_loss, test_accuracy);
                fprintf(file_nn_output, "%d %d %0.60f %0.60f \n", epoch_counter, total_iter, mean_loss, test_accuracy);
                // Reset mean_loss for next reporting period
                mean_loss = 0.0;
            }
            // Evaluate forward pass and calculate gradients
            obj_func = evaluate_objective_function(training_sample);
            mean_loss += obj_func;
            // Update iteration counters (reset at end of training set to allow multiple epochs)
            total_iter++;
            training_sample++;
            // On epoch completion:
            if (training_sample == N_TRAINING_SET) {
                training_sample = 0;
                epoch_counter++;

            }
        }
        // Update weights on batch completion
        update_parameters_3(batch_size);
    }
    // Print final performance
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss / ((double)log_freq)), test_accuracy);
}

/*
void run_optimisation_4(void) {
    unsigned int training_sample = 0;
    unsigned int total_iter = 0;
    double obj_func = 0.0;
    unsigned int epoch_counter = 0;
    double test_accuracy = 0.0;  //evaluate_testing_accuracy();
    double mean_loss = 0.0;

    double batch_coeff;

    //double alpha;

    FILE* file_nn_output;
    char filename[0x100];
    snprintf(filename, sizeof(filename), "nu%0.3f_rms%0.3f_d%0.9f_b%u_ep%u_r%u.txt", learning_rate, rms_coeff, delta, batch_size, total_epochs, run_ID);
    file_nn_output = fopen(filename, "w");
    fprintf(file_nn_output, "Epoch_Counter Total_Iter Mean_Loss Test_Accuracy \n");

    while (epoch_counter < total_epochs) {
        // Run optimiser - update parameters after each minibatch
        for (int i = 0; i < num_batches; i++) {
            for (int j = 0; j < batch_size; j++) {
                // Evaluate accuracy on testing set (expensive, evaluate infrequently)
                if (total_iter % log_freq == 0 || total_iter == 0) {
                    if (total_iter > 0) {
                        mean_loss = mean_loss / ((double)log_freq);
                    }
                    test_accuracy = evaluate_testing_accuracy();
                    print_training_stats_2(epoch_counter, learning_rate, total_iter, mean_loss, test_accuracy);
                    fprintf(file_nn_output, "%d %d %0.60f %0.60f \n", epoch_counter, total_iter, mean_loss, test_accuracy);
                    // Reset mean_loss for next reporting period
                    mean_loss = 0.0;
                }
                // Evaluate forward pass and calculate gradients
                obj_func = evaluate_objective_function(training_sample);
                mean_loss += obj_func;
                // Update iteration counters (reset at end of training set to allow multiple epochs)
                total_iter++;
                training_sample++;
                // On epoch completion:
                if (training_sample == N_TRAINING_SET) {
                    training_sample = 0;
                    epoch_counter++;

                }
            }
            // Update weights on batch completion
            update_parameters_3(batch_size);
        }

        batch_size = batch_size_0 * (1 - batch_coeff) + batch_coeff * batch_size_n;
        num_batches = total_epochs * (N_TRAINING_SET / batch_size);

    }
    // Print final performance
    test_accuracy = evaluate_testing_accuracy();
    print_training_stats(epoch_counter, total_iter, (mean_loss / ((double)log_freq)), test_accuracy);
}*/

double evaluate_objective_function(unsigned int sample){

    // Compute network performance
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);
    
    // Evaluate gradients
    //evaluate_backward_pass(training_labels[sample], sample);
    evaluate_backward_pass_sparse(training_labels[sample], sample);
    
    // Evaluate parameter updates
    store_gradient_contributions();
    
    return loss;
}

double evaluate_objective_function_reduced(unsigned int sample) {

    // Compute network performance
    evaluate_forward_pass(training_data, sample);
    double loss = compute_xent_loss(training_labels[sample]);

    return loss;
}

/*
void update_parameters_num_check(unsigned int batch_size) {
    // Part I To-do
    double coefficient = ((double)learning_rate / (double)batch_size);
    for (int i = 0; i < N_NEURONS_LI; i++) {
        for (int j = 0; j < N_NEURONS_L1; j++) {

            w_LI_L1[i][j].w =  w_LI_L1[i][j].w - coefficient * w_LI_L1[i][j].dw;
            w_LI_L1[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L1; i++) {
        for (int j = 0; j < N_NEURONS_L2; j++) {
            w_L1_L2[i][j].w = w_L1_L2[i][j].w - coefficient * w_L1_L2[i][j].dw;
            w_L1_L2[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L2; i++) {
        for (int j = 0; j < N_NEURONS_L3; j++) {
            w_L2_L3[i][j].w = w_L2_L3[i][j].w - coefficient * w_L2_L3[i][j].dw;
            w_L2_L3[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L3; i++) {
        for (int j = 0; j < N_NEURONS_LO; j++) {
            w_L3_LO[i][j].w = w_L3_LO[i][j].w - coefficient * w_L3_LO[i][j].dw;
            w_L3_LO[i][j].dw = 0;
        }
    }
}*/

void update_parameters(unsigned int batch_size){
    // Part I To-do
    double coefficient =  ((double)learning_rate / (double)batch_size);
    for (int i = 0; i < N_NEURONS_LI; i++) {
        for (int j = 0; j < N_NEURONS_L1; j++) {
            w_LI_L1[i][j].v = momentum_coeff * w_LI_L1[i][j].v - coefficient * w_LI_L1[i][j].dw;
            w_LI_L1[i][j].w = w_LI_L1[i][j].w + w_LI_L1[i][j].v;
            w_LI_L1[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L1; i++) {
        for (int j = 0; j < N_NEURONS_L2; j++) {
            w_L1_L2[i][j].v = momentum_coeff * w_L1_L2[i][j].v - coefficient * w_L1_L2[i][j].dw;
            w_L1_L2[i][j].w = w_L1_L2[i][j].w + w_L1_L2[i][j].v;
            w_L1_L2[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L2; i++) {
        for (int j = 0; j < N_NEURONS_L3; j++) {
            w_L2_L3[i][j].v = momentum_coeff * w_L2_L3[i][j].v - coefficient * w_L2_L3[i][j].dw;
            w_L2_L3[i][j].w = w_L2_L3[i][j].w + w_L2_L3[i][j].v;
            w_L2_L3[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L3; i++) {
        for (int j = 0; j < N_NEURONS_LO; j++) {
            w_L3_LO[i][j].v = momentum_coeff * w_L3_LO[i][j].v - coefficient * w_L3_LO[i][j].dw;
            w_L2_L3[i][j].w = w_L3_LO[i][j].w + w_L3_LO[i][j].v;
            w_L3_LO[i][j].dw = 0;
        }
    }
}

void update_parameters_3(unsigned int batch_size) {
    // Part I To-do

    double coefficient = ((double)learning_rate / (double)batch_size);
    double coefficient_p = ((double)(1 - rms_coeff) / (double)(pow(batch_size, 2)));
    for (int i = 0; i < N_NEURONS_LI; i++) {
        for (int j = 0; j < N_NEURONS_L1; j++) {
            if ((i <3 )&&(j<3)) {
                //printf("%0.6f %0.6f %0.6f %0.6f\n", rms_coeff * w_LI_L1[i][j].p, coefficient_p*pow(((double)w_LI_L1[i][j].dw), 2), 
                  //  (double)(sqrt(delta + w_LI_L1[i][j].p)), coefficient * ((double)w_LI_L1[i][j].dw / (double)(delta+sqrt(w_LI_L1[i][j].p))));

                //printf("%0.6f %0.6f \n", rms_coeff * w_LI_L1[i][j].p + coefficient_p * pow(((double)w_LI_L1[i][j].dw), 2),
                  //  w_LI_L1[i][j].w - coefficient * ((double)w_LI_L1[i][j].dw / (double)(sqrt(delta + w_LI_L1[i][j].p))));
            }

            w_LI_L1[i][j].p = rms_coeff * w_LI_L1[i][j].p + coefficient_p*pow(((double)w_LI_L1[i][j].dw),2);
            w_LI_L1[i][j].w = w_LI_L1[i][j].w - coefficient*((double)w_LI_L1[i][j].dw/(double)(sqrt(delta+ w_LI_L1[i][j].p)));
            w_LI_L1[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L1; i++) {
        for (int j = 0; j < N_NEURONS_L2; j++) {
            w_L1_L2[i][j].p = rms_coeff * w_L1_L2[i][j].p + coefficient_p*pow(((double)w_L1_L2[i][j].dw),2);
            w_L1_L2[i][j].w = w_L1_L2[i][j].w - coefficient*((double)w_L1_L2[i][j].dw/(double)(sqrt(delta+w_L1_L2[i][j].p)));
            w_L1_L2[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L2; i++) {
        for (int j = 0; j < N_NEURONS_L3; j++) {
            w_L2_L3[i][j].p = rms_coeff * w_L2_L3[i][j].p + coefficient_p *pow(((double)w_L2_L3[i][j].dw),2);
            w_L2_L3[i][j].w = w_L2_L3[i][j].w - coefficient*((double)w_L2_L3[i][j].dw/ (double)(sqrt(delta+w_L2_L3[i][j].p)));
            w_L2_L3[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L3; i++) {
        for (int j = 0; j < N_NEURONS_LO; j++) {
            w_L3_LO[i][j].p = rms_coeff * w_L3_LO[i][j].p + coefficient_p *pow(((double)w_L3_LO[i][j].dw),2);
            w_L3_LO[i][j].w = w_L3_LO[i][j].w - coefficient*((double)w_L3_LO[i][j].dw/(double)(sqrt(delta+w_L3_LO[i][j].p)));
            w_L3_LO[i][j].dw = 0;
        }
    }
}

/*
void check_derivatives() {
    // Part I To-do

    int batch_size = 1;
    double coefficient = (learning_rate / (double)batch_size);

    for (int i = 0; i < N_NEURONS_LI; i++) {
        for (int j = 0; j < N_NEURONS_L1; j++) {
            //w_LI_L1[i][j].w = w_LI_L1[i][j].w - coefficient * w_LI_L1[i][j].dw;
            w_LI_L1[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L1; i++) {
        for (int j = 0; j < N_NEURONS_L2; j++) {
            w_L1_L2[i][j].w = w_L1_L2[i][j].w - coefficient * w_L1_L2[i][j].dw;
            w_L1_L2[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L2; i++) {
        for (int j = 0; j < N_NEURONS_L3; j++) {
            w_L2_L3[i][j].w = w_L2_L3[i][j].w - coefficient * w_L2_L3[i][j].dw;
            w_L2_L3[i][j].dw = 0;
        }
    }

    for (int i = 0; i < N_NEURONS_L3; i++) {
        for (int j = 0; j < N_NEURONS_LO; j++) {
            w_L3_LO[i][j].w = w_L3_LO[i][j].w - coefficient * w_L3_LO[i][j].dw;
            w_L3_LO[i][j].dw = 0;
        }
    }

}*/
