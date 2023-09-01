#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mnist_helper.h"
#include "neural_network.h"
#include "optimiser.h"

void print_help_and_exit(char **argv) {
    printf("usage: %s <path_to_dataset> <learning_rate> <rms_coeff> <delta> <batch_size> <total_epochs> <run_ID>\n", argv[0]);
    exit(0);
}

int main(int argc, char** argv) {
    
    
    if(argc != 8){
        printf("ERROR: incorrect number of arguments\n");
        print_help_and_exit(argv);
    }
    
    //Part 1
    const char* path_to_dataset = argv[1];
    double learning_rate = atof(argv[2]);
    double rms_coeff = atof(argv[3]);
    double delta = atof(argv[4]);
    unsigned int batch_size = atoi(argv[5]);
    unsigned int total_epochs = atoi(argv[6]);
    unsigned int run_ID = atoi(argv[7]);

    
    if(!path_to_dataset || !learning_rate || !delta || !batch_size || !total_epochs) {
        printf("ERROR: invalid argument\n");
        print_help_and_exit(argv);
    }
    
    printf("********************************************************************************\n");
    printf("Initialising Dataset... \n");
    printf("********************************************************************************\n");
    initialise_dataset(path_to_dataset,
                       0 // print flag
                       );

    printf("********************************************************************************\n");
    printf("Initialising neural network... \n");
    printf("********************************************************************************\n");
    initialise_nn();

    printf("********************************************************************************\n");
    printf("Initialising optimiser...\n");
    printf("********************************************************************************\n");
    initialise_optimiser_3(learning_rate,rms_coeff, delta, batch_size, total_epochs,run_ID);



    printf("********************************************************************************\n");
    printf("Performing training optimisation...\n");
    printf("********************************************************************************\n");
    run_optimisation_3();
    
    printf("********************************************************************************\n");
    printf("Program complete... \n");
    printf("********************************************************************************\n");
    free_dataset_data_structures();
    return 0;
}
