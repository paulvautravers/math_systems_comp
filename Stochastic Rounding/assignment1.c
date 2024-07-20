#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

/* --------------------------------- */
/*              PART 1               */
/* --------------------------------- */
union DoubleToInt {
  double dVal;
  uint64_t iVal;
};

float SR(double x) {
    //rounds a binary64 value to a binary32 value stochastically.
    //Implemented by treating FP number representations as integer values.

  union DoubleToInt temp;
  temp.dVal = x;
  uint32_t r = rand() & 0x1FFFFFFF;
  temp.iVal += r;
  temp.iVal = temp.iVal & 0xFFFFFFFFE0000000;

  return (float)temp.dVal;
}

float* get_neighbouring_values(double sample) {
    //Function to return neighbouring float values
    //from a double precision input
    float closest = (float)sample;
    float down, up;
    static float neighbours[2];

    if (closest > sample) {
        down = nextafterf(closest, -INFINITY);
        up = closest;
    }
    else {
        up = nextafterf(closest, INFINITY);
        down = closest;
    }
    neighbours[0] = down;
    neighbours[1] = up;
    return neighbours;
}

float round_to_zero(double sample) {
    //implementation of RZ rounding
    //handles negative input case
    float rz_value;
    float *neighbours = get_neighbouring_values(sample);
    if (sample < 0) {
        rz_value = neighbours[1];
    }
    else {
        rz_value = neighbours[0];
    }
    return rz_value;
}

float round_away_from_zero(double sample) {
    //implementation of RA rounding
    //handles negative input case
    float ra_value;
    float* neighbours = get_neighbouring_values(sample);
    if (sample < 0) {
        ra_value = neighbours[0];
    }
    else {
        ra_value = neighbours[1];
    }
    return ra_value;
}
  
// Implement SR_alternative according to the Eqn 1.
double get_probability(double x) {
    //calculation of rounding probability q(x)
    //for use in SR_alternative
    float x_rz = round_to_zero(x);
    float x_ra = round_away_from_zero(x);

    double probability = (x - x_rz)/(x_ra - x_rz);
    return probability;
}

float SR_alternative(double x) {
    // Alterative implementation of SR using
    //equation 1 from brief 
    float r = (double)rand()/(double)RAND_MAX;
    double probability = get_probability(x);
    float x_rounded;

    if (r < probability) {
        x_rounded = round_away_from_zero(x);
    }
    else {
        x_rounded = round_to_zero(x);
    }

    return x_rounded;
}

double* get_expected_down_up(double x) {
    //Calculation of expected up down rounding
    //probabilities from theory
    static double down_up_probabilities[2];
    float* neighbours = get_neighbouring_values(x);
    float down = neighbours[0];
    float up = neighbours[1];

    double distance_up = up - x;
    double distance_down = x - down;
    double distance_up_down = up - down;

    down_up_probabilities[0] = distance_up / distance_up_down;
    down_up_probabilities[1] = distance_down / distance_up_down;

    return down_up_probabilities;
}
/*-----------------------------------*/
/*              PART 2               */
/* --------------------------------- */

void fastTwoSum(float x, float y, float* s, float* t) {
    //Function to sum two variables and 
    //capture error induced by addition
    //Used in Compensated summation (Kahan)
    float temp;

    *s = x + y;
    temp = *s - x;
    *t = y - temp;
}

/*-----------------------------------*/
/*              PART 3               */
/* --------------------------------- */

void get_rand_array(double* array, int array_size) {
    //Function to produce 1D array (vector) of 
    //uniformly generated values in bound [0,1)
    for (int i = 0; i < array_size; i++) {
        array[i] = (double)rand() / (double)RAND_MAX;
    }
}

int main() {
  //srand(time(NULL))
  //set seed for random numbers
  srand(1);
  // An arbitrary value for rounding.
  double sample = M_PI;
  /* --------------------------------- */
  /*              PART 1               */
  /* --------------------------------- */

  // Calculate the neighbouring binary32 values.
  float* neighbours = get_neighbouring_values(sample);
  float closest = (float)sample;
  float down = neighbours[0];
  float up = neighbours[1];

  // Round many times, and calculate the average values as well as count
  // the numbers of times rounding was up/down.

  int sr_down_up[2] = { 0,0 };
  int sr_alt_down_up[2] = { 0,0 };

  double sum_sr=0;
  double sum_sr_alt=0;

  const long int K = 5000000;

  float sr_val, sr_alt_val;
  double sr_avg, sr_alt_avg;
  double sr_abs_error, sr_alt_abs_error;

  //generate output txt file recording data, sent to 
  //folder where C code is held
  const char* filename_sr = "sr_averages.txt";
  FILE* file_sr_averages;
  file_sr_averages = fopen(filename_sr, "w");
  fprintf(file_sr_averages, "Index SR_avg SR_Abs_Error SR_alternative_avg SR_alternative_Abs_Error \n");

  for (int i = 0; i < K; i++) {

      sr_val = SR(sample);
      sr_alt_val = SR_alternative(sample);

      //rounding up and down is counted for 
      //both implementations
      if (sr_val > sample) {
          sr_down_up[1] += 1;
      } else {
          sr_down_up[0] += 1;
      }

      if (sr_alt_val > sample) {
          sr_alt_down_up[1] += 1;
      } else {
          sr_alt_down_up[0] += 1;
      }

      //sum for average is incremented
      sum_sr += sr_val;
      sum_sr_alt += sr_alt_val;

      //every 1000th term is written to txt file
      if (i % 1000 == 0) {
          sr_avg = sum_sr / (i + 1);
          sr_abs_error = fabs(sample - sr_avg);

          sr_alt_avg = sum_sr_alt / (i + 1);
          sr_alt_abs_error = fabs(sample - sr_alt_avg);

          fprintf(file_sr_averages, "%d %0.60f %0.60f %0.60f %0.60f \n", i, sr_avg,
              sr_abs_error, sr_alt_avg, sr_alt_abs_error);
      }
  }
  fclose(file_sr_averages);

  // Print out the average of all rounded values
  // Check that SR_alternative function is correct by comparing the probabilities
  // of rounding up/down, and the expected probability. Print them out below

  double* down_up_probs = get_expected_down_up(sample);
  double down_probability = down_up_probs[0];
  double up_probability = down_up_probs[1];

  printf("Printing some useful values from rounding Pi once: \n"
      "###################################################### \n");
  printf("Value being rounded:           %.60f \n", sample);
  printf("SR value:                      %.60f \n", SR(sample));
  printf("SR_alternative value:          %.60f \n", SR_alternative(sample));
  printf("Binary32 value before:         %.60f \n", down);
  printf("Binary32 value after:          %.60f \n", up);
  printf("Closest binary32:              %.60f \n", closest);
  printf("###################################################### \n");

  printf("Printing some useful values from rounding Pi %ld times: \n"
      "###################################################### \n",K);
  //only shown with %.30f here as diverge from each other far earlier and 
  //improves alignment in terminal 
  printf("Expected probability of rounding down and up:        %.30f,\n" 
         "                                                     %.30f \n", down_probability, up_probability);
  printf("SR: Calculated probability of RZ and RA:             %.30f,\n"
         "                                                     %.30f \n",sr_down_up[0]/(double)K, sr_down_up[1]/(double)K);
  printf("SR_alternative: Calculated probability of RZ and RA: %.30f,\n"
         "                                                     %.30f \n", sr_alt_down_up[0]/(double)K,sr_alt_down_up[1]/(double)K);
  printf("SR average value:              %.60f \n", sr_avg);
  printf("SR_alternative average value:  %.60f \n", sr_alt_avg);
  //double precision pi outputted again to aid comparison with two SRs
  printf("Value being rounded:           %.60f \n", sample);
  printf("###################################################### \n");

  /* --------------------------------- */
  /*              PART 2               */
  /* --------------------------------- */

  //files created for output of regular and reversed harmonic summations
  const char* filename_harmonic_sums = "harmonic_sums.txt";
  FILE* file_harmonic_sums;
  file_harmonic_sums = fopen(filename_harmonic_sums, "w");
  fprintf(file_harmonic_sums, "index dharmonic fharmonic fharmonic_error"
          " fharmonic_sr fharmonic_sr_error fharmonic_comp fharmonic_comp_error \n");

  const char* filename_harmonic_sums_reversed = "harmonic_sums_reversed.txt";
  FILE* file_harmonic_sums_reversed;
  file_harmonic_sums_reversed = fopen(filename_harmonic_sums_reversed, "w");
  fprintf(file_harmonic_sums_reversed, "index dharmonic_rev fharmonic_rev fharmonic_rev_error"
      " fharmonic_sr_rev fharmonic_sr_rev_error \n");

  long int N = 500000000;

  //many variables need to be initialised and set to zero
  double dharmonic_val,dharmonic,dharmonic_rev_val,dharmonic_rev = 0;

  float temp_fharmonic,fharmonic, fharmonic_rev = 0;
  double fharmonic_error, fharmonic_rev_error;
  int i_fharmonic_stagnation=0;

  double fharmonic_sr_temp,fharmonic_sr_temp_rev = 0;
  float fharmonic_sr,fharmonic_sr_rev = 0;
  double fharmonic_sr_error,fharmonic_sr_rev_error;

  float fharmonic_comp = 0;
  double fharmonic_comp_error;

  // Error term in the compensated summation.
  float t = 0;

  for (int i = 1; i <= N; i++) {
    //define reverse variable for reversed sums
    int j = N - i + 1;
    //Double recursive summation
    dharmonic_val = (double)1 / (double)i;
    dharmonic += dharmonic_val;

    dharmonic_rev_val = (double)1 / (double)j;
    dharmonic_rev += dharmonic_rev_val;

    // Recursive sum, binary32 RN
    temp_fharmonic += (float)1 / i;
    if (i_fharmonic_stagnation == 0) {
        if (temp_fharmonic == fharmonic) {
            i_fharmonic_stagnation = i;
        }
    }
    fharmonic = temp_fharmonic;
    fharmonic_error = fabs(dharmonic - fharmonic);
    //reverse summation is also performed!
    fharmonic_rev += (float)1 / j;
    fharmonic_rev_error = fabs(dharmonic_rev - fharmonic_rev);

    // Other summation methods.
    //Stochastic rounding of double precision harmonic

    //temporary double precision SR sum is held and 
    // set to value from binary32 after rounding
    fharmonic_sr_temp = fharmonic_sr;
    fharmonic_sr_temp_rev = fharmonic_sr_rev;

    fharmonic_sr_temp += dharmonic_val;
    fharmonic_sr_temp_rev += dharmonic_rev_val;

    fharmonic_sr = SR_alternative(fharmonic_sr_temp);
    fharmonic_sr_rev = SR_alternative(fharmonic_sr_temp_rev);

    fharmonic_sr_error = fabs(dharmonic - fharmonic_sr);
    fharmonic_sr_rev_error = fabs(dharmonic_rev - fharmonic_sr_rev);

    //Compensated summation 
    //error term from FastTwoSum is added to addend to 
    //acount for previous arithmetic induced error
    float addend = (float)1 / (float)i + t;
    fastTwoSum(fharmonic_comp, addend, &fharmonic_comp, &t);
    fharmonic_comp_error = fabs(dharmonic - fharmonic_comp);

    //every 1million terms are written to file
    if (i % 1000000 == 0) {
        fprintf(file_harmonic_sums, "%d %0.60f %0.60f %0.60f %0.60f %0.60f %0.60f %0.60f \n", i, dharmonic,
                fharmonic, fharmonic_error, fharmonic_sr,fharmonic_sr_error, fharmonic_comp,fharmonic_comp_error);

        fprintf(file_harmonic_sums_reversed, "%d %0.60f %0.60f %0.60f %0.60f %0.60f \n", j, dharmonic_rev,
            fharmonic_rev, fharmonic_rev_error,fharmonic_sr_rev, fharmonic_sr_rev_error);
    }
  }
  fclose(file_harmonic_sums);
  fclose(file_harmonic_sums_reversed);

  //Final sums are printed out, with error recorded in txt harmonic_sums.txt
  printf("###################################################### \n");
  printf("Values of the harmonic series after %ld iterations \n", N);
  printf("Recursive summation, binary32:                   %.30f \n", fharmonic);
  printf("Recursive summation with SR, binary32:           %.30f \n", fharmonic_sr);
  printf("Compensated summation, binary32:                 %.30f \n", fharmonic_comp);
  printf("Recursive summation, binary64:                   %.30f \n", dharmonic);
  printf("Recursive summation stagnation index, binary32:  %ld \n", i_fharmonic_stagnation);
  printf("###################################################### \n");

  /* --------------------------------- */
  /*              PART 3               */
  /* --------------------------------- */

  //File made to store output data from inner product calculations
  const char* filename_dot_products = "dot_products.txt";
  FILE* file_dot_products;
  file_dot_products = fopen(filename_dot_products, "w");
  fprintf(file_dot_products, "vector_size d_dot_product upper_error_bound lower_error_bound "
            " f_dot_product f_dot_backward_error f_dot_product_sr f_dot_backward_sr_error \n");

  //unit round off (u) value stored to generate error bounds n*u
  // and sqrt(n)*u for backward error
  double unit_round_off = pow(2, -24);
  double upper_error_bound;
  double lower_error_bound;

  //variables initialised and set to zero
  int size_n;
  float f_dot_product, f_dot_product_sr = 0;
  double mod_vector1, mod_vector2 = 0;
  double d_dot_product, d_dot_product_val, f_dot_product_sr_temp = 0;
  double f_dot_backward_error, f_dot_sr_backward_error = 0;

  //Iteration over n, which specifies the power of 2
  //for the vector size, size_n
  for (int n = 1; n <= 26; n++) {

      size_n = pow(2,n);
      //malloc used for dynamic memory allocation to 
      //create array of pointers at run time
      double* vector1 = malloc(size_n * sizeof(double));
      double* vector2 = malloc(size_n * sizeof(double));

      //initialise vectors with random elements [0,1)
      get_rand_array(vector1, size_n);
      get_rand_array(vector2, size_n);

      //iterate over elements of vector to get inner product
      for (int i = 0; i <= size_n; i++) {

          d_dot_product_val = vector1[i] * vector2[i];
          d_dot_product += d_dot_product_val;

          //moduli of vectors calculated for backwards error
          mod_vector1 += vector1[i] * vector1[i];
          mod_vector2 += vector2[i] * vector2[i];

          //RN rounding used to convert to binary32
          f_dot_product += (float)(d_dot_product_val);

          //temporary SR variable held in double, same 
          //implementation as for harmonic series
          f_dot_product_sr_temp = f_dot_product_sr;
          f_dot_product_sr_temp += d_dot_product_val;
          f_dot_product_sr = SR_alternative(f_dot_product_sr_temp);
      }
      //moduli updated to true modulus value, with sqrt
      mod_vector1 = sqrt(mod_vector1);
      mod_vector2 = sqrt(mod_vector2);

      //backward error calculating using equation from higham ,'what is backward error?'
      f_dot_backward_error = fabs(d_dot_product - f_dot_product) / (mod_vector1 * mod_vector2);
      f_dot_sr_backward_error = fabs(d_dot_product - f_dot_product_sr) / (mod_vector1 * mod_vector2);
      
      //upper and lower RN backward error bounds calculated
      upper_error_bound = unit_round_off * size_n;
      lower_error_bound = unit_round_off * sqrt(size_n);

      //data written to file for every power of 2, n
      fprintf(file_dot_products, "%d %0.60f %0.60f %0.60f %0.60f %0.60f %0.60f %0.60f \n", size_n, d_dot_product,
          upper_error_bound, lower_error_bound, f_dot_product, f_dot_backward_error, f_dot_product_sr, f_dot_sr_backward_error);

      //final innerproduct values are outputted to terminal
      if (n == 26) {
          printf("###################################################### \n");
          printf("Printing values from the dot product of size %d vectors: \n", size_n);
          printf("Dot Product, binary64:          %.30f \n", d_dot_product);
          printf("Dot Product, binary32, RN:      %.30f \n", f_dot_product);
          printf("Dot Product, binary32, SR:      %.30f \n", f_dot_product_sr);
          printf("###################################################### \n");
      }

      //values reset to zero to ensure summations from previous
      //value of n do not add to next n
      mod_vector1 = 0;
      mod_vector2 = 0;
      d_dot_product = 0;
      f_dot_product = 0;
      f_dot_product_sr = 0;
      f_dot_product_sr_temp = 0;
  }
  fclose(file_dot_products);

 return 0;
}
