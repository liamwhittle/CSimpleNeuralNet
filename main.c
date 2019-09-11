/*

Code by Liam Whittle

Any sensible person intent on doing non-trivial machine learning ought to use a
high level language such as python- we really can't do without the power of 
graphics processors, and should try at all costs to avoid re-inventing the wheel. 

I wrote this code a few years prior to uploading to Github. 
It was not initially written with a practical purpose in mind, it was 
more a curious exploration of both the nitty gritty of simple feed-forward 
neural nets, and the c programming language. I wrote it at a time in my 
education where I was learning both of these concepts and wanted to make 
something real. In some ways, the low level, raw nature of C is well suited to 
illustrating the exact maths of neural nets and back propogation.

C Program containing functions designed for training feedforward neural networks.
Features:
	-Utilizes Simple Gradient Descent with back-propogation to find gradient vectors
	-Each hidden layer has the same "HEIGHT" = number of neurons
	-Currently only supports 1 hidden layer
	-Optional weights with -1.0 inputs connected to every neuron to act as a threshold variable 
	-Performance function P = 1/2|d-z|^2
	-Includes two sigmoid threshold functions: logistic, and hyperbolic tangent 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>

// Neural Network structure
#define I_HEIGHT 			2 		// Number of inputs
#define H_HEIGHT 			2 		// Number of neurons in a hidden layer
#define O_HEIGHT 			1		// Number of neurons in the output layer
#define NUM_H_LAYERS 		1 		// Number of hidden layers only
// Trainning
#define NUM_T_VECS 			20 		// Number of trainning data vector sets (1 "vec" is a pair of input and output vectors)
#define DESCENT_FACTOR 		0.1	 	// factor by which we reduce weight vector by gradient vector (is a constant)
#define NUM_DESCENTS		100		// number of times we forward propogate, back propogate and then reduce by gradient vector * DESCENT_FACTOR
#define NUM_CROSSES			100		// the number of times we perform a full set of NUM_DESCENTS descents on each test vector
#define GRAD_TEST_CONSTANT 	1		// what all the gradients are set to for numerical gradient testing
// Hyperbolic tangent
#define PRE_FACTOR			1.7159
#define X_FACTOR			(2.0/3.0)


// This is the only structure we need. It only requires stack memory. If using_threshold == 0 we waste a bit of stack space
typedef struct{
	// Member variable: 1 = use threshold weights, 0 = do not use threshold weights
	int using_threshold;

	// Test data
	double t_x[NUM_T_VECS][I_HEIGHT]; //trainning data input vector  - [vector set number][input index]
	double t_y[NUM_T_VECS][O_HEIGHT]; //trainning data output vector - [vector set number][output index]
	
	// Weights for each connection
	double i_w[H_HEIGHT][I_HEIGHT + 1]; //input weights - [1st hiddden layer neuron index][input index + 1 for threshold value]
	double h_w[NUM_H_LAYERS - 1][H_HEIGHT][H_HEIGHT + 1]; //hidden neuron weights - [layer index, index 0 refers to the second hidden layer (if it exists)][which neuron it connects to][which neuron connects from + 1 for threshold]
	double o_w[O_HEIGHT][H_HEIGHT + 1]; //output weights - [output neuron index][hidden neuron index + 1 for threshhold] 
	
	// Hidden and output neuron forward propogation (post sigmoid) output to synapses
	double h_out[NUM_H_LAYERS][H_HEIGHT]; // - [layer index][neuron index]	
	double f_out[O_HEIGHT]; // - [output index]
	
	// Weights gradient vectors
	double i_grad[H_HEIGHT][I_HEIGHT + 1];
	double h_grad[NUM_H_LAYERS - 1][H_HEIGHT][H_HEIGHT + 1]; 
	double o_grad[O_HEIGHT][H_HEIGHT + 1];
} data;


// Maths functions
double make_pos(double);									// Modulus function for single doubles
double norm_rand();											// random number between -1 and 1. Make sure to seed time in main
double pos_norm_rand();										// returns pos double <= 1. Make sure to seed time in main
double sigmoid(double); 									// returns sigmoid of argument
// Sigmoid based
void f_propogate(data*, int);								// updates all outputs through a forward pass using test vector. HAS NOT been tested on 1 hidden layer
void b_propogate(data*, int);								// updates gradient vector through back-propogation, given a forward propogation has occured									//assigns random weights
double cost(data*);											// performs cost function on current weights and all examples
void fit(data*, int);										// Forward props, back props, and descends NUM_DESCENDS times on &param int example
void cross_fit(data*, int);									// A stochastic method for performing gradient descent: back-props and descends on a random example @param int times
// General
void descend(data*);										// reduce weights by current grad vector * DESCEND_FACTOR
void show_weights(data*);									// show all weights
void show_grad(data*);										// show grad vector
void go_through_weights(data*, void (*)(double*));			// polymorphic: iterates through each weight. takes "init_single_weight" and "print_weight" functions
void init_single_weight(double*);							// to be passed into "go_through_weights"
void init_single_c_weight(double*);							// dereferences &param weight pointer and assigns a random value
void show_weight(double*);									// to be passed into "go_through_weights"
void compare_outputs(data*);								// Forward props given current weights and shows desired outcome v forward proped outcome for all examples
void simple_math_application(data*);						// SEE DEFINITION README BEFORE USING 
void verify_grad(data*);									// SEE DEFINITION README BEFORE USING

// main function runs the simple math application
int main(int argc, char *argv[]){
	data d;

	// seeding
	srand(time(NULL));

	simple_math_application(&d);

	return 0;
}

double make_pos(double d){
	return (d > 0) ? d : -d;
}

void init_single_c_weight(double* w){
	*w = GRAD_TEST_CONSTANT;
}

void verify_grad(data* d){
	// Make sure to set up a simple environment for testing. Wise to only choose 1 test example, makes costing easy
	// to make things easy, use 2 inputs and 1 output with 1 test vector and assign the following:
	d->t_x[0][0] = 0.5;
	d->t_x[0][1] = 0.3;
	d->t_y[0][0] = 0.6;

	// some ints for iteration (arbitrary naming)
	int i, j, k;
	double run = 0.00001;
	double rise;
	double y1;
	double total_diff = 0.0;
	int num_w = 0;

	// firstly, assign all weights to GRAD_TEST_CONSTANT
	go_through_weights(d, init_single_c_weight);

	// now compute unaltered cost
	y1 = cost(d);

	// now calculate the gradients as back_propogate says they should be. MAKE SURE AN EXAMPLE IS DEFINED
	b_propogate(d, 0);

	// now we compute aproximate gradients and compare them
	// i_w, h_w, and o_w selected in seperate loops
	// i_w
	for(i = 0; i < H_HEIGHT; i++){
		for(j = 0; j < I_HEIGHT + 1; j++){
			d->i_w[i][j] += run;
			rise = cost(d) - y1;
			printf("Exact: %.10lf | Approx: %.10lf\n", d->i_grad[i][j], rise / run);
			total_diff += (rise / run - d->i_grad[i][j] > 0) ? rise / run - d->i_grad[i][j] : d->i_grad[i][j] - rise / run; //acts as norm function
			d->i_w[i][j] -= run;
			num_w += 1;
		}
	}
	
	// h_w
	for(i = 0; i < NUM_H_LAYERS - 1; i++){
		for(j = 0; j < H_HEIGHT; j++){
			for(k = 0; k < H_HEIGHT + 1; k++){
			}
		}
	}
	
	// o_w
	for(i = 0; i < O_HEIGHT; i++){
		for(j = 0; j < H_HEIGHT + 1; j++){
			d->o_w[i][j] += run;
			rise = cost(d) - y1;
			printf("Exact: %.10lf | Approx: %.10lf\n", d->o_grad[i][j], rise / run);
			total_diff += (rise / run - d->o_grad[i][j] > 0) ? rise / run - d->o_grad[i][j] : d->o_grad[i][j] - rise / run; //acts as norm function
			d->o_w[i][j] -= run;
			num_w += 1;
		}
	}

	printf("\n\nTotal Difference: %.20lf \nAverage Difference: %.20lf", total_diff, total_diff / (double)num_w);

}

void cross_fit(data* d, int num_iters){
	int i, n;
	for(i = 0; i < num_iters; i++){
			fit(d, rand() % NUM_T_VECS);
	}
}

// performs all propogations and descents for 1 particular example of data
void fit(data* d, int example){
	int i = 0;
	double temp_cost = 1; //!= 0.0 (to start the loop)
	while(temp_cost >= 0.0 && i < NUM_DESCENTS){
		b_propogate(d, example); //function includes an f_propogate
		descend(d);
		// output/ test
		// if (i % 1 == 0){ //change the "i % C" constant C to greater numbers for more efficient output
			// printf("\rCost: %.15lf | e.g. weight: %lf", temp_cost = cost(d), d->o_w[0][1]); //includes total cost of all examples
		// }
		i++;
	}
}

// compares all outputs
void compare_outputs(data *d){
	int i, j;
	printf("--------------------Comparing Outputs-------------------");
	printf("\nTotal Cost: %.15lf\n", cost(d));
	printf("\nExamples 1 to %d\n", NUM_T_VECS);
	for(i = 0; i < NUM_T_VECS; i++){
		f_propogate(d, i);
		for(j = 0; j < O_HEIGHT; j++){
			printf("Desired Y%d: %lf Vs. Generated Y%d: %lf\n", j + 1, d->t_y[i][j], j + 1, d->f_out[j]);
		}
	}
	printf("\n");
}

// brother to show_weights
void show_grad(data* d){
	int i, j;
	//input weights
	printf("Input weights: \n");
	for(i = 0; i < H_HEIGHT; i++){
		printf("[-1] --> H[%d] : %lf\n", i + 1, d->i_grad[i][0]);
		for(j = 1; j < I_HEIGHT + 1; j++){
			printf("in[%d] --> H[%d] : %lf\n", j, i + 1, d->i_grad[i][j]);
		}
	}
	
	// for one hidden layer
	if (NUM_H_LAYERS == 1){
	//output weights
		printf("\nOutput Weights: \n");
		for(i = 0; i < O_HEIGHT; i++){
			printf("[-1] --> O[%d] : %lf\n", i + 1, d->o_grad[i][0]);
			for(j = 1; j < H_HEIGHT + 1; j++){
				printf("H[%d] --> O[%d] : %lf\n", j, i + 1, d->o_grad[i][j]);
			}
		}
	}
	else {
		// > 1 hidden layers not implemented yet
	}
}

//subtracts proportion of grad_vector from weights
void descend(data* d){
	int i, j;
	//input weights
	for(i = 0; i < H_HEIGHT; i++){
		for(j = 0; j < I_HEIGHT + 1; j++){
			d->i_w[i][j] -= (DESCENT_FACTOR * d->i_grad[i][j]);
		}
	}
	
	for(i = 0; i < O_HEIGHT; i++){
		for(j = 0; j < H_HEIGHT + 1; j++){
			d->o_w[i][j] -= (DESCENT_FACTOR * d->o_grad[i][j]);
		}
	}

	if(NUM_H_LAYERS > 1){
		// > 1 hidden layers not implemented yet
	}
}

//prints all weight data in list
void show_weights(data* d){
	int i, j;
	//input weights
	printf("Input weights: \n");
	for(i = 0; i < H_HEIGHT; i++){
		printf("[-1] --> H[%d] : %lf\n", i + 1, d->i_w[i][0]);
		for(j = 1; j < I_HEIGHT + 1; j++){
			printf("in[%d] --> H[%d] : %lf\n", j, i + 1, d->i_w[i][j]);
		}
	}
	
	// for one hidden layer
	if (NUM_H_LAYERS == 1){
	//output weights
		printf("\nOutput Weights: \n");
		for(i = 0; i < O_HEIGHT; i++){
			printf("[-1] --> O[%d] : %lf\n", i + 1, d->o_w[i][0]);
			for(j = 1; j < H_HEIGHT + 1; j++){
				printf("H[%d] --> O[%d] : %lf\n", j, i + 1, d->o_w[i][j]);
			}
		}
	}
	else {
		// do ALL hidden layers etc.
	}
}

// non batch style: finding (probably local) minimums for a particular example
void b_propogate(data* d, int example){
	int i, n, h;
	double add;
	f_propogate(d, example);
	
	// all w0 threshold values for output synapses
	for(i = 0; i < O_HEIGHT; i++){
		d->o_grad[i][0] =  d->f_out[i] * (d->t_y[example][i] - d->f_out[i]) * (1.0 - d->f_out[i]);
	}
	
	// all normal hidden - output weights
	for (i = 0; i < O_HEIGHT; i++){
		for (n = 1; n < H_HEIGHT + 1; n++){
			d->o_grad[i][n] = (-1.0) * (d->t_y[example][i] - d->f_out[i]) * d->f_out[i] * (1.0 - d->f_out[i]) * d->h_out[NUM_H_LAYERS - 1][n - 1];
		}
	}
	
	// IF 1 HIDDEN LAYER (SIMPLER ALGORITHM):
	if(NUM_H_LAYERS == 1){
		for(h = 0; h < H_HEIGHT; h++){
			add = 0.0;
			for(n = 0; n < O_HEIGHT; n++){
				add += d->o_w[n][h + 1] * (-1.0) * d->f_out[n] * (1.0 - d->f_out[n]) * (d->t_y[example][n] - d->f_out[n]);
			}
			//threshold (-1) weights
			d->i_grad[h][0] = add * (-1.0) * d->h_out[0][h] * (1.0 - d->h_out[0][h]);
			//all other wights
			for (i = 1; i < I_HEIGHT + 1; i++){
				d->i_grad[h][i] = add * d->h_out[0][h] * (1 - d->h_out[0][h]) * d->t_x[example][i - 1]; 
			}
		}
	}
}

// total cost for all test inputs
double cost(data* d){
	double P = 0.0;
	int i, j;
	for(i = 0; i < NUM_T_VECS; i++){
		f_propogate(d, i);
		for(j = 0; j < O_HEIGHT; j++){ 
			P += (0.5)*(d->t_y[i][j] - d->f_out[j])*(d->t_y[i][j] - d->f_out[j]);
		}
	}
	return P;
}

// -1 <= return_value <= 1 (by normalyzing rand())
double norm_rand(){
	return ((double)rand()/(double)RAND_MAX)*(pow((-1.0), (double)rand()));
}

//0 <= return_value <= 1 (by normalyzing rand())
double pos_norm_rand(){
	return (double)rand()/(double)RAND_MAX;
}

// sigmoid activation function
double sigmoid(double a){
	return 1/(1 + exp(-a));
}

// completes the forward pass through the net
void f_propogate(data* d, int example){
	int i, j ,k;
	double sum;
	
	//1st layer
	for(i = 0; i < H_HEIGHT; i++){
		//thresh weights
		sum = (-1.0)*d->i_w[i][0];
		//normal connections
		for(j = 1; j < I_HEIGHT + 1; j++){
			sum += (d->t_x[example][j - 1]) * (d->i_w[i][j]);
		}
		d->h_out[0][i] = sigmoid(sum);
	}
	
	//2nd and on layers
	if (H_HEIGHT > 1) {
		//go through each hidden layer, starting from the 2nd
		for(i = 0; i < NUM_H_LAYERS - 1; i++){
			//go through each neuron in this layer
			for(j = 0; j < H_HEIGHT; j++){
				 //threshhold weight
				sum = (-1.0)*d->h_w[i][j][0];
				//sum up the previous weights * output
				for(k = 1; k < H_HEIGHT + 1; k++){
					sum += d->h_w[i][j][k] * d->h_out[i][j];
				}
				d->h_out[i + 1][j] = sigmoid(sum);
			}
		}
	}
	
	//output layer
	for(i = 0; i < O_HEIGHT; i++){
		//threshhold weights
		sum = (-1.0)*d->o_w[i][0];
		for(j = 1; j < H_HEIGHT + 1; j++){
			sum += d->h_out[NUM_H_LAYERS - 1][j - 1] * d->o_w[i][j];
		}
		d->f_out[i] = sigmoid(sum);
	}
	
}

// calls an abstract function for each weight pointer argument
void go_through_weights(data* d, void (*func)(double*)){
	//some ints for iteration (arbitrary naming)
	int i, j, k;
	
	//i_w, h_w, and o_w selected in seperate loops
	//i_w
	for(i = 0; i < H_HEIGHT; i++){
		for(j = 0; j < I_HEIGHT + 1; j++){
			func(&(d->i_w[i][j]));
		}
	}
	
	//h_w
	for(i = 0; i < NUM_H_LAYERS - 1; i++){
		for(j = 0; j < H_HEIGHT; j++){
			for(k = 0; k < H_HEIGHT + 1; k++){
				func(&(d->h_w[i][j][k]));
			}
		}
	}
	
	//o_w
	for(i = 0; i < O_HEIGHT; i++){
		for(j = 0; j < H_HEIGHT + 1; j++){
			func(&(d->o_w[i][j]));
		}
	}
}

void init_single_weight(double* w){
	*w = norm_rand();
}

void show_weight(double* w){
	printf("%lf\n", *w);
}

void simple_math_application(data* d){
	/*
	This function can be modified at will. It tests the existing functions and a hand made net 
	by producing test data representing specific mathematical relations- 
	e.g. for 2 inputs and 1 output, the test data might simply be output = addition of both inputs.
	We train it to this data, then see how well it can handle some unseen examples. This is a useful
	test in noting that there are serious limits as to the accuracy of certain applications which we can expect
	these nets to achieve. How does it achieve perfect addition for any number between 0 and 1, if the sigmoid 
	function is non linear? It doesn't.
	*/

	// for the following example (addition) let't try a 2 input - 2 hidden - 1 ooutput net
	// (change the #define constants)

	int i;
	//define some NEW test data
	for(i = 0; i < NUM_T_VECS; i++){
		d->t_x[i][0] = pos_norm_rand()/2;
		d->t_x[i][1] = pos_norm_rand()/2;
		d->t_y[i][0] = make_pos(d->t_x[i][0] - d->t_x[i][1]);
	}

	//learn for this data
	cross_fit(d, NUM_CROSSES);

	//show how well it learned that data
	compare_outputs(d);

	//give it some unseen examples!
	//define some NEW test data
	for(i = 0; i < NUM_T_VECS; i++){
		d->t_x[i][0] = pos_norm_rand()/2;
		d->t_x[i][1] = pos_norm_rand()/2;
		d->t_y[i][0] = make_pos(d->t_x[i][0] - d->t_x[i][1]);
	}

	// and now we see how it performs...
	compare_outputs(d);

	show_weights(d);

	// a little note worth mentioning...
	printf("\nWe performed %d gradient descents which required individual forward and backwards propogation\n", NUM_DESCENTS * NUM_CROSSES * NUM_T_VECS);
}
