/**********************************************************************
 * DESCRIPTION:
 *   Parallel Concurrent Wave Equation - C with CUDA Version
 *   This program implements the concurrent wave equation
 *********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265

void check_param(void);
void init_line(void);
void update (void);
void printfinal (void);

int nsteps,                 	/* number of time steps */
    tpoints; 	     		    /* total points along string */   
float *values;                  /* values in the end, in host */
float *doldval;                 /* values at time (t-dt), in device */
float *dnewval;                 /* values at time (t), in device */

/**********************************************************************
 *	Checks input values from parameters
 *********************************************************************/
void check_param(void) {
   char tchar[20];

   /* check number of points, number of iterations */
    while ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS)) {
        printf("Enter number of points along vibrating string [%d-%d]: ", MINPOINTS, MAXPOINTS);
        scanf("%s", tchar);
        tpoints = atoi(tchar);
        if ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS))
            printf("Invalid. Please enter value between %d and %d\n", MINPOINTS, MAXPOINTS);
    }
    while ((nsteps < 1) || (nsteps > MAXSTEPS)) {
        printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
        scanf("%s", tchar);
        nsteps = atoi(tchar);
        if ((nsteps < 1) || (nsteps > MAXSTEPS))
            printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
    }

    printf("Using points = %d, steps = %d\n", tpoints, nsteps);
}

/**********************************************************************
 *     Initialize points on line
 *********************************************************************/
__global__ void init_line(float *doldval, float *dnewval, int tpoints) {
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (index > tpoints)    return ;

    /* Calculate initial values based on sine curve */
    float fac = 2.0 * PI;
    float x = (float)(index - 1) / (tpoints - 1);
    doldval[index] = dnewval[index] = sin(fac * x);
}

/**********************************************************************
 *     Update all values along line a specified number of times
 *********************************************************************/
 __global__ void update(float *doldval, float *dnewval, int tpoints, int nsteps) {
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (index > tpoints)    return ;

    /* Update values for each time step */
    for (int i = 1; i <= nsteps; i++) {
        float value;
        /* global endpoints */
        if ((index == 1) || (index  == tpoints))
            value = 0.0;
        else
            value = (2.0 * dnewval[index]) - doldval[index] + (-0.18 * dnewval[index]);

        /* Update old values with new values */
        doldval[index] = dnewval[index];
        dnewval[index] = value;
    }
}

/**********************************************************************
 *     Print final results
 *********************************************************************/
void printfinal() {
    for (int i = 1; i <= tpoints; i++) {
        printf("%6.4f ", values[i]);
        if (i % 10 == 0)  printf("\n");
    }
}

/**********************************************************************
 *	Main program
 *********************************************************************/
int main(int argc, char *argv[]) {
	sscanf(argv[1], "%d", &tpoints);
    sscanf(argv[2], "%d", &nsteps);
    check_param();

    const int block_size = 256;
    int block_num = tpoints / block_size + ((tpoints % block_size) > 0);
    const int array_size = (tpoints + 2) * sizeof(float);

    cudaMalloc(&doldval, array_size);
    cudaMalloc(&dnewval, array_size);

	printf("Initializing points on the line...\n");
    init_line<<<block_num, block_size>>>(doldval, dnewval, tpoints);

	printf("Updating all points for all time steps...\n");
    update<<<block_num, block_size>>>(doldval, dnewval, tpoints, nsteps);

    values = (float*) malloc(array_size);
    cudaMemcpy(values, dnewval, array_size, cudaMemcpyDeviceToHost);

	printf("Printing final results...\n");
	printfinal();
    printf("\nDone.\n\n");
    
    free(values);
    cudaFree(doldval);
    cudaFree(dnewval);
	
	return 0;
}