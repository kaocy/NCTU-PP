#pragma GCC optimize ("O3")
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#ifndef W
#define W 20                                      // Width
#endif

int main(int argc, char **argv) {
    int L = atoi(argv[1]);                        // Length
    int iteration = atoi(argv[2]);                // Iteration
    srand(atoi(argv[3]));                         // Seed

    float d = (float) random() / RAND_MAX * 0.2;  // Diffusivity
    int *temp = malloc((L + 2) * W * sizeof(int));      // Current temperature
    int *next = malloc((L + 2) * W * sizeof(int));      // Next time step

    // cannot parallelize, since random() will get same number
    for (int i = 1; i <= L; i++) {
        for (int j = 0; j < W; j++) {
            temp[i * W + j] = random() >> 3;
        }
    }

    int size, rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_L = L / size;
    int start = local_L * rank + 1;
    int end = local_L * (rank + 1);

    int count = 0, local_balance = 0;
    while (iteration--) {
        local_balance = 1;
        count++;

        // add two row, for coding convenience
        if (rank == 0) {
            for (int j = 0; j < W; j++) {
                temp[j] = temp[W + j];
            }
        }
        if (rank == size - 1) {
            for (int j = 0; j < W; j++) {
                temp[(L + 1) * W + j] = temp[L * W + j];
            }
        }

        // Compute with up, left, right, down points
        for (int i = start; i <= end; i++) {
            for (int j = 0; j < W; j++) {
                float t = temp[i * W + j] / d;
                t += temp[i * W + j] * -4;
                t += temp[(i - 1) * W + j];
                t += temp[(i + 1) * W + j];
                t += temp[i * W + (j - 1 <  0 ? 0 : j - 1)];
                t += temp[i * W + (j + 1 >= W ? j : j + 1)];
                t *= d;
                next[i * W + j] = t ;
                if (next[i * W + j] != temp[i * W + j]) {
                    local_balance = 0;
                }
            }
        }

        // check if finished
        if (size > 1) {
            if (rank == 0) { // receive local balance, send local_break
                int balance = local_balance;
                int balance_from_other_nodes;
                for (int i = 1; i < size; i++) {
                    MPI_Recv(&balance_from_other_nodes, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
                    balance &= balance_from_other_nodes;
                }

                for (int i = 1; i < size; i++) {
                    MPI_Send(&balance, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
                if (balance)   break;
            }
            else { // send local balance, receive local_break
                int balance;
                MPI_Send(&local_balance, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Recv(&balance, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                if (balance)   break;
            }
        }
        else {
            int balance = local_balance;
            if (balance)    break;
        }

        // if not finished yet
        if (size > 1) {
            // send data from rank 0
            if (rank == 0) {
                MPI_Send(next + end * W, W, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            }
            else if (rank == size - 1) {
                MPI_Recv(next + (start - 1) * W, W, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
            }
            else {
                MPI_Recv(next + (start - 1) * W, W, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(next + end * W, W, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
            }

            // send data from rank (size - 1)
            if (rank == 0) {
                MPI_Recv(next + (end + 1) * W, W, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &status);
            }
            else if (rank == size - 1) {
                MPI_Send(next + start * W, W, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
            }
            else {
                MPI_Recv(next + (end + 1) * W, W, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(next + start * W, W, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
            }
        }

        int *tmp = temp;
        temp = next;
        next = tmp;
    }

    // calculate local minimum
    int local_min = temp[start * W];
    for (int i = start; i <= end; i++) {
        for (int j = 0; j < W; j++) {
            if (temp[i * W + j] < local_min) {
                local_min = temp[i * W + j];
            }
        }
    }

    if (size > 1) {
        if (rank == 0) { // receive local minimum
            int min = local_min;
            int min_from_other_nodes;
            for (int i = 1; i < size; i++) {
                MPI_Recv(&min_from_other_nodes, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
                if (min_from_other_nodes < min) {
                    min = min_from_other_nodes;
                }
            }
            printf("Size: %d*%d, Iteration: %d, Min Temp: %d\n", L, W, count, min);
        }
        else { // send local minimum
            MPI_Send(&local_min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }
    else {
        int min = local_min;
        printf("Size: %d*%d, Iteration: %d, Min Temp: %d\n", L, W, count, min);
    }

    MPI_Finalize();
    return 0;
}
