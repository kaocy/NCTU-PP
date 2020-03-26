#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

typedef unsigned long long ull;

ull total_cpu, total_toss;
ull average_toss, remain_toss;
ull total_in_circle;
pthread_mutex_t mutex;

void *start_toss(void* thread_id) {
    ull id = (ull) thread_id;
    ull num_toss = average_toss;
    ull in_circle = 0;
    double distance_squared, x, y;
    unsigned seed = rand();

    if (id == 0)    num_toss += remain_toss;
    for (ull toss = 0; toss < num_toss; toss++) {
        x = ((double) rand_r(&seed)) / RAND_MAX;
        y = ((double) rand_r(&seed)) / RAND_MAX;
        distance_squared = x * x + y * y;
        if (distance_squared <= 1) in_circle++;
    }
    pthread_mutex_lock(&mutex);
    total_in_circle += in_circle;
    pthread_mutex_unlock(&mutex);

    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2)   exit(-1);

    total_cpu = atoi(argv[1]);
    total_toss = atoi(argv[2]);
    if ((total_cpu < 1) || (total_toss < 0))  exit(-1);

    total_in_circle = 0;
    average_toss = total_toss / total_cpu;
    remain_toss = total_toss % total_cpu;

    pthread_t* threads = (pthread_t*) malloc(sizeof(pthread_t) * total_cpu);
    pthread_mutex_init(&mutex, NULL);
    srand(time(NULL));

    for (ull i = 0; i < total_cpu; i++) {
        pthread_create(&threads[i], NULL, start_toss, (void*) i);
    }
    for (ull i = 0; i < total_cpu; i++) {
        pthread_join(threads[i], NULL);
    }

    double pi_estimate = 4 * total_in_circle / ((double) total_toss);
    printf("%lf\n", pi_estimate);

    pthread_mutex_destroy(&mutex);
    free(threads);

    return 0;
}
