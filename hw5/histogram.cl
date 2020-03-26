typedef struct {
    unsigned char R;
    unsigned char G;
    unsigned char B;
    unsigned char align;
} RGB;

__kernel void histogram(__global unsigned int *img_data,
                        __global unsigned int *R,
                        __global unsigned int *G,
                        __global unsigned int *B,
                        unsigned int img_size,
                        unsigned int task_num) {
    int global_id = get_global_id(0);
    int global_size = get_global_size(0);

    for (unsigned int i = 0; i < task_num; i++) {
        int index = global_size * i + global_id;
        if (index >= img_size) break;

        unsigned int data = img_data[index];
        RGB *pixel = (RGB*) &data;
        // R[pixel->R]++;
        // G[pixel->G]++;
        // B[pixel->B]++;
        atomic_inc(&R[pixel->R]);
        atomic_inc(&G[pixel->G]);
        atomic_inc(&B[pixel->B]);
    }
}