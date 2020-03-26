#include <fstream>
#include <iostream>
#include <string>
#include <ios>
#include <cstdlib>
#include <sstream>
#include <CL/cl.h>

typedef struct {
    uint8_t R;
    uint8_t G;
    uint8_t B;
    uint8_t align;
} RGB;

typedef struct {
    bool type;
    uint32_t size;
    uint32_t height;
    uint32_t weight;
    RGB *data;
} Image;

Image *readbmp(const char *filename) {
    std::ifstream bmp(filename, std::ios::binary);
    char header[54];
    bmp.read(header, 54);
    uint32_t size = *(int *)&header[2];
    uint32_t offset = *(int *)&header[10];
    uint32_t w = *(int *)&header[18];
    uint32_t h = *(int *)&header[22];
    uint16_t depth = *(uint16_t *)&header[28];
    if (depth != 24 && depth != 32) {
        printf("we don't suppot depth with %d\n", depth);
        exit(0);
    }
    bmp.seekg(offset, bmp.beg);

    Image *ret = new Image();
    ret->type = 1;
    ret->height = h;
    ret->weight = w;
    ret->size = w * h;
    ret->data = new RGB[w * h]{};
    for (int i = 0; i < ret->size; i++) {
        bmp.read((char *)&ret->data[i], depth / 8);
    }
    bmp.close();
    return ret;
}

int writebmp(const char *filename, Image *img) {
    uint8_t header[54] = {
        0x42,        // identity : B
        0x4d,        // identity : M
        0, 0, 0, 0,  // file size
        0, 0,        // reserved1
        0, 0,        // reserved2
        54, 0, 0, 0, // RGB data offset
        40, 0, 0, 0, // struct BITMAPINFOHEADER size
        0, 0, 0, 0,  // bmp width
        0, 0, 0, 0,  // bmp height
        1, 0,        // planes
        32, 0,       // bit per pixel
        0, 0, 0, 0,  // compression
        0, 0, 0, 0,  // data size
        0, 0, 0, 0,  // h resolution
        0, 0, 0, 0,  // v resolution
        0, 0, 0, 0,  // used colors
        0, 0, 0, 0   // important colors
    };

    // file size
    uint32_t file_size = img->size * 4 + 54;
    header[2] = (unsigned char)(file_size & 0x000000ff);
    header[3] = (file_size >> 8) & 0x000000ff;
    header[4] = (file_size >> 16) & 0x000000ff;
    header[5] = (file_size >> 24) & 0x000000ff;

    // width
    uint32_t width = img->weight;
    header[18] = width & 0x000000ff;
    header[19] = (width >> 8) & 0x000000ff;
    header[20] = (width >> 16) & 0x000000ff;
    header[21] = (width >> 24) & 0x000000ff;

    // height
    uint32_t height = img->height;
    header[22] = height & 0x000000ff;
    header[23] = (height >> 8) & 0x000000ff;
    header[24] = (height >> 16) & 0x000000ff;
    header[25] = (height >> 24) & 0x000000ff;

    std::ofstream fout;
    fout.open(filename, std::ios::binary);
    fout.write((char *)header, 54);
    fout.write((char *)img->data, img->size * 4);
    fout.close();
}

void error_exit(std::string msg, cl_int err) {
    std::cerr << "[Error] " << msg << " : " << err << "\n"; 
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *filename;
    if (argc >= 2) {
        // read kernel function write in ./histogram.cl
        std::ifstream source_file("./histogram.cl", std::ios_base::in);
        std::stringstream ss;
        ss << source_file.rdbuf();
        std::string source_str = ss.str();
        const char *source = source_str.c_str();
        size_t source_len = source_str.size();
        source_file.close();

        int many_img = argc - 1;
        for (int i = 0; i < many_img; i++) {
            filename = argv[i + 1];
            Image *img = readbmp(filename);
            std::cout << img->weight << ":" << img->height << "\n";

            unsigned int R[256];
            unsigned int G[256];
            unsigned int B[256];
            std::fill(R, R + 256, 0);
            std::fill(G, G + 256, 0);
            std::fill(B, B + 256, 0);

            cl_platform_id platform_id;
            cl_device_id device_id;
            cl_context context;
            cl_command_queue command_queue;
            cl_mem args_img_data, args_R, args_G, args_B;
            cl_program program;
            cl_kernel kernel;
            cl_int err_code;

            size_t global_work_size = 1024;
            size_t local_work_size = 64;
            unsigned int task_num = img->size / global_work_size + (img->size % global_work_size != 0);
            
            err_code = clGetPlatformIDs(1, &platform_id, NULL);
            if (err_code != CL_SUCCESS) error_exit("clGetPlatformIDs()", err_code);

            err_code = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
            if (err_code != CL_SUCCESS) error_exit("clGetDeviceIDs()", err_code);

            context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err_code);
            if (err_code != CL_SUCCESS) error_exit("clCreateContext()", err_code);

            command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err_code);
            if (err_code != CL_SUCCESS) error_exit("clCreateCommandQueueWithProperties()", err_code);

            args_img_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(RGB) * img->size, NULL, &err_code);
            if (err_code != CL_SUCCESS) error_exit("clCreateBuffer()", err_code);
            args_R = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(R), NULL, &err_code);
            if (err_code != CL_SUCCESS) error_exit("clCreateBuffer()", err_code);
            args_G = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(G), NULL, &err_code);
            if (err_code != CL_SUCCESS) error_exit("clCreateBuffer()", err_code);
            args_B = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(B), NULL, &err_code);
            if (err_code != CL_SUCCESS) error_exit("clCreateBuffer()", err_code);

            err_code = clEnqueueWriteBuffer(command_queue, args_img_data, CL_TRUE, 0, sizeof(RGB) * img->size, img->data, 0, NULL, NULL);
            if (err_code != CL_SUCCESS) error_exit("clEnqueueWriteBuffer()", err_code);

            program = clCreateProgramWithSource(context, 1, &source, &source_len, &err_code);
            if (err_code != CL_SUCCESS) error_exit("clCreateProgramWithSource()", err_code);

            err_code = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
            if (err_code != CL_SUCCESS) {
                size_t len = 0;
                cl_int err_build;
                err_build = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
                char *buffer = new char[len];
                err_build = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
                std::cerr << buffer << std::endl;
                free(buffer);
                error_exit("clBuildProgram()", err_code);
            }

            kernel = clCreateKernel(program, "histogram", &err_code);
            if (err_code != CL_SUCCESS) error_exit("clCreateKernel()", err_code);

            err_code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &args_img_data);
            if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);
            err_code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &args_R);
            if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);
            err_code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &args_G);
            if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);
            err_code = clSetKernelArg(kernel, 3, sizeof(cl_mem), &args_B);
            if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);
            err_code = clSetKernelArg(kernel, 4, sizeof(unsigned int), &(img->size));
            if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);
            err_code = clSetKernelArg(kernel, 5, sizeof(unsigned int), &task_num);
            if (err_code != CL_SUCCESS) error_exit("clSetKernelArg()", err_code);

            err_code = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL);
            if (err_code != CL_SUCCESS) error_exit("clEnqueueNDRangeKernel()", err_code);

            // read data from device to host
            err_code = clEnqueueReadBuffer(command_queue, args_R, CL_TRUE, 0, sizeof(R), R, 0, NULL, NULL);
            if (err_code != CL_SUCCESS) error_exit("clEnqueueReadBuffer()", err_code);
            err_code = clEnqueueReadBuffer(command_queue, args_G, CL_TRUE, 0, sizeof(G), G, 0, NULL, NULL);
            if (err_code != CL_SUCCESS) error_exit("clEnqueueReadBuffer()", err_code);
            err_code = clEnqueueReadBuffer(command_queue, args_B, CL_TRUE, 0, sizeof(B), B, 0, NULL, NULL);
            if (err_code != CL_SUCCESS) error_exit("clEnqueueReadBuffer()", err_code);

            err_code = clReleaseProgram(program);
            if (err_code != CL_SUCCESS) error_exit("clReleaseProgram()", err_code);
            err_code = clReleaseKernel(kernel);
            if (err_code != CL_SUCCESS) error_exit("clReleaseKernel()", err_code);
            err_code = clReleaseCommandQueue(command_queue);
            if (err_code != CL_SUCCESS) error_exit("clReleaseCommandQueue()", err_code);
            err_code = clReleaseContext(context);
            if (err_code != CL_SUCCESS) error_exit("clReleaseContext()", err_code);
            err_code = clReleaseMemObject(args_img_data);
            if (err_code != CL_SUCCESS) error_exit("clReleaseMemObject()", err_code);
            err_code = clReleaseMemObject(args_R);
            if (err_code != CL_SUCCESS) error_exit("clReleaseMemObject()", err_code);
            err_code = clReleaseMemObject(args_G);
            if (err_code != CL_SUCCESS) error_exit("clReleaseMemObject()", err_code);
            err_code = clReleaseMemObject(args_B);
            if (err_code != CL_SUCCESS) error_exit("clReleaseMemObject()", err_code);

            int max = 0;
            for (int i = 0; i < 256; i++) {
                max = R[i] > max ? R[i] : max;
                max = G[i] > max ? G[i] : max;
                max = B[i] > max ? B[i] : max;
            }

            Image *ret = new Image();
            ret->type = 1;
            ret->height = 256;
            ret->weight = 256;
            ret->size = 256 * 256;
            ret->data = new RGB[256 * 256];

            for (int i = 0; i < ret->height; i++) {
                for (int j = 0; j < 256; j++) {
                    int index = 256 * i + j;
                    ret->data[index].R = 0;
                    if (R[j] * 256 / max > i)   ret->data[index].R = 255;
                    ret->data[index].G = 0;
                    if (G[j] * 256 / max > i)   ret->data[index].G = 255;
                    ret->data[index].B = 0;
                    if (B[j] * 256 / max > i)   ret->data[index].B = 255;
                }
            }

            std::string newfile = "hist_" + std::string(filename); 
            writebmp(newfile.c_str(), ret);
        }
    }
    else {
        printf("Usage: ./hist <img.bmp> [img2.bmp ...]\n");
    }
    return 0;
}