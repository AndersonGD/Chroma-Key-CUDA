#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "stb_image.h"
#include "stb_image_write.h"
using namespace std;

struct Image {
	unsigned char* data;
	int* dataCompressed;
	int width;
	int height;
	int nrChannels;
};

//Image host_Image;
//Image* dev_Image;

unsigned char* host_img;
int* host_imgCompressed;
int host_width;
int host_height;
int host_channels;

unsigned char* dev_img;
int* dev_imgCompressed;
int* dev_width;
int* dev_height;
int* dev_channels;

void LoadImage(const char* path, Image* img) {

	img->data = stbi_load(path, &img->width, &img->height, &img->nrChannels, 4);
}

void EmpacotarBits(Image* img) {
	int j = 0;

	//cout << "Empacotando bits de: " << img << endl;

	for (int i = 0; i < img->height * img->width * img->nrChannels; i += img->nrChannels)
	{
		int r = img->data[i];
		int g = img->data[i + 1];
		int b = img->data[i + 2];
		int a = img->data[i + 3];
		int rgba = (r << 24) | (g << 16) | (b << 8) | (a);
		img->dataCompressed[j] = rgba;
		//cout << rgba << endl;
		j++;
	}
}

void DesempacotarBits(Image* img) {

	int j = 0;

	for (int i = 0; i < img->height * img->width; i++)
	{
		int rgba = img->dataCompressed[i];
		int r1 = (rgba >> 24) & 0xff;
		int g1 = (rgba >> 16) & 0xff;
		int b1 = (rgba >> 8) & 0xff;
		int a1 = rgba & 0xff;

		img->data[j] = r1;
		img->data[j + 1] = g1;
		img->data[j + 2] = b1;
		img->data[j + 3] = a1;

		j += img->nrChannels;
	}
}

//#define cudaCheckErrors(msg) \
//    do { \
//        cudaError_t __err = cudaGetLastError(); \
//        if (__err != cudaSuccess) { \
//            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
//                msg, cudaGetErrorString(__err), \
//                __FILE__, __LINE__); \
//            fprintf(stderr, "*** FAILED - ABORTING\n"); \
//            exit(1); \
//        } \
//    } while (0)


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	//if (code != cudaSuccess)
	//{
	fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	//if (abort) exit(code);
	//}
}

float DIST(unsigned char c0, unsigned char c1) {
	int r1 = (c0 >> 16) & 0xff;
	int g1 = (c0 >> 8) & 0xff;
	int b1 = c0 & 0xff;

	int r2 = (c1 >> 16) & 0xff;
	int g2 = (c1 >> 8) & 0xff;
	int b2 = c1 & 0xff;

	int r = r1 - r2;
	int g = g1 - g2;
	int b = b1 - b2;

	return sqrt(r * r + g * g + b * b);
}

unsigned char** ConvertFloatImageToUC(float* img, int width, int height) {
	int r, g, b, a, l;
	l = width * height;
	unsigned int c;
	unsigned char** char_buffer = new unsigned char*[width];
	for (int i = 0; i < width; i++)
	{
		char_buffer[i] = new unsigned char[height];
	}
	int index = 0;
	for (size_t i = 0; i < width; i++)
	{
		for (size_t j = 0; j < height; j++)
		{
			index = i + j * width;
			r = img[index] * 255;
			g = img[index + 1] * 255;
			b = img[index + 2] * 255;
			a = img[index + 3] * 255;
			c = a + (r << 16) + (g << 8) + b;

			char_buffer[i][j] = c;
		}
	}

	return char_buffer;
}

//__global__
//void ChromaKey(Image *img, int r, int g, int b)
//{
//	double red = 0.0, green = 0.0, blue = 0.0;
//	int index;
//	
//	index = threadIdx.x + blockIdx.x * img->width;
//	int pixel = img->dataCompressed[index];
//	int r1 = (pixel >> 24) & 0xff;
//	int g1 = (pixel >> 16) & 0xff;
//	int b1 = (pixel >> 8)  & 0xff;
//	int a1 = pixel & 0xff;
//	
//
//	red = r1 - r;
//	green = g1 - g;
//	blue = b1 - b;
//
//	float dist = sqrt(red * red + green * green + blue * blue);
//
//	if (dist <= 500) {
//		a1 = 0;
//		pixel = (r1 << 24 | g1 << 16 | b1 << 8 | a1);
//		img->dataCompressed[index] = pixel;
//	}
//}

__global__
void ChromaKey(int* img, int w, int h, int r, int g, int b)
{
	double red = 0.0, green = 0.0, blue = 0.0;
	int index;

	index = threadIdx.x + blockIdx.x * w;
	int pixel = img[index];
	int r1 = (pixel >> 24) & 0xff;
	int g1 = (pixel >> 16) & 0xff;
	int b1 = (pixel >> 8) & 0xff;
	int a1 = pixel & 0xff;


	red = r1 - r;
	green = g1 - g;
	blue = b1 - b;

	float dist = sqrt(red * red + green * green + blue * blue);

	if (dist <= 200) {
		a1 = 0;
		pixel = 0;// (r1 << 24 | g1 << 16 | b1 << 8 | a1);
		img[index] = pixel;
	}
}

void normalChroma(float* pixels, int i, int r, int g, int b) {
	double red = 0.0, green = 0.0, blue = 0.0;
	float r1 = pixels[i];
	float g1 = pixels[i + 1];
	float b1 = pixels[i + 2];

	red = r1 - (r / 255);
	green = g1 - (g / 255);
	blue = b1 - (b / 255);

	float dist = sqrt(red * red + green * green + blue * blue);

	if (dist <= 0.85f) {
		pixels[i + 3] = 0.0f;
	}

}

void init(void)
{
	//LoadImage("florest.jpg", &host_Image);
	host_img = stbi_load("colors.jpg", &host_width, &host_height, &host_channels, 4);
	host_channels = 4;

	int n = host_width * host_height * host_channels;

	host_imgCompressed = new int[host_width * host_height];

	int j = 0;
	//cout << "Empacotando bits de: " << img << endl;
	for (int i = 0; i < n; i += host_channels)
	{
		int r = host_img[i];
		int g = host_img[i + 1];
		int b = host_img[i + 2];
		int a = host_img[i + 3];
		int rgba = (r << 24) | (g << 16) | (b << 8) | (a);
		host_imgCompressed[j] = rgba;
		//cout << rgba << endl;
		j++;
	}
	int n2 = host_width * host_height;
	gpuErrchk(cudaMallocManaged(&dev_img, (n * sizeof(unsigned char))));
	gpuErrchk(cudaMallocManaged(&dev_imgCompressed, n2 * sizeof(int)));
	gpuErrchk(cudaMallocManaged(&dev_channels, sizeof(int)));
	gpuErrchk(cudaMallocManaged(&dev_height, sizeof(int)));
	gpuErrchk(cudaMallocManaged(&dev_width, sizeof(int)));

	gpuErrchk(cudaMemcpy(dev_img, host_img, n * sizeof(unsigned char), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_imgCompressed, host_imgCompressed, n2 * sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_channels, &host_channels, sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_height, &host_height, sizeof(int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_width, &host_width, sizeof(int), cudaMemcpyHostToDevice));

	ChromaKey << <*dev_width, *dev_height >> > (dev_imgCompressed, *dev_width, *dev_height, 0, 255, 0);

	gpuErrchk(cudaMemcpy(host_imgCompressed, dev_imgCompressed, n2 * sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	j = 0;

	for (int i = 0; i < host_height * host_width; i++)
	{
		int rgba = host_imgCompressed[i];
		int r1 = (rgba >> 24) & 0xff;
		int g1 = (rgba >> 16) & 0xff;
		int b1 = (rgba >> 8) & 0xff;
		int a1 = rgba & 0xff;

		host_img[j] = r1;
		host_img[j + 1] = g1;
		host_img[j + 2] = b1;
		host_img[j + 3] = a1;

		j += host_channels;
	}
	stbi_write_png("test.png", host_width, host_height, 4, host_img, host_width * 4);


	//EmpacotarBits(&host_Image);

	/*int dSize = host_Image.width * host_Image.height * host_Image.nrChannels;
	int dcSize = host_Image.width * host_Image.height;*/

	//gpuErrchk(cudaMallocManaged(&dev_Image, (dSize + dcSize) * sizeof(Image)));
	//cudaCheckErrors("cudaMalloc1 fail");
	////dev_Image->dataCompressed = host_Image.dataCompressed;

	////gpuErrchk(cudaMemcpy(dev_Image, &host_Image, sizeof(Image), cudaMemcpyHostToDevice));
	//cudaCheckErrors("cudaMalloc1 fail");
	//for (int i = 0; i < dSize; i++) {
	//	cudaMalloc((void**) & (host_Image.data[i]), 300 * sizeof(unsigned char));
	//	//cudaCheckErrors("cudaMalloc2 fail");
	//	cudaMemcpy(&dev_Image->data[i], &host_Image.data[i], sizeof(unsigned char*), cudaMemcpyHostToDevice);
	//	//cudaCheckErrors("cudaMemcpy1 fail");
	//}
	//gpuErrchk(cudaMemcpy(dev_width, &host_width, 512 * sizeof(int), cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(dev_height, &host_height, 512 * sizeof(int), cudaMemcpyHostToDevice));

	//ChromaKey << <dev_Image->width, dev_Image->height >> > (dev_Image, 255, 0, 0);


	/*gpuErrchk(cudaMemcpy(&host_Image, dev_Image, sizeof(Image), cudaMemcpyDeviceToHost));
	DesempacotarBits(&host_Image);
	stbi_write_png("test.png", host_Image.width, host_Image.height, 4, host_Image.data, 0);*/

	/*int l = host_height * host_width *STBI_rgb_alpha;
	for (int i = 0; i < l; i+=4)
	{
	normalChroma(host_ImageData, i, 0, 255, 0);
	}
	*/
	/*cudaMemcpy(host_ImageData, dev_pixels,512 * sizeof(unsigned char), cudaMemcpyDeviceToHost);*/


	//save
	//unsigned char** buffer = ConvertFloatImageToUC(host_ImageData, host_width, host_height);

	//l = host_width * host_height * 4;
	//unsigned char* char_buffer = new unsigned char[l];

	//for (int i = 0; i < host_width * host_height * 4; i += 4) {
	//	char_buffer[i] = host_ImageData[i] * 255;
	//	char_buffer[i+1] = host_ImageData[i + 1] * 255;
	//	char_buffer[i+2] = host_ImageData[i + 2] * 255;
	//	char_buffer[i+3] = host_ImageData[i + 3] * 255;
	//}

	//stbi_write_png("test.png", host_width, host_height, 4, char_buffer, host_width*4);
}



int main(int argc, char** argv)
{

	/*gpuErrchk(cudaMallocManaged(&dev_pixels, 512 * sizeof(unsigned char)));
	gpuErrchk(cudaMallocManaged(&dev_width, 512 * sizeof(int)));
	gpuErrchk(cudaMallocManaged(&dev_height, 512 * sizeof(int)));*/

	init();

	//cudaFree(reg)
	system("PAUSE");

	return 0;
}
