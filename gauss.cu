#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define NUM_THREADS 256
#define BLOCK_SIZE	16


__global__ static void Gauss(const float* a, int lda, const float* b, int ldb, float* c, int ldc)
{
	extern __device__ float data[];
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
	const int row = idx / ldc;
	const int column = idx % ldc;
	int i,  j;
	
	for(i = 0; i < ldb*ldb; i ++){
		data[i] = b[i];
	}
	__syncthreads();
    // printf("bid %d \n",bid); 
    if(row < ldc && column < ldc) 
	{
		float t = 0;
		for (j = 0;j<ldb*ldb;j++)
		{
			t += a[ lda*row + column + lda*(j/ldc) + j%ldc ] * data[j];	     
		}
        // printf("t %.0f ",t);
        // printf("c %d \t",bid * blockDim.x + tid);
        if(t/256>255){c[bid * blockDim.x + tid]=255;}
        else if(t/256<0){ c[bid * blockDim.x + tid]=0;}
        else{ c[idx] = t/256;}
		// printf(" %.0f ",c[row * ldc + column]);
	}
}


clock_t gaussConv(const float* a, int lda, const float* b, int ldb, float* c, int ldc)
{
	float *ac, *bc, *cc;
	clock_t start, end;

	start = clock();

    cudaMalloc((void**) &ac, sizeof(float) * lda * lda);
	cudaMalloc((void**) &bc, sizeof(float) * ldb * ldb);
	cudaMalloc((void**) &cc, sizeof(float) * ldc * ldc);



    cudaMemcpy2D(ac, sizeof(float) * lda, a, sizeof(float) * lda, sizeof(float) * lda, lda, cudaMemcpyHostToDevice);
	cudaMemcpy2D(bc, sizeof(float) * ldb, b, sizeof(float) * ldb, sizeof(float) * ldb, ldb, cudaMemcpyHostToDevice);
	int blocks = (ldc*ldc) / NUM_THREADS;
    // printf("blocks %d\n ",blocks);
    // Conv<<<blocks, NUM_THREADS>>>(ac, lda, bc, ldb, cc, ldc);
    Gauss<<<blocks, NUM_THREADS,sizeof(float) * ldb*ldb>>>(ac, lda, bc, ldb, cc, ldc);

    //函式名稱<<<block 數目, thread 數目, shared memory 大小>>>(參數...);

    cudaMemcpy2D(c, sizeof(float) * ldc, cc, sizeof(float) * ldc, sizeof(float) *ldc, ldc, cudaMemcpyDeviceToHost);

	cudaFree(ac);
	cudaFree(bc);
	cudaFree(cc);

	end = clock();
	return end-start;
}



bool InitCUDA()
{
	int count;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}

	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}

int main(int argc, char* argv[]) {//name of input file and output file 
    FILE *fp = fopen(argv[1], "rb");
    FILE *outfile = fopen(argv[2], "wb"); 
    if(!fp) { 
        perror("無法讀取檔案"); 
        return EXIT_FAILURE; 
    } 
    char  *filesize = strtok(argv[1], ".");
    // printf("%s\n", filesize); //輸出www

    float *img, *gauss, *output;

	int kl = 5;
    int l = atoi(filesize);
    unsigned char ch; 

    img = (float*) malloc(sizeof(float) * l * l);
	gauss = (float*) malloc(sizeof(float) * kl * kl);
	output = (float*) malloc(sizeof(float) * (l-kl+1) * (l-kl+1));
    // gauss = (float*) malloc(sizeof(float) * 5 * 5);

	gauss[0] = 1;
    gauss[1] = 4;
    gauss[2] = 6;
    gauss[3] = 4;
    gauss[4] = 1;

    gauss[5] = 4;
    gauss[6] = 16;
    gauss[7] = 24;
    gauss[8] = 16;
    gauss[9] = 4;

    gauss[10] = 6;
    gauss[11] = 24;
    gauss[12] = 36;
    gauss[13] = 24;
    gauss[14] = 6;

    gauss[15] = 4;
    gauss[16] = 16;
    gauss[17] = 24;
    gauss[18] = 16;
    gauss[19] = 4;

    gauss[20] = 1;
    gauss[21] = 4;
    gauss[22] = 6;
    gauss[23] = 4;
    gauss[24] = 1;



    int wcount = 0; 
    while(!feof(fp)) 
    { 
        fread(&ch, sizeof(char), 1, fp);
        img[wcount] = int(ch);
        // printf(" %.0f ",img[wcount]);
        
        wcount++; 
        // if(wcount % 320==0) {  // 換行 
            // putchar('\n');
        // } 
    } 
    // putchar('\n');

    // for (int i=0;i<320;i++)
    // {
    //     printf(" %.0f ",img[i]);
    // }

    clock_t time = gaussConv(img, l, gauss, kl, output, l-kl+1);


    if(!outfile) { 
            puts("檔案輸出失敗"); 
            return 1; 
        }

    for(int w=0;w<(l-kl+1) * (l-kl+1);w++)
    {
        ch = float(output[w]);
        // printf(" %03d ",int(ch));
        fwrite(&ch, sizeof(char), 1, outfile);
    }

	double sec = (double) time / CLOCKS_PER_SEC;
	printf("\nTime used: %.4lf   (%.2lf GFLOPS)\n", sec, 2.0 * l * l * l / (sec * 1E9));

    fclose(fp);
    fclose(outfile);

	free(img);
	free(gauss);
	free(output);
    
    return 0; 
} 