#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>


#define K 5
#define IMG 320
#define OUT 316
// float kernel[9] = {1,2,1,0,0,0,-1,-2,-1};
float gauss[25] = {1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1};
float img [IMG*IMG];
float output [OUT*OUT];

int main(int argc, char *argv[]) 
{

    double time;
    
    FILE *fp = fopen(argv[1], "rb");
    FILE *outfile = fopen(argv[2], "wb"); 
    fprintf(stderr, "here = %d\n", atoi(argv[3]));
    // omp_set_num_threads(atoi(argv[3]));
    omp_set_num_threads(5);

	int pnum = omp_get_num_procs();
    fprintf(stderr, "Thread_pnum = %d\n", pnum);
    
    // printf('%d',NUM);
    if(!fp) { 
            perror("無法讀取檔案"); 
            return EXIT_FAILURE; 
    } 


    unsigned char ch; 
    int wcount = 0; 
    while(!feof(fp)) 
    { 
            fread(&ch, sizeof(char), 1, fp);
            img[wcount] = (int)ch;
        //   printf(" %.0f ",img[wcount]);
            
            wcount++; 
    } 



    int i,j, k = 0;
    time= omp_get_wtime() ;
    #pragma omp parallel shared(img,gauss,output)
	{
    #pragma omp for //schedule(dynamic)
		for(int i =0;i<OUT;i++)
        {
            for(int j = 0;j<OUT;j++)
            {
                float sum =0;
                for (int kl= 0;kl<K;kl++)
                {
                    for(int kc=0;kc<K;kc++)
                    {
                        int row = (i*OUT+j)/OUT;
                        int col = (i*OUT+j)%OUT;
                        sum += img[(row+kl)*IMG+col+kc] * gauss[kl*K+kc];
                    }
                }
                if((sum/256)>255){sum=255;}
                else if((sum/256)<0){ sum=0;}
                else{ sum = (sum/256);}
                output[i*OUT+j] = sum;
            }
        }
	}
    fprintf(stderr, "The Execution Time of %d Threads: %.16g s \n", omp_get_num_threads(), omp_get_wtime() - time);


    if(!outfile) {
                puts("檔案輸出失敗"); 
                return 1; 
    }

    for(int w=0;w<OUT*OUT;w++)
    {
            ch = (float)output[w];
            // printf(" %03d ",int(ch));
            fwrite(&ch, sizeof(char), 1, outfile);
    }



    fclose(fp);
    fclose(outfile);
        
    return 0; 
}