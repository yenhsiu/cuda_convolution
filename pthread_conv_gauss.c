#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// #define NUM_THREADS 256
#define K 5
#define IMG 320
#define OUT 316
float gauss[25] = {1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1};
float img [IMG*IMG];
float output [OUT*OUT];


void *runner(void *param); /* the thread */

struct v {
   int where;
};

int main(int argc, char *argv[]) {

   clock_t start, end;
   FILE *fp = fopen(argv[1], "rb");
   FILE *outfile = fopen(argv[2], "wb"); 
   int NUM_THREADS = atoi(argv[3]);
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

 	pthread_t threads[NUM_THREADS];

   int i,t, count = 0;
   start = clock();
   for(i = 0; i < OUT*OUT/NUM_THREADS; i++) {
      for(t=0;t<NUM_THREADS;t++)
      {
         struct v *data = (struct v *) malloc(sizeof(struct v));
         data -> where = i*NUM_THREADS+t;
         pthread_attr_t attr; //Set of thread attributes
         //Get the default attributes
         pthread_attr_init(&attr);
         //Create the thread
         pthread_create(&threads[t],&attr,runner,data);
         //Make sure the parent waits for all thread to complete
      }
      for(t=0;t<NUM_THREADS;t++)
 	         pthread_join(threads[t],NULL);
   }
   end = clock();



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

	double sec = (double) end-start / CLOCKS_PER_SEC;
	printf("\n NUM_THREADS %d    Time used: %.4lf \n",NUM_THREADS, sec/CLOCKS_PER_SEC);

   fclose(fp);
   fclose(outfile);
    
   return 0; 
}

//The thread will begin control in this function
void *runner(void *param) {
   struct v *data = param; 
   // printf(data->where);
   int i,j=0;
   int row = data->where/OUT;
   int col = data->where%OUT;
   float sum = 0; //the counter and sum
   //Row multiplied by column
   for(i = 0; i<K; i++){
      for(j= 0; j<K; j++)
      sum += img[(row+i)*IMG+col+j]* gauss[i*K+j];
   }
   if(sum/256>255){sum=255;}
   else if(sum/256<0){ sum=0;}
   else{ sum = sum/256;}
   output[data->where] = sum;
   //Exit the thread
   pthread_exit(0);
}