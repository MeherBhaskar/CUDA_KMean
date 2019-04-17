#include <stdio.h>
#include <stdlib.h>
#include <string.h>


//TODO : used sharedvar to have (xi - yi)^2 generated in ||lel

__global__ void updateMeans(float *means, float *entries, int *closestMean, int num_entries, int num_means, int num_attribs)
{
	int id = threadIdx.y;
	int thisMeanCount = 0;
	float temp[100];
	for(int j = 0; j < num_attribs; j++)
	{
		temp[j] = 0;
	}
	for(int i = 0; i < num_entries; i++)
	{
		if(closestMean[i] == id)
		{
			//printf("Entry : %d :: Closest : %d \n", i, id);

			for(int j = 0; j < num_attribs; j++)
			{
				//printf("Adding %f to mean %d, attrib %d \n",entries[i*num_attribs + j],closestMean[i],j);
				temp[j]+=entries[i*num_attribs + j];
			}
			thisMeanCount++;
		}
	}
	
	for(int i = 0 ; i < num_attribs; i++)
	{
		if(thisMeanCount != 0)
		{
			
			means[id*num_attribs + i] = temp[i]/thisMeanCount;
			printf("mean : %d , attrib : %d , newMean : %f, count : %d \n",id, i,means[id*num_attribs + i],thisMeanCount);
		}
	}
}

__global__ void getClosestMeans(float *means, float *entries, int *closestMean, int num_entries, int num_means, int num_attribs, int *flag)
{
	printf("ENTERED!!!!\n");
	int id = threadIdx.y;
	int closestDist = 9999999;
	int closest = -1;
	float currDist = 0;
	flag[0] = 0;
	for(int j = 0; j < num_means ; j++)
	{
		currDist = 0;
		
		for(int i = 0; i < num_attribs; i++)
		{
			currDist+= (means[j*num_attribs + i] - entries[id*num_attribs + i]) * (means[j*num_attribs + i] - entries[id*num_attribs + i]);
		}
		
		printf("Entry %d to mean %d distance : %f \n", id, j, currDist);
		
		if(currDist < closestDist)
		{
			closestDist = currDist;
			closest = j;
		}
	}
	if(closest != closestMean[id])
	{
		flag[0] = 1;
	}
	closestMean[id] = closest; 
}


int main()
{
	//Initial Declarations 
	int num_entries;
	int num_means;
	int num_attribs;

	//Read vals for init declarations
	printf("Enter the number of entries : \n");
	scanf("%d", &num_entries);
	printf("Enter the number of means : \n");
	scanf("%d", &num_means);
	printf("Enter the number of attributes : \n");
	scanf("%d", &num_attribs);

	//Utility declarations
	float means[num_means*num_attribs];
	float entries[num_entries*num_attribs];
	float distances[num_entries*num_means];
	int closestMean[num_entries];

	printf("Enter the entries : \n");
	for(int i = 0; i < num_entries*num_attribs; i++)
	{
		scanf("%f", &entries[i]);
	}

	printf("Enter the initial -- means : \n");
	for(int i = 0; i < num_means*num_attribs; i++)
	{
		scanf("%f", &means[i]);
	}

	dim3 gridCM (1,1);
	dim3 threadCM (1, num_entries);

	dim3 gridUM (1,1);
	dim3 threadUM (1, num_means);

	float *dmeans, *dentries, *ddistances;
	int *dclosestMean;
	float *dtemp;	//for UpdateMeans
	int *dflag;
	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	cudaMalloc((void **)&dmeans, 		sizeof(float)*num_means*num_attribs);
	cudaMalloc((void **)&dentries, 		sizeof(float)*num_entries*num_attribs);
	cudaMalloc((void **)&dclosestMean, 	sizeof(int)*num_entries);
	cudaMalloc((void **)&dtemp,			sizeof(float)*num_attribs);
	cudaMalloc((void **)&dflag, 			sizeof(int));
	int flag[1] = {1};
	cudaMemcpy(dflag, flag, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dmeans, means, sizeof(float)*num_means*num_attribs, cudaMemcpyHostToDevice);
	cudaMemcpy(dentries, entries, sizeof(float)*num_entries*num_attribs, cudaMemcpyHostToDevice);

	while(flag[0] == 1)
	{
		getClosestMeans<<<gridCM, threadCM>>>(dmeans, dentries, dclosestMean, num_entries, num_means, num_attribs, dflag);
		cudaMemcpy(closestMean, dclosestMean, sizeof(int)*num_entries, cudaMemcpyDeviceToHost);

		for(int i = 0; i < num_entries; i++)
			printf("%d -- ", closestMean[i]);
		printf("\n");

		updateMeans<<<gridUM, threadUM>>>(dmeans, dentries, dclosestMean, num_entries,num_means, num_attribs);
		cudaMemcpy(flag, dflag, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(means, dmeans, sizeof(float)*num_means*num_attribs, cudaMemcpyDeviceToHost);
		cudaMemcpy(flag, dflag, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(dflag, flag, sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemcpy(dflag, flag, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dmeans, means, sizeof(float)*num_means*num_attribs, cudaMemcpyHostToDevice);
		cudaMemcpy(dclosestMean, closestMean, sizeof(int)*num_entries, cudaMemcpyDeviceToHost);
	}
		
	cudaMemcpy(means, dmeans, sizeof(float)*num_means*num_attribs, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float elapsedtime;
	cudaEventElapsedTime(&elapsedtime,start,stop);


	for(int i = 0; i < num_means*num_attribs; i++)
	{
		if(i == num_attribs)
			printf("\n");

		printf("%f  ",means[i]);
	}
		printf("\nThe elapsed timer is %f\n", elapsedtime);
}
