#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h> 
using namespace std ;
//structure to store coordinates
struct coordinates{
	float x;
	float y;
};

//global variables


//finding the nearest city
int nearest(int n, double** d , int index , int visited[])
{
	double min=99999999;
	int min_index;
	for(int i=0; i<n; i++)
	{
		if(visited[i]!=1)
		{		
			if(d[index][i]<=min)
			{
				min=d[index][i];
				min_index=i;
			}
		}
	}
	return min_index;
}
//function to calculate objective function 
double cost(int* temp, double** distances, int cities)
{
	double weight=0;
	for (int i = 0; i < cities-1; ++i)
	{
		weight+= distances[temp[i]][temp[i+1]];
		/* code */
	}
	weight+=distances[0][cities-1];
	// cout<<weight;
	return weight;
}

//random path generation function
double reverse(int i, int j, int* result, double ** distances, double k,int cities)
{
	double cost= k-(distances[result[i]][result[i-1]]+distances[result[j]][result[j+1]]) + (distances[result[i]][result[j+1]]+distances[result[i-1]][result[j]]);
	double temp;
	if(cost<k)
	{
		while(i<j&&i!=0&&j!=cities-1)
		{
			temp=result[i];
			result[i]=result[j];
			result[j]=temp;
			i++;
			j--;
		}
	}
	else
		return k;
	return cost;
}

//Driver function
int main(int argc, char** argv)
{
	string distance_type;
	int cities;

	cin >> distance_type;
	cin >> cities;
	double **distances = new double*[cities];
	for(int i=0; i<cities; i++)
	{
		distances[i]=new double[cities];
	}

	int temp[cities];
	struct coordinates points[cities];
	/////stream input start
	for(int i1=0;i1<cities;i1++)
	{
		cin >> points[i1].x;
		cin >> points[i1].y;
	}
	for(int i1=0;i1<cities;i1++)
	{
		for(int j1=0;j1<cities;j1++)
		{
			double x;
			cin >> x;
			distances[i1][j1] = x;
		}
	}
double f_out;
int f1=0;
for(int l=0;l<cities;l++)
{
	int goo=l;
	int j=goo;
	temp[0]=goo;
	int visited[cities]={0};



	//calling nearest function for NN Algo
	for(int i=1; i<cities; i++)
	{ 
		visited[j]=1;
		j=nearest(cities, distances, j, visited);
		temp[i]=j;
	}
	double k= cost(temp, distances, cities);
	double mod_cost= k;
	double better_cost=k;
if(f1==0)
{
f1=1;
f_out=k;
}
	for (int i = 0; i < 9000000; ++i)
	{

	//random number generation
	int num1=rand()%cities;
	int num2=rand()%cities;
	if((num1>=num2)||(num1==0)||(num2==cities-1))
		continue;
	//done
	//function call for path reversal
	
	mod_cost=reverse(num1, num2, temp, distances,mod_cost,cities); 


 		if(better_cost>mod_cost&&f_out>mod_cost)//out put
		{
			for(int i=0;i<cities  ;i++)
			{
			cout << temp[i]<<" ";
			}
		better_cost=mod_cost;
		
			f_out=better_cost;
cout<<"\n";
		}
	}
}	
	return 0;
}
