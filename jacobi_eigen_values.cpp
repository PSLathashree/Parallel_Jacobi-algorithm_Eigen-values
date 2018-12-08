#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include "mpi.h"
#include "stdio.h"
#include "time.h"
 
#ifndef JACOBI
#define JACOBI

typedef float** Matrix;

void SerialJacobi(Matrix mat, const int n, const float eps);

void ParallelJacobi(Matrix mat, const int n, const float eps);

void PrintMatrix(Matrix mat, const int i);
#endif  
using namespace std;
//Function to initialize the values of matrix
Matrix SquareMatrix(const int n,long double v[]){
	int k=0;
  Matrix matrix = new float*[n];
  for (int i = 0; i < n; i++) {
    matrix[i] = new float[n];
  }
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      matrix[i][j] = v[k];
	k++; }
  }
  return matrix;
}

//Structure for the index of a particuar element
struct Ind2D {
  Ind2D(int _i=0, int _j=0) {
    k = _i;
    j = _j;
  }
  int k;
  int j;
};
//Function to find the index of the largest off-diagonal element
Ind2D find_abs_max(Matrix m, const int n) {
  float max = m[1][0];
  Ind2D max_ind(1, 0);
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      if (j != k && abs(m[j][k]) > abs(max)) {
        max = m[j][k];
        max_ind.j = j;
        max_ind.k = k;
      }
    }
  }
  return max_ind;
}
//Function to find the norm of the matrix
float NormMatrix(Matrix m, const int n) {
  float norm = 0;
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < n; ++k) {
      if (j != k) {
        norm += m[j][k] * m[j][k];
      }
    }
  }
  return std::sqrt(norm);
}
//FUnction to print the actual matrix A
void PrintMatrix(Matrix m, const int n) {
   for ( int i = 0; i < n; i++) {
	printf("ROW %d\n",i+1);
	for(int j=0;j<n;j++)
	{ printf("%f\t ", m[i][j]);
	}
	printf("\n\n");
    }
}
//Function to get the odd and even ranks of threads 
int** get_even_and_odd_ranks(int world_ranksize) {
  int** ranks = new int*[2];
  ranks[0] = new int[world_ranksize / 2];
  ranks[1] = new int[world_ranksize / 2];
  for (int i = 1; i < world_ranksize; ++i) {
      if (i % 2 == 1) {
        ranks[0][(i + 1) / 2 - 1] = i;

      } else {
        ranks[1][(i / 2) - 1] = i;
      }
  }
  return ranks;
}
void ParallelJacobiRotate(Matrix m, int ind_j, int ind_k, const int n) {
   
/* Basic computations of the Jacobi parallel algorithm for calculating eigenvalues ​​of a symmetric matrix
   * The minimum number of processes for this implementation is 3:
   * 0 process calculates the values ​​of the rotation matrix and accumulates data from other processes
   * 1 and 2 processes calculate new diagonal elements
   * 3 and 4 processes calculate new values ​​of strings j and k
   * Other processes also calculate new values ​​of strings j and k.*/
  int rank, world_ranksize;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_ranksize);
  float c, s;
  int j, k;
  float* row_j = new float[n];
  float* row_k = new float[n];
  if (rank == 0) {
    j = ind_j;
    k = ind_k;
    //  Calculation of the angles of the rotation matrix
    if (m[j][j] == m[k][k]) {
      c = cos(M_PI / 4);
      s = sin(M_PI / 4);
    }
    else {
      float tau = (m[j][j] - m[k][k]) / (2 * m[j][k]);
      float t = ((tau > 0) ? 1 : -1) / (abs(tau) + sqrt(1 + tau * tau));
      c = 1 / sqrt(1 + t * t);
      s = c * t;
    }
    //  Copy the j and k lines to the appropriate buffers to send them to other threads.
    for (int i = 0; i < n; ++i) {
      row_j[i] = m[j][i];
      row_k[i] = m[k][i];
    }

  }
  // We send data to all threads
  MPI_Bcast(&j, 1, MPI_INT, 0, MPI_COMM_WORLD);  //  line number j
  MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);  //  line number k
  MPI_Bcast(&c, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);  // cos(theta)
  MPI_Bcast(&s, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);  // sin(theta)
  MPI_Bcast(row_j, n, MPI_FLOAT, 0, MPI_COMM_WORLD);  // line j
  MPI_Bcast(row_k, n, MPI_FLOAT, 0, MPI_COMM_WORLD);  // line k
  //MPI_Barrier(MPI_COMM_WORLD);
  float m_jj, m_kk;
  
  //  We create two groups of processes: one for recalculating the string j, the other for converting the string k
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  // The string group k consists of odd processes.( >= 3)
  // The string group j consists of even processes. ( >= 4)
  int** odd_even_ranks = get_even_and_odd_ranks(world_ranksize); //get the thread numbers
  MPI_Group row_j_group;
  MPI_Group row_k_group;
  const int group_k_size = (world_ranksize - 1) / 2;
  const int group_j_size = group_k_size;
  MPI_Group_incl(world_group, group_j_size, odd_even_ranks[0], &row_j_group);
  MPI_Group_incl(world_group, group_k_size, odd_even_ranks[1], &row_k_group);
  // We create communicators based on groups.
  MPI_Comm row_j_comm;
  MPI_Comm row_k_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, row_j_group, 0, &row_j_comm);
  MPI_Comm_create_group(MPI_COMM_WORLD, row_k_group, 0, &row_k_comm);

  int row_j_rank = -1;
  int row_j_size = -1;
  float* row_j_new = new float[n];
  //  check the existence of the communicator
  if (MPI_COMM_NULL != row_j_comm) {
    MPI_Comm_rank(row_j_comm, &row_j_rank);
    MPI_Comm_size(row_j_comm, &row_j_size);
    //  the portion of the row that one stream recounts from the row_j_comm group
    int size = n / row_j_size;
    //  Allocate memory for buffers
    float* row_j_part = new float[size];
    float* row_k_part = new float[size];
    float* row_j_new_part = new float[size];
    //  We divide k and j rows between processes of row_j_comm group
    MPI_Scatter(row_j, size, MPI_FLOAT, row_j_part, size, MPI_FLOAT, 0, row_j_comm);
    MPI_Scatter(row_k, size, MPI_FLOAT, row_k_part, size, MPI_FLOAT, 0, row_j_comm);
    //  Recount part of the line
    for (int i = 0; i < size; ++i) {
        row_j_new_part[i] = c * row_j_part[i] + s * row_k_part[i];
    }
    // We assemble a new line from the parts in the 0 process in relation to the row_j_comm communicator
    // (3 - with respect to MPI_COMM_WORLD)
    MPI_Gather(row_j_new_part, size, MPI_FLOAT, row_j_new, size, MPI_FLOAT, 0, row_j_comm);
    if (row_j_rank == 0) {
      // We send a new line to the 0 process (in relation to MPI_COMM_WORLD)
      MPI_Send(row_j_new, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    // Freeing memory, group and communicator
    delete[] row_j_new_part;
    delete[] row_k_part;
    delete[] row_j_part;
    MPI_Group_free(&row_j_group);
    MPI_Comm_free(&row_j_comm);
  }

  int row_k_rank = -1;
  int row_k_size = -1;
  float* row_k_new = new float[n];
  if (MPI_COMM_NULL != row_k_comm) {
    MPI_Comm_rank(row_k_comm, &row_k_rank);
    MPI_Comm_size(row_k_comm, &row_k_size);
    int size = n / row_k_size;
    float* row_j_part = new float[size];
    float* row_k_part = new float[size];
    float* row_k_new_part = new float[size];
    MPI_Scatter(row_j, size, MPI_FLOAT, row_j_part, size, MPI_FLOAT, 0, row_k_comm);
    MPI_Scatter(row_k, size, MPI_FLOAT, row_k_part, size, MPI_FLOAT, 0, row_k_comm);
    for (int i = 0; i < size; ++i) {
        row_k_new_part[i] = s * row_j_part[i] - c * row_k_part[i];
    }
    MPI_Gather(row_k_new_part, size, MPI_FLOAT, row_k_new, size, MPI_FLOAT, 0, row_k_comm);
    if (row_k_rank == 0) {
      MPI_Send(row_k_new, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    delete[] row_k_new_part;
    delete[] row_k_part;
    delete[] row_j_part;
    MPI_Group_free(&row_k_group);
    MPI_Comm_free(&row_k_comm);
  }
  //  Modifying the matrix in the main process
  //MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
     
    MPI_Recv(row_j_new, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // k line recalculated
    MPI_Recv(row_k_new, n, MPI_FLOAT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Replace the values ​​of the original matrix
    m[j][k] = (c * c - s * s) * row_j[k] + s * c * (row_k[k] - row_j[j]);
    m[k][j] = m[j][k];
    
    m[j][j] = c * c * row_j[j] + 2 * s * c * row_j[k] + s * s * row_k[k];
    m[k][k] = s * s * row_j[j] - 2 * s * c * row_j[k] + c * c * row_k[k];;
    for (int i = 0; i < n; ++i) {
      if (i != j && i != k) {
        m[j][i] = row_j_new[i];
        m[k][i] = row_k_new[i];
        m[i][j] = m[j][i];
        m[i][k] = m[k][i];
      }
    }
  }
  //freeing memory
  delete[] row_k_new;
  delete[] row_j_new;
  delete[] odd_even_ranks[1];
  delete[] odd_even_ranks[0];
  delete[] odd_even_ranks;
  delete[] row_k;
  delete[] row_j;
}

void ParallelJacobi(Matrix mat, const int n, const float eps) {
  Ind2D ind_max;
  float elapsed_time = 0;
  ind_max = find_abs_max(mat, n);
  float norm = NormMatrix(mat, n);;
  float tol;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    norm = NormMatrix(mat, n);
    tol = eps * norm;
    printf("eps = %f, norm = %f, tol = %f\n",eps, norm, tol);
  }
  MPI_Bcast(&norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&tol, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  while (norm > tol) {
    elapsed_time -= MPI_Wtime();
    ParallelJacobiRotate(mat, ind_max.j, ind_max.k, n);
    if (rank == 0) {
      norm = NormMatrix(mat, n);
      //printf("\nnorm = %f\n", norm);
    }
    elapsed_time += MPI_Wtime();
    MPI_Bcast(&norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    ind_max = find_abs_max(mat, n);
  }
  
}

int main(int argc, char** argv){
  //  the dimension is passed through the command line arguments
  int n = std::atoi(argv[1]);
  float start=0, end=0,a;
  double elapsed_time = 0;
  long double v[n*n];
  int i=0,j=0;

  std::fstream myfile("/home/pslathashree/Desktop/pc/dataset.txt", std::ios_base::in);

     
    while (myfile >> a)
    {    v[i]=a;
        //printf("%f ", a);
	i++;
    }
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  elapsed_time -= MPI_Wtime();
  Matrix m = SquareMatrix(n,v);   
 if (rank == 0) 
 PrintMatrix(m, n);
    	 
  ParallelJacobi(m, n, 1e-5);
  elapsed_time += MPI_Wtime();
  if (rank == 0) {
	printf("\nTHE EIGENVALUES OF THE MATRIX ARE-\n\n");
    for (int i = 0; i < n; ++i) {
      printf("\t X[%d] \t=\t %f \n",i, m[i][i]);
    }
    printf("\n\tDimensionof the matrix = %i\n\tTotal time elapsed = %fsecs\n", n, elapsed_time);
   
  }
  MPI_Finalize();
 
  return 0;

}

